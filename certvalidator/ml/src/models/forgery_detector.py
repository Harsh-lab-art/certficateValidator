"""
ml/src/models/forgery_detector.py

EfficientNet-B4 based certificate forgery detector.

Architecture:
  - Input: 6-channel tensor (3 RGB + 3 ELA) — the ELA channel is the
    forensic secret weapon; it reveals JPEG recompression artefacts
    left by digital tampering that are invisible to the human eye.
  - Backbone: EfficientNet-B4 pretrained on ImageNet, first Conv2d
    patched to accept 6 channels (RGB weights kept, ELA channels
    initialised to mean of RGB weights so pretrained features transfer).
  - Head: Dropout → Linear(1792, 512) → GELU → Dropout → Linear(512, 2)
  - Output: logits for [genuine, fake]

GradCAM is built in — call model.gradcam(img_tensor) to get a spatial
heatmap showing which certificate regions triggered the forgery signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
import numpy as np
from typing import Optional, Tuple


class ForgeryDetector(nn.Module):
    """
    6-channel EfficientNet-B4 forgery detector.

    Parameters
    ----------
    num_classes  : int   2 (genuine=0, fake=1)
    dropout      : float dropout rate before classifier head
    pretrained   : bool  load ImageNet weights for RGB channels
    freeze_bn    : bool  freeze BatchNorm during early training epochs
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_bn: bool = False,
    ):
        super().__init__()

        # ── Load pretrained EfficientNet-B4 ──────────────────────────────
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b4(weights=weights)

        # ── Patch first Conv2d: 3ch → 6ch ────────────────────────────────
        # Strategy: keep RGB weights, initialise ELA weights as
        # channel-mean of RGB so pretrained spatial features transfer.
        old_conv = base.features[0][0]   # Conv2d(3, 48, kernel_size=3, stride=2)
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            # Copy original RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # ELA channels initialised to mean of RGB weights
            ela_init = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            new_conv.weight[:, 3:, :, :] = ela_init
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        base.features[0][0] = new_conv

        # ── Feature extractor + pooling ───────────────────────────────────
        self.features   = base.features
        self.avgpool    = base.avgpool          # AdaptiveAvgPool2d(1)
        in_features     = base.classifier[1].in_features   # 1792

        # ── Custom classification head ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        # ── GradCAM hooks ─────────────────────────────────────────────────
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

        if freeze_bn:
            self._freeze_bn()

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 6, H, W)  — RGB + ELA concatenated on channel dim
        returns: (B, 2) logits
        """
        feat = self.features(x)
        pooled = self.avgpool(feat)
        flat = torch.flatten(pooled, 1)
        return self.classifier(flat)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (probabilities, predicted_class).
        probabilities shape: (B, 2) — [p_genuine, p_fake]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1)
        return probs, preds

    def forgery_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a scalar in [0, 1] for each image:
          0.0 = definitely genuine
          1.0 = definitely fake
        """
        probs, _ = self.predict(x)
        return probs[:, 1]   # probability of class=1 (fake)

    # ------------------------------------------------------------------ #
    #  GradCAM                                                             #
    # ------------------------------------------------------------------ #

    def _register_hooks(self):
        """Hook into the last feature block to capture gradients + activations."""
        target_layer = self.features[-1]   # last MBConv block

        def fwd_hook(module, inp, out):
            self._activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def gradcam(
        self,
        x: torch.Tensor,
        target_class: int = 1,         # 1 = fake (the interesting class)
        interpolate_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap for a single image.

        Parameters
        ----------
        x               : (1, 6, H, W) input tensor
        target_class    : which class to explain (1 = fake)
        interpolate_size: output spatial size (H, W) — defaults to input size

        Returns
        -------
        heatmap : np.ndarray (H, W) float32 in [0, 1]
                  High values = regions that most activated the forgery detector
        """
        assert x.shape[0] == 1, "GradCAM requires batch_size=1"
        self.eval()

        x = x.requires_grad_(True)
        logits = self.forward(x)

        # Backprop through target class score
        self.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Pool gradients over spatial dims → channel importance weights
        grads = self._gradients          # (1, C, h, w)
        acts  = self._activations        # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()                # (h, w)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Upsample to input resolution
        if interpolate_size is None:
            interpolate_size = (x.shape[2], x.shape[3])

        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(
            cam_tensor,
            size=interpolate_size,
            mode="bilinear",
            align_corners=False,
        )
        return cam_up.squeeze().numpy()

    def gradcam_overlay(
        self,
        x: torch.Tensor,
        original_img_bgr: np.ndarray,
        alpha: float = 0.4,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Returns a BGR image with the GradCAM heatmap overlaid on
        the original certificate image. Ready for display or saving.
        """
        import cv2
        h, w = original_img_bgr.shape[:2]
        cam = self.gradcam(x, target_class=target_class, interpolate_size=(h, w))

        # Apply jet colourmap
        cam_uint8 = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

        # Blend with original
        overlay = cv2.addWeighted(original_img_bgr, 1 - alpha, heatmap, alpha, 0)
        return overlay

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def freeze_backbone(self):
        """Freeze all feature layers, only train the head."""
        for p in self.features.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad_(True)

    def parameter_groups(self, lr_backbone: float, lr_head: float) -> list:
        """
        Return two parameter groups with different learning rates.
        Use this with AdamW for discriminative fine-tuning:
          backbone LR = lr_head / 10  (don't destroy pretrained features)
          head LR     = lr_head
        """
        return [
            {"params": self.features.parameters(),   "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable}

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ForgeryDetector":
        """Load a saved checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(
            num_classes=ckpt.get("num_classes", 2),
            dropout=ckpt.get("dropout", 0.3),
            pretrained=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model


# ── Quick sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = ForgeryDetector(pretrained=False).to(device)
    params = model.count_parameters()
    print(f"Parameters: {params['total']:,} total | {params['trainable']:,} trainable")

    # Forward pass test
    dummy = torch.randn(2, 6, 512, 724).to(device)
    t0 = time.time()
    out = model(dummy)
    print(f"Output shape: {out.shape}  |  time: {(time.time()-t0)*1000:.1f}ms")

    # GradCAM test
    single = torch.randn(1, 6, 512, 724).to(device)
    cam = model.gradcam(single, target_class=1)
    print(f"GradCAM shape: {cam.shape}  |  range [{cam.min():.3f}, {cam.max():.3f}]")
    print("[OK] ForgeryDetector smoke test passed")
