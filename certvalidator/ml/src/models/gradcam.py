"""
ml/src/models/gradcam.py

Standalone GradCAM engine for the forgery detector.

Provides:
  - GradCAMEngine      — wraps any model, computes class activation maps
  - generate_report()  — produces the full visual report for a certificate:
                         original | ELA | GradCAM overlay | field boxes

This is the module that generates the heatmap shown in the React dashboard
and Chrome extension popup — the visual that wins hackathon judges.

Usage:
    engine = GradCAMEngine(model, target_layer=model.features[-1])
    cam    = engine.compute(img_tensor, target_class=1)
    overlay= engine.overlay(cam, original_bgr)
    cv2.imwrite("heatmap.png", overlay)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path


class GradCAMEngine:
    """
    Generic GradCAM engine — works with any CNN that has a
    target convolutional layer.

    Hooks into the target layer to capture:
      - Forward activations (feature maps)
      - Backward gradients (importance weights)

    Then computes: CAM = ReLU(Σ_c  α_c · A_c)
    where α_c = global-average-pooled gradient for channel c.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None
        self._handles: list = []
        self._register()

    def _register(self):
        def fwd(m, inp, out):
            self._activations = out.detach().clone()

        def bwd(m, grad_in, grad_out):
            self._gradients = grad_out[0].detach().clone()

        self._handles.append(self.target_layer.register_forward_hook(fwd))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ #

    def compute(
        self,
        x: torch.Tensor,
        target_class: int = 1,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap.

        Parameters
        ----------
        x            : (1, C, H, W) input tensor
        target_class : class to explain (1 = fake / tampered)
        output_size  : (H, W) to upsample to — defaults to input spatial dims

        Returns
        -------
        cam : np.ndarray (H, W) float32 in [0, 1]
        """
        assert x.ndim == 4 and x.shape[0] == 1, "Input must be (1, C, H, W)"
        self.model.eval()

        x = x.float().requires_grad_(True)
        logits = self.model(x)

        self.model.zero_grad()
        logits[0, target_class].backward()

        # α = global average of gradients over spatial dims
        grads   = self._gradients    # (1, C, h, w)
        acts    = self._activations  # (1, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam).squeeze().cpu().numpy()         # (h, w)

        if cam.ndim == 0:   # single-pixel edge case
            cam = np.array([[float(cam)]])

        # Normalise
        mn, mx = cam.min(), cam.max()
        cam = (cam - mn) / (mx - mn + 1e-8)

        # Upsample
        sz = output_size or (x.shape[2], x.shape[3])
        cam_t = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_t, size=sz, mode="bilinear", align_corners=False)
        return cam_up.squeeze().numpy().astype(np.float32)

    def compute_batch(
        self,
        xs: List[torch.Tensor],
        target_class: int = 1,
    ) -> List[np.ndarray]:
        """Compute GradCAM for a list of individual image tensors."""
        return [self.compute(x.unsqueeze(0), target_class) for x in xs]

    # ------------------------------------------------------------------ #

    @staticmethod
    def to_heatmap(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Convert float [0,1] CAM to BGR uint8 heatmap."""
        cam_u8 = (cam * 255).clip(0, 255).astype(np.uint8)
        return cv2.applyColorMap(cam_u8, colormap)

    @staticmethod
    def overlay(
        cam: np.ndarray,
        bgr_image: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend GradCAM heatmap over the original BGR image.

        alpha = 0.0 → only original
        alpha = 1.0 → only heatmap
        """
        h, w = bgr_image.shape[:2]
        # Resize cam to match image
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap = GradCAMEngine.to_heatmap(cam_resized, colormap)
        blended = cv2.addWeighted(bgr_image, 1 - alpha, heatmap, alpha, 0)
        return blended

    @staticmethod
    def find_high_activation_regions(
        cam: np.ndarray,
        threshold: float = 0.6,
        min_area_frac: float = 0.001,
        img_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict]:
        """
        Find bounding boxes of high-activation regions in the CAM.
        Used to populate `tamper_regions` in the API response.

        Returns list of dicts: {x, y, width, height, confidence}
        """
        if img_shape:
            cam = cv2.resize(cam, (img_shape[1], img_shape[0]),
                             interpolation=cv2.INTER_LINEAR)

        binary = (cam >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = cam.shape[:2]
        min_area = int(w * h * min_area_frac)
        regions  = []

        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < min_area:
                continue
            roi_confidence = float(cam[y:y+bh, x:x+bw].mean())
            regions.append({
                "x": int(x), "y": int(y),
                "width": int(bw), "height": int(bh),
                "confidence": round(roi_confidence, 3),
            })

        return sorted(regions, key=lambda r: r["confidence"], reverse=True)


# ── Full report generator ─────────────────────────────────────────────────

def generate_report_image(
    original_bgr: np.ndarray,
    ela_bgr: np.ndarray,
    cam: np.ndarray,
    trust_score: float,
    verdict: str,
    field_scores: Optional[List[Dict]] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate a 3-panel visual report:
      Panel 1: Original certificate
      Panel 2: ELA forensics image
      Panel 3: GradCAM overlay with verdict badge

    Returns a single BGR image.
    If output_path is given, saves to disk.
    This is the image embedded in the PDF report.
    """
    TARGET_H = 600
    PANEL_W  = 700

    def _resize(img, h=TARGET_H):
        ratio = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), h),
                          interpolation=cv2.INTER_AREA)

    p1 = _resize(original_bgr)
    p1 = cv2.resize(p1, (PANEL_W, TARGET_H))

    p2 = _resize(ela_bgr)
    p2 = cv2.resize(p2, (PANEL_W, TARGET_H))

    p3_base = cv2.resize(original_bgr, (PANEL_W, TARGET_H))
    p3_cam  = cv2.resize(cam, (PANEL_W, TARGET_H))
    p3 = GradCAMEngine.overlay(p3_cam, p3_base, alpha=0.5)

    # Add verdict badge to panel 3
    color = (50, 200, 50) if verdict == "GENUINE" else (50, 50, 220)
    score_text = f"{verdict}  {trust_score:.0f}/100"
    cv2.rectangle(p3, (10, 10), (350, 55), (0, 0, 0), -1)
    cv2.rectangle(p3, (10, 10), (350, 55), color, 2)
    cv2.putText(p3, score_text, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    # Panel labels
    for panel, label in [(p1, "Original"), (p2, "ELA forensics"), (p3, "GradCAM")]:
        cv2.putText(panel, label, (10, TARGET_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

    # Separator lines
    sep = np.zeros((TARGET_H, 4, 3), dtype=np.uint8) + 30
    report = np.hstack([p1, sep, p2, sep, p3])

    if output_path:
        cv2.imwrite(str(output_path), report)

    return report


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from ml.src.models.forgery_detector import ForgeryDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ForgeryDetector(pretrained=False).to(device)

    engine = GradCAMEngine(model, target_layer=model.features[-1])
    dummy  = torch.randn(1, 6, 512, 724).to(device)
    cam    = engine.compute(dummy, target_class=1, output_size=(512, 724))

    print(f"CAM shape:  {cam.shape}")
    print(f"CAM range:  [{cam.min():.3f}, {cam.max():.3f}]")

    regions = GradCAMEngine.find_high_activation_regions(cam, threshold=0.6)
    print(f"High-activation regions: {len(regions)}")

    # Test overlay
    fake_bgr = np.zeros((512, 724, 3), dtype=np.uint8) + 200
    overlay  = GradCAMEngine.overlay(cam, fake_bgr)
    print(f"Overlay shape: {overlay.shape}")

    engine.remove()
    print("[OK] GradCAMEngine smoke test passed")
