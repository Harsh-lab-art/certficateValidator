"""
ml/src/augmentation/augmentor.py

Certificate-specific augmentation pipeline.

Augmentations are chosen to simulate real-world certificate degradation:
  - Scanning artefacts (noise, slight blur, brightness variation)
  - Photography conditions (perspective distortion, lighting shadows)
  - Physical degradation (paper yellowing, fold lines, watermarks)

Crucially: augmentations are applied IDENTICALLY to both the image
and its paired ELA map so spatial alignment is preserved.

Uses albumentations for speed (CPU-optimised, applied before GPU transfer).
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np


class CertificateAugmentor:
    """
    Applies matched augmentations to (image, ela_image) pairs.

    Call signature:
        img_out, ela_out = augmentor(img_bgr, ela_bgr)

    Both inputs and outputs are np.ndarray uint8 (H, W, 3) in BGR.
    """

    def __init__(self, img_size: Tuple[int, int] = (768, 1088), p: float = 0.8):
        self.img_size = img_size   # (H, W)
        self.p = p

        # Geometric transforms — applied jointly so ELA stays aligned
        self._geo = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.05,
                rotate_limit=2.5,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.6,
            ),
            A.Perspective(
                scale=(0.01, 0.04),
                keep_size=True,
                p=0.3,
            ),
            A.HorizontalFlip(p=0.0),    # certificates are not horizontally symmetric
            A.VerticalFlip(p=0.0),
        ])

        # Photometric transforms — applied to IMAGE only, not ELA
        # (ELA is a differential signal — photometric changes would corrupt it)
        self._photo = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.6,
            ),
            A.GaussNoise(p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.ImageCompression(
                quality_range=(75, 95),
                p=0.3,
            ),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=4,
                p=0.15,
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02,
                p=0.3,
            ),
            # Paper yellowing
            A.ToSepia(p=0.05),
        ])

        # Document-specific degradation
        self._degrade = A.Compose([
            # Fold line simulation
            _FoldLineTransform(p=0.15),
            # Watermark/stamp simulation
            _WatermarkTransform(p=0.10),
        ])

    def __call__(
        self, img_bgr: np.ndarray, ela_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.p:
            return img_bgr, ela_bgr

        # Geometric — same transform for both
        geo_result = self._geo(image=img_bgr, mask=ela_bgr[:, :, 0])
        img_out = geo_result["image"]

        # Reconstruct ELA from the single-channel mask + apply same geo
        ela_geo = self._geo(image=ela_bgr)["image"]

        # Photometric — image only
        img_out = self._photo(image=img_out)["image"]

        # Document degradation — image only
        img_out = self._degrade(image=img_out)["image"]

        # Final resize guard (perspective can slightly change dimensions)
        h, w = self.img_size
        if img_out.shape[:2] != (h, w):
            img_out = cv2.resize(img_out, (w, h), interpolation=cv2.INTER_AREA)
        if ela_geo.shape[:2] != (h, w):
            ela_geo = cv2.resize(ela_geo, (w, h), interpolation=cv2.INTER_AREA)

        return img_out, ela_geo


# ---------------------------------------------------------------------------
# Custom albumentations transforms
# ---------------------------------------------------------------------------

class _FoldLineTransform(A.ImageOnlyTransform):
    """Simulates a horizontal or vertical fold crease in the paper."""

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        out = img.copy().astype(np.float32)

        if random.random() < 0.5:
            # Horizontal fold
            y = random.randint(h // 4, 3 * h // 4)
            thickness = random.randint(2, 6)
            darkness  = random.uniform(0.75, 0.92)
            out[y:y + thickness, :] = out[y:y + thickness, :] * darkness
        else:
            # Vertical fold
            x = random.randint(w // 4, 3 * w // 4)
            thickness = random.randint(2, 5)
            darkness  = random.uniform(0.78, 0.94)
            out[:, x:x + thickness] = out[:, x:x + thickness] * darkness

        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ()


class _WatermarkTransform(A.ImageOnlyTransform):
    """Overlays a semi-transparent circular 'VERIFIED' stamp watermark."""

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        out = img.copy()
        h, w = out.shape[:2]
        r = random.randint(80, 180)
        cx = random.randint(r + 20, w - r - 20)
        cy = random.randint(r + 20, h - r - 20)
        alpha = random.uniform(0.05, 0.18)

        overlay = out.copy()
        color = random.choice([
            (0, 0, 180),    # blue
            (0, 140, 0),    # green
            (180, 0, 0),    # red
        ])
        cv2.circle(overlay, (cx, cy), r, color, 3)
        cv2.circle(overlay, (cx, cy), r - 8, color, 1)
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
        return out

    def get_transform_init_args_names(self):
        return ()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python augmentor.py <image_path>")
        sys.exit(1)

    src = sys.argv[1]
    img = cv2.imread(src)
    assert img is not None, f"Cannot load {src}"

    # Fake ELA (grey) for demo
    ela = np.full_like(img, 128)

    aug = CertificateAugmentor()
    for i in range(4):
        aug_img, aug_ela = aug(img.copy(), ela.copy())
        cv2.imwrite(f"/tmp/aug_{i}.png", aug_img)
        cv2.imwrite(f"/tmp/aug_ela_{i}.png", aug_ela)
        print(f"Saved /tmp/aug_{i}.png and /tmp/aug_ela_{i}.png")
