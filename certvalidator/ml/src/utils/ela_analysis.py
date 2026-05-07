"""
ml/src/utils/ela_analysis.py

Standalone ELA (Error Level Analysis) utilities used during dataset
exploration and model debugging.

Provides:
  - compute_ela()         pure function, no class needed
  - ela_heatmap()         colour-maps ELA to jet for easy visualisation
  - batch_ela_stats()     aggregate statistics across a directory
  - compare_genuine_fake() side-by-side ELA comparison plot
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance


# ── Core ELA ──────────────────────────────────────────────────────────────

def compute_ela(
    img_bgr: np.ndarray,
    quality: int = 90,
    amplify: float = 10.0,
) -> np.ndarray:
    """
    Compute Error Level Analysis for a BGR image.

    Parameters
    ----------
    img_bgr  : np.ndarray  uint8 BGR image
    quality  : int         JPEG recompression quality (80-95 typical)
    amplify  : float       brightness amplification factor for visualisation

    Returns
    -------
    ela_bgr  : np.ndarray  uint8 BGR ELA image (same shape as input)

    Theory
    ------
    JPEG compression is lossy — each round-trip through JPEG at the same
    quality level converges toward a stable error floor.  Regions that
    have been digitally manipulated and then saved are effectively at
    a DIFFERENT position on that convergence curve than the surrounding
    unmodified areas.  The difference (ELA) exposes those regions as
    bright patches even when they are invisible to the human eye.
    """
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    diff = ImageChops.difference(pil, recompressed)
    arr  = np.array(diff, dtype=np.float32)

    # Normalise to [0, 255]
    mx = arr.max()
    if mx > 0:
        arr = arr / mx * 255.0
    ela_u8 = arr.astype(np.uint8)

    # Amplify for visibility
    ela_pil = ImageEnhance.Brightness(Image.fromarray(ela_u8)).enhance(amplify)
    ela_rgb = np.array(ela_pil, dtype=np.uint8)
    return cv2.cvtColor(ela_rgb, cv2.COLOR_RGB2BGR)


def ela_heatmap(ela_bgr: np.ndarray) -> np.ndarray:
    """
    Convert ELA image to a jet-colourmap heatmap (BGR).
    High error = red/yellow, low error = blue/green.
    Useful for judge demonstrations.
    """
    gray = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap


def overlay_ela(
    original_bgr: np.ndarray,
    ela_bgr: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend ELA heatmap over the original image.
    alpha = heatmap opacity (0.0–1.0).
    """
    hmap = ela_heatmap(ela_bgr)
    h, w = original_bgr.shape[:2]
    hmap = cv2.resize(hmap, (w, h))
    return cv2.addWeighted(original_bgr, 1 - alpha, hmap, alpha, 0)


# ── Batch statistics ──────────────────────────────────────────────────────

def batch_ela_stats(directory: str | Path, quality: int = 90) -> dict:
    """
    Compute aggregate ELA statistics across all images in a directory.

    Returns
    -------
    dict with keys: mean_error, std_error, max_error, file_count
    """
    directory = Path(directory)
    exts = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
    files = [f for f in directory.rglob("*") if f.suffix.lower() in exts]

    if not files:
        return {"mean_error": 0, "std_error": 0, "max_error": 0, "file_count": 0}

    errors = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ela = compute_ela(img, quality=quality, amplify=1.0)
        errors.append(float(np.mean(ela)))

    return {
        "mean_error": float(np.mean(errors)),
        "std_error":  float(np.std(errors)),
        "max_error":  float(np.max(errors)),
        "file_count": len(errors),
    }


# ── Visualisation ─────────────────────────────────────────────────────────

def compare_genuine_fake(
    genuine_path: str | Path,
    fake_path:    str | Path,
    output_path:  Optional[str | Path] = None,
    quality:      int = 90,
) -> plt.Figure:
    """
    Generate a 2×3 comparison figure:
      Row 1: genuine original | ELA | heatmap overlay
      Row 2: fake original    | ELA | heatmap overlay

    This is the key demo plot for the hackathon presentation.
    """
    g_img = cv2.imread(str(genuine_path), cv2.IMREAD_COLOR)
    f_img = cv2.imread(str(fake_path),    cv2.IMREAD_COLOR)

    assert g_img is not None, f"Cannot load {genuine_path}"
    assert f_img is not None, f"Cannot load {fake_path}"

    def _row(img):
        ela  = compute_ela(img, quality=quality)
        over = overlay_ela(img, ela)
        return [
            cv2.cvtColor(img,  cv2.COLOR_BGR2RGB),
            cv2.cvtColor(ela,  cv2.COLOR_BGR2RGB),
            cv2.cvtColor(over, cv2.COLOR_BGR2RGB),
        ]

    g_row = _row(g_img)
    f_row = _row(f_img)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor("#0f172a")

    titles_top = ["Genuine — original", "Genuine — ELA", "Genuine — heatmap overlay"]
    titles_bot = ["Fake — original",    "Fake — ELA",    "Fake — heatmap overlay"]

    for col, (g, f, tt, tb) in enumerate(zip(g_row, f_row, titles_top, titles_bot)):
        for ax, img, title, border_color in [
            (axes[0, col], g, tt, "#4ade80"),
            (axes[1, col], f, tb, "#f87171"),
        ]:
            ax.imshow(img)
            ax.set_title(title, color="#e2e8f0", fontsize=11, pad=8)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)
            ax.set_facecolor("#0f172a")

    plt.tight_layout(pad=1.5)

    if output_path:
        fig.savefig(str(output_path), dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved comparison to {output_path}")

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        genuine, fake = sys.argv[1], sys.argv[2]
        fig = compare_genuine_fake(genuine, fake, output_path="/tmp/ela_comparison.png")
        plt.show()
    elif len(sys.argv) == 2:
        img = cv2.imread(sys.argv[1])
        ela = compute_ela(img)
        cv2.imwrite("/tmp/ela_output.png", ela)
        cv2.imwrite("/tmp/ela_heatmap.png", ela_heatmap(ela))
        cv2.imwrite("/tmp/ela_overlay.png", overlay_ela(img, ela))
        print("Saved to /tmp/ela_output.png, /tmp/ela_heatmap.png, /tmp/ela_overlay.png")
    else:
        print("Usage: python ela_analysis.py <image>")
        print("   or: python ela_analysis.py <genuine_image> <fake_image>")
