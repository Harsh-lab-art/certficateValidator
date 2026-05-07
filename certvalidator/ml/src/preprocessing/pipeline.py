"""
preprocessing/pipeline.py

Full certificate image preprocessing pipeline.
Handles: deskew · adaptive denoising · contrast normalisation ·
         DPI standardisation · ELA channel generation · border crop

Usage (CLI):
    python -m ml.src.preprocessing.pipeline \
        --input  ml/data/raw \
        --output ml/data/processed \
        --ela    ml/data/ela \
        --workers 4

Usage (API):
    from ml.src.preprocessing.pipeline import CertificatePreprocessor
    pp = CertificatePreprocessor()
    result = pp.process(image_path="cert.jpg")
    # result.processed_image  → np.ndarray  (H, W, 3)  uint8
    # result.ela_image        → np.ndarray  (H, W, 3)  uint8
    # result.metadata         → dict
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import fitz  # PyMuPDF
from rich.console import Console
from rich.progress import track
import typer

console = Console()
app = typer.Typer()

TARGET_DPI = 300
TARGET_LONG_SIDE = 2480   # px at 300 DPI for A4 long side
ELA_QUALITY = 90          # JPEG quality for ELA recompression


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    source_path: str
    processed_image: np.ndarray          # BGR uint8
    ela_image: np.ndarray                # BGR uint8 — forensic ELA channel
    grayscale: np.ndarray                # single channel uint8
    metadata: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Core preprocessor
# ---------------------------------------------------------------------------

class CertificatePreprocessor:
    """
    End-to-end preprocessing for certificate images.

    Steps (in order):
      1. Load  — supports JPG, PNG, TIFF, PDF (first page), BMP
      2. DPI normalise — resize so long side = TARGET_LONG_SIDE
      3. Deskew — detect skew angle via Hough lines, rotate to correct
      4. Border crop — remove white/black borders with content-aware crop
      5. CLAHE denoise — adaptive histogram equalisation + bilateral filter
      6. Contrast normalise — gamma correction to flatten illumination
      7. ELA generation — Error Level Analysis for forgery forensics
      8. Output — returns BGR processed image + ELA image + metadata
    """

    def __init__(
        self,
        target_dpi: int = TARGET_DPI,
        ela_quality: int = ELA_QUALITY,
        deskew: bool = True,
        denoise: bool = True,
        crop_borders: bool = True,
    ):
        self.target_dpi = target_dpi
        self.ela_quality = ela_quality
        self.deskew = deskew
        self.denoise = denoise
        self.crop_borders = crop_borders

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process(self, image_path: str | Path) -> PreprocessResult:
        start = time.time()
        path = Path(image_path)

        try:
            img_bgr = self._load(path)
            original_shape = img_bgr.shape

            img_bgr = self._normalise_dpi(img_bgr)

            if self.deskew:
                img_bgr, angle = self._deskew(img_bgr)
            else:
                angle = 0.0

            if self.crop_borders:
                img_bgr = self._crop_borders(img_bgr)

            if self.denoise:
                img_bgr = self._denoise(img_bgr)

            img_bgr = self._normalise_contrast(img_bgr)

            ela = self._compute_ela(img_bgr)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            metadata = {
                "source": str(path),
                "original_shape": original_shape,
                "processed_shape": img_bgr.shape,
                "skew_angle_deg": round(angle, 3),
                "processing_time_s": round(time.time() - start, 3),
                "target_dpi": self.target_dpi,
            }

            return PreprocessResult(
                source_path=str(path),
                processed_image=img_bgr,
                ela_image=ela,
                grayscale=gray,
                metadata=metadata,
            )

        except Exception as exc:
            return PreprocessResult(
                source_path=str(path),
                processed_image=np.zeros((64, 64, 3), dtype=np.uint8),
                ela_image=np.zeros((64, 64, 3), dtype=np.uint8),
                grayscale=np.zeros((64, 64), dtype=np.uint8),
                success=False,
                error=str(exc),
            )

    def process_bytes(self, data: bytes, suffix: str = ".jpg") -> PreprocessResult:
        """Process raw bytes (for direct API upload handling). Works on Windows and Linux."""
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())
        tmp = tmp_dir / f"_cert_upload{suffix}"
        tmp.write_bytes(data)
        result = self.process(tmp)
        try:
            tmp.unlink()
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------ #
    #  Step 1 — Load                                                       #
    # ------------------------------------------------------------------ #

    def _load(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            # Try PIL for exotic formats (TIFF, HEIC, etc.)
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if img is None:
            raise ValueError(f"Cannot load image: {path}")
        return img

    def _load_pdf(self, path: Path) -> np.ndarray:
        doc = fitz.open(str(path))
        page = doc[0]
        # Render at 300 DPI: scale = 300/72
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8)
        img_array = img_array.reshape(pix.h, pix.w, 3)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------------ #
    #  Step 2 — DPI normalisation                                          #
    # ------------------------------------------------------------------ #

    def _normalise_dpi(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        long_side = max(h, w)
        if long_side == TARGET_LONG_SIDE:
            return img
        scale = TARGET_LONG_SIDE / long_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    # ------------------------------------------------------------------ #
    #  Step 3 — Deskew                                                     #
    # ------------------------------------------------------------------ #

    def _deskew(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Detect skew using probabilistic Hough transform on edge map.
        Returns corrected image and the angle (degrees) that was applied.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Adaptive threshold to binarise
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 4
        )
        # Dilate horizontally to connect text runs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        dilated = cv2.dilate(binary, kernel)

        # Hough lines on the dilated text mask
        lines = cv2.HoughLinesP(
            dilated, rho=1, theta=np.pi / 180,
            threshold=100, minLineLength=img.shape[1] // 4,
            maxLineGap=20
        )

        angle = 0.0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:
                    a = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    # Only consider near-horizontal lines (text baselines)
                    if abs(a) < 15:
                        angles.append(a)
            if angles:
                angle = float(np.median(angles))

        if abs(angle) < 0.2:
            return img, angle

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        # Expand canvas to avoid cropping after rotation
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            img, M, (new_w, new_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, angle

    # ------------------------------------------------------------------ #
    #  Step 4 — Border crop                                               #
    # ------------------------------------------------------------------ #

    def _crop_borders(self, img: np.ndarray) -> np.ndarray:
        """Remove uniform white/black borders while preserving content."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold: separate content from near-white/near-black borders
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        # Add 1% padding so we don't clip content
        pad_x = int(img.shape[1] * 0.01)
        pad_y = int(img.shape[0] * 0.01)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(img.shape[1] - x, w + 2 * pad_x)
        h = min(img.shape[0] - y, h + 2 * pad_y)
        return img[y:y + h, x:x + w]

    # ------------------------------------------------------------------ #
    #  Step 5 — CLAHE denoise                                             #
    # ------------------------------------------------------------------ #

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE on L channel (LAB colour space) + bilateral filter.
        Preserves edges (important for seal detection) while reducing
        scanning noise and compression artefacts.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_ch)

        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

        # Bilateral: d=9, sigmaColor=75, sigmaSpace=75
        denoised = cv2.bilateralFilter(bgr_eq, 9, 75, 75)
        return denoised

    # ------------------------------------------------------------------ #
    #  Step 6 — Contrast normalise                                        #
    # ------------------------------------------------------------------ #

    def _normalise_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Gamma correction to flatten uneven lighting (common in phone-scanned
        certificates). Target mean luminance ~128.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_lum = gray.mean()
        if mean_lum == 0:
            return img
        gamma = math.log(128 / 255) / math.log(mean_lum / 255 + 1e-7)
        gamma = float(np.clip(gamma, 0.4, 2.5))

        lut = np.array([
            min(255, int((i / 255.0) ** gamma * 255))
            for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(img, lut)

    # ------------------------------------------------------------------ #
    #  Step 7 — ELA (Error Level Analysis)                                #
    # ------------------------------------------------------------------ #

    def _compute_ela(self, img: np.ndarray) -> np.ndarray:
        """
        Error Level Analysis — the forensic secret weapon.

        Re-compresses the image at a known JPEG quality and computes the
        per-pixel absolute difference between the original and recompressed
        versions. Regions that have been digitally edited show HIGHER error
        levels because they were recompressed from a different base quality.

        The ELA image is amplified 10× and saved as a separate channel
        that the forgery-detection CNN sees alongside RGB.

        Reference: Krawetz, 2007 — "A Picture's Worth"
        """
        # Convert BGR → PIL RGB for JPEG round-trip
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Compress at target quality
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=self.ela_quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        # Absolute difference
        ela_pil = ImageChops.difference(pil_img, compressed)

        # Amplify: scale so max error = 255
        ela_array = np.array(ela_pil, dtype=np.float32)
        max_val = ela_array.max()
        if max_val > 0:
            ela_array = ela_array / max_val * 255.0
        ela_uint8 = ela_array.astype(np.uint8)

        # Enhance contrast for visualisation
        ela_enhanced = Image.fromarray(ela_uint8)
        ela_enhanced = ImageEnhance.Brightness(ela_enhanced).enhance(10.0)
        ela_enhanced = np.array(ela_enhanced, dtype=np.uint8)

        return cv2.cvtColor(ela_enhanced, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def batch_process(
    input_dir: Path,
    output_dir: Path,
    ela_dir: Optional[Path] = None,
    workers: int = 4,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".pdf", ".bmp"),
) -> dict:
    """
    Process all certificate images in input_dir, write results to output_dir.
    Returns summary statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if ela_dir:
        ela_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = CertificatePreprocessor()
    all_files = [
        f for f in sorted(input_dir.rglob("*"))
        if f.suffix.lower() in extensions and f.is_file()
    ]

    if not all_files:
        console.print(f"[yellow]No images found in {input_dir}[/yellow]")
        return {"total": 0, "success": 0, "failed": 0}

    console.print(f"[cyan]Processing {len(all_files)} certificates...[/cyan]")

    stats = {"total": len(all_files), "success": 0, "failed": 0, "errors": []}

    for img_path in track(all_files, description="Preprocessing"):
        result = preprocessor.process(img_path)

        # Preserve relative structure inside output dir
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel.with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if result.success:
            cv2.imwrite(str(out_path), result.processed_image)
            if ela_dir:
                ela_path = ela_dir / rel.with_suffix(".png")
                ela_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ela_path), result.ela_image)
            stats["success"] += 1
        else:
            stats["failed"] += 1
            stats["errors"].append({"file": str(img_path), "error": result.error})
            console.print(f"[red]FAILED[/red] {img_path.name}: {result.error}")

    console.print(
        f"\n[green]Done.[/green] "
        f"{stats['success']}/{stats['total']} processed, "
        f"{stats['failed']} failed."
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    input: Path = typer.Option(..., help="Directory containing raw certificate images"),
    output: Path = typer.Option(..., help="Directory to write processed images"),
    ela: Optional[Path] = typer.Option(None, help="Directory to write ELA images"),
    workers: int = typer.Option(4, help="Parallel worker count"),
):
    """Preprocess all certificate images in INPUT directory."""
    stats = batch_process(input, output, ela_dir=ela, workers=workers)
    if stats["failed"] > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
