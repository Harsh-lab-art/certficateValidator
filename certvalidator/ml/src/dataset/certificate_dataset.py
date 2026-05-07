"""
ml/src/dataset/certificate_dataset.py

PyTorch Dataset for certificate authenticity detection.

Supports two modes:
  - "forgery"      → binary (0=genuine, 1=fake) for EfficientNet forgery detector
  - "field_layout" → returns image + field-level annotations for LayoutLMv3

Each sample returns a dict with:
  - image        : Tensor (3, H, W) — preprocessed certificate
  - ela          : Tensor (3, H, W) — ELA channel (same spatial dims)
  - combined     : Tensor (6, H, W) — RGB + ELA concatenated (forgery model input)
  - label        : int  (0=genuine, 1=fake)
  - tamper_type  : str
  - metadata     : dict (filename, institution, etc.)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from ml.src.preprocessing.pipeline import CertificatePreprocessor
from ml.src.augmentation.augmentor import CertificateAugmentor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = (768, 1088)   # ~A4 at 96 DPI — balanced quality vs VRAM budget


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CertificateDataset(Dataset):
    """
    Loads processed certificate images from a directory + labels CSV.

    Directory layout expected:
        root/
            genuine/  *.png
            fake/     *.png
            labels.csv  (columns: filename, label, tamper_type, ...)

    If processed_dir is None, preprocessing is applied on-the-fly (slower
    but useful during development). For training always pre-process first.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",         # "train" | "val" | "test"
        img_size: Tuple[int, int] = IMG_SIZE,
        augment: bool = True,
        preprocess_online: bool = False,
        ela_dir: Optional[str | Path] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.preprocess_online = preprocess_online
        self.ela_dir = Path(ela_dir) if ela_dir else None

        self._preprocessor = CertificatePreprocessor() if preprocess_online else None
        self._augmentor = CertificateAugmentor(img_size=img_size) if self.augment else None

        self.samples: List[Dict] = self._load_labels()
        self._split_samples()

        # Normalise to ImageNet stats (transfer learning baseline)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.custom_transform = transform

    # ------------------------------------------------------------------ #
    #  Label loading                                                       #
    # ------------------------------------------------------------------ #

    def _load_labels(self) -> List[Dict]:
        labels_path = self.root / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.csv not found in {self.root}. "
                "Run scripts/generate_synthetic.py first."
            )

        samples = []
        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                img_path = self.root / row["filename"]
                if not img_path.exists():
                    continue
                samples.append({
                    "path": str(img_path),
                    "label": int(row["label"]),
                    "tamper_type": row.get("tamper_type", ""),
                    "student_name": row.get("student_name", ""),
                    "institution": row.get("institution", ""),
                    "degree": row.get("degree", ""),
                    "issue_date": row.get("issue_date", ""),
                    "grade": row.get("grade", ""),
                    "cgpa": float(row.get("cgpa", 0.0)),
                })
        return samples

    def _split_samples(self):
        """Deterministic 70/15/15 split stratified by label."""
        from sklearn.model_selection import train_test_split

        genuine = [s for s in self.samples if s["label"] == 0]
        fake    = [s for s in self.samples if s["label"] == 1]

        def _split(data):
            train, rest = train_test_split(data, test_size=0.30, random_state=42)
            val, test   = train_test_split(rest, test_size=0.50, random_state=42)
            return train, val, test

        g_train, g_val, g_test = _split(genuine)
        f_train, f_val, f_test = _split(fake)

        mapping = {
            "train": g_train + f_train,
            "val":   g_val   + f_val,
            "test":  g_test  + f_test,
        }
        self.samples = mapping[self.split]

    # ------------------------------------------------------------------ #
    #  Dataset interface                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_image(path: Path):
        """Load image from file — supports JPG, PNG, TIFF and PDF (first page)."""
        if path.suffix.lower() == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(path))
                page = doc[0]
                mat = fitz.Matrix(1.0, 1.0)  # 1x = 72 DPI, keeps memory low for training
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                arr = np.frombuffer(pix.samples, dtype=np.uint8)
                arr = arr.reshape(pix.h, pix.w, 3)
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[Dataset] PDF load failed for {path}: {e}")
                return None
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            # Fallback: try PIL for JPEG/TIFF variants cv2 can't handle
            try:
                from PIL import Image as PILImage
                pil = PILImage.open(str(path)).convert("RGB")
                return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                return None
        return img

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        path = Path(sample["path"])

        # --- Load image ---
        if self.preprocess_online:
            result = self._preprocessor.process(path)
            img_bgr = result.processed_image
            ela_bgr = result.ela_image
        else:
            img_bgr = self._load_image(path)
            if img_bgr is None:
                raise RuntimeError(f"Cannot read image: {path}")

            # Load paired ELA if directory provided
            if self.ela_dir:
                # ELA files are stored flat in ela_dir with same filename as processed image
                ela_path = self.ela_dir / path.name
                if not ela_path.exists():
                    # Also try relative path from root
                    try:
                        ela_path = self.ela_dir / path.relative_to(self.root)
                    except ValueError:
                        ela_path = self.ela_dir / path.name
                ela_bgr = cv2.imread(str(ela_path), cv2.IMREAD_COLOR)
                if ela_bgr is None:
                    # Compute on-the-fly as fallback
                    ela_bgr = CertificatePreprocessor()._compute_ela(img_bgr)
            else:
                ela_bgr = CertificatePreprocessor()._compute_ela(img_bgr)

        # --- Resize ---
        img_bgr = cv2.resize(img_bgr, (self.img_size[1], self.img_size[0]),
                             interpolation=cv2.INTER_AREA)
        ela_bgr = cv2.resize(ela_bgr, (self.img_size[1], self.img_size[0]),
                             interpolation=cv2.INTER_AREA)

        # --- Augment ---
        if self.augment and self._augmentor:
            img_bgr, ela_bgr = self._augmentor(img_bgr, ela_bgr)

        # --- BGR → RGB → Tensor ---
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ela_rgb = cv2.cvtColor(ela_bgr, cv2.COLOR_BGR2RGB)

        img_tensor = self.to_tensor(img_rgb)    # (3, H, W)
        ela_tensor = self.to_tensor(ela_rgb)    # (3, H, W)
        combined   = torch.cat([img_tensor, ela_tensor], dim=0)  # (6, H, W)

        if self.custom_transform:
            img_tensor = self.custom_transform(img_tensor)

        return {
            "image":       img_tensor,
            "ela":         ela_tensor,
            "combined":    combined,
            "label":       torch.tensor(sample["label"], dtype=torch.long),
            "tamper_type": sample["tamper_type"],
            "path":        str(path),
            "metadata": {
                "student_name": sample["student_name"],
                "institution":  sample["institution"],
                "degree":       sample["degree"],
                "issue_date":   sample["issue_date"],
                "grade":        sample["grade"],
                "cgpa":         sample["cgpa"],
            },
        }

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for WeightedRandomSampler."""
        labels = [s["label"] for s in self.samples]
        counts = np.bincount(labels)
        weights = 1.0 / (counts + 1e-6)
        sample_weights = torch.tensor([weights[l] for l in labels], dtype=torch.float)
        return sample_weights

    def summary(self) -> str:
        labels = [s["label"] for s in self.samples]
        n_genuine = sum(1 for l in labels if l == 0)
        n_fake    = sum(1 for l in labels if l == 1)
        return (
            f"CertificateDataset [{self.split}] — "
            f"{len(self)} samples | genuine: {n_genuine} | fake: {n_fake}"
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    root: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    img_size: Tuple[int, int] = IMG_SIZE,
    balanced_sampling: bool = True,
    ela_dir: Optional[str | Path] = None,
) -> Dict[str, DataLoader]:
    """
    Returns train / val / test DataLoaders.

    balanced_sampling: use WeightedRandomSampler to handle class imbalance
                       (important when real:fake ratio is uneven).
    """
    loaders = {}

    for split in ("train", "val", "test"):
        ds = CertificateDataset(
            root=root,
            split=split,
            img_size=img_size,
            augment=(split == "train"),
            ela_dir=ela_dir,
        )

        if split == "train" and balanced_sampling:
            weights = ds.class_weights()
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            shuffle = False
        else:
            sampler = None
            shuffle = False

        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

        print(ds.summary())

    return loaders


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "ml/data/synthetic"
    loaders = build_dataloaders(root, batch_size=2, num_workers=0)
    batch = next(iter(loaders["train"]))
    print("combined shape :", batch["combined"].shape)
    print("labels         :", batch["label"])
    print("tamper types   :", batch["tamper_type"])
