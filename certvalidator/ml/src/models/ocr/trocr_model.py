"""
ml/src/models/ocr/trocr_model.py

TrOCR fine-tuning for academic certificate text recognition.

TrOCR (Microsoft) is a transformer-based OCR model that combines a
Vision Transformer (ViT) encoder with a text decoder. It handles both
printed and handwritten text — critical for certificates that mix
printed institution names with handwritten signatures.

We fine-tune the base model on synthetic certificate text crops to
improve accuracy on:
  - Institution names (long, often all-caps)
  - Student names (mixed case, diverse scripts)
  - Dates (various formats: DD Month YYYY, MM/DD/YYYY, etc.)
  - Degree titles (specialised vocabulary)
  - Roll numbers (alphanumeric patterns)
  - Grade / CGPA strings

Architecture:
  Encoder: ViT-Base (pretrained on document images)
  Decoder: RoBERTa-like causal LM
  Input:   384×384 cropped image patch (single field region)
  Output:  token sequence → decoded text string

Usage:
    ocr = CertificateOCR()
    text = ocr.read_field(field_image_bgr)        # single field
    all_text = ocr.read_full(certificate_bgr)     # sliding window over full cert
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from torch.utils.data import Dataset
import evaluate


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class OCRResult:
    text: str
    confidence: float          # mean token probability
    processing_time_ms: float
    source_region: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h


# ── Certificate OCR wrapper ───────────────────────────────────────────────

class CertificateOCR:
    """
    TrOCR-based OCR engine for certificate fields.

    Loads microsoft/trocr-base-printed by default.
    Pass model_path to load a fine-tuned checkpoint instead.

    Parameters
    ----------
    model_path   : local HuggingFace checkpoint dir (fine-tuned)
                   or None to use base pretrained model
    device       : "cuda" | "cpu" | "auto"
    beam_size    : beam search width (4 is a good balance)
    max_length   : max output token length
    """

    BASE_MODEL = "microsoft/trocr-base-printed"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        beam_size: int = 4,
        max_length: int = 128,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.beam_size  = beam_size
        self.max_length = max_length

        src = model_path or self.BASE_MODEL
        self.processor = TrOCRProcessor.from_pretrained(src)
        self.model     = VisionEncoderDecoderModel.from_pretrained(src)
        self.model.to(self.device).eval()

        # Force greedy/beam tokens
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id           = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id           = self.processor.tokenizer.sep_token_id

    # ------------------------------------------------------------------ #
    #  Single-field reading                                               #
    # ------------------------------------------------------------------ #

    def read_field(self, img_bgr: np.ndarray) -> OCRResult:
        """
        Read text from a single cropped certificate field image.

        img_bgr : np.ndarray  BGR uint8, any size — will be resized
        """
        t0  = time.time()
        pil = self._prepare(img_bgr)

        pixel_values = self.processor(
            images=pil, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=self.beam_size,
                max_length=self.max_length,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )[0].strip()

        # Confidence: mean of per-token max probabilities
        confidence = self._compute_confidence(outputs)

        return OCRResult(
            text=text,
            confidence=confidence,
            processing_time_ms=(time.time() - t0) * 1000,
        )

    def read_batch(self, images: List[np.ndarray]) -> List[OCRResult]:
        """Read multiple field images in a single forward pass."""
        t0   = time.time()
        pils = [self._prepare(img) for img in images]

        pixel_values = self.processor(
            images=pils, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                num_beams=self.beam_size,
                max_length=self.max_length,
                output_scores=True,
                return_dict_in_generate=True,
            )

        texts = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        elapsed = (time.time() - t0) * 1000

        return [
            OCRResult(
                text=t.strip(),
                confidence=self._compute_confidence(outputs, idx=i),
                processing_time_ms=elapsed / len(images),
            )
            for i, t in enumerate(texts)
        ]

    # ------------------------------------------------------------------ #
    #  Full certificate text extraction (sliding window)                  #
    # ------------------------------------------------------------------ #

    def read_full_certificate(
        self,
        img_bgr: np.ndarray,
        stride_y: int = 80,
        window_h: int = 160,
    ) -> str:
        """
        Extract all text from a full certificate using horizontal strips.
        Combines strip outputs into a single string.
        Used as fallback when LayoutLM field regions are not available.
        """
        h, w = img_bgr.shape[:2]
        strips, positions = [], []

        y = 0
        while y + window_h <= h:
            strip = img_bgr[y:y + window_h, :, :]
            strips.append(strip)
            positions.append(y)
            y += stride_y

        # Process in batches of 8
        batch_size = 8
        all_text   = []
        for i in range(0, len(strips), batch_size):
            batch   = strips[i:i + batch_size]
            results = self.read_batch(batch)
            for r in results:
                if r.text and r.confidence > 0.3:
                    all_text.append(r.text)

        return "\n".join(all_text)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare(img_bgr: np.ndarray) -> Image.Image:
        """BGR uint8 → PIL RGB, enhance contrast for better OCR."""
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

        # Adaptive threshold for binarisation pre-processing
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Convert back to RGB PIL
        rgb = cv2.cvtColor(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.COLOR_BGR2RGB,
        )
        return Image.fromarray(rgb)

    @staticmethod
    def _compute_confidence(outputs, idx: int = 0) -> float:
        """
        Compute per-sequence confidence as mean of max per-token probs.
        Falls back to 0.5 if scores not available.
        """
        if not hasattr(outputs, "scores") or outputs.scores is None:
            return 0.5
        try:
            probs = [
                torch.softmax(score[idx], dim=-1).max().item()
                for score in outputs.scores
            ]
            return float(np.mean(probs)) if probs else 0.5
        except Exception:
            return 0.5

    def save(self, path: str):
        """Save fine-tuned model and processor to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "CertificateOCR":
        return cls(model_path=path, **kwargs)


# ── Training dataset ──────────────────────────────────────────────────────

class CertificateOCRDataset(Dataset):
    """
    Dataset for fine-tuning TrOCR on certificate field crops.

    Each sample: (field_image_crop, ground_truth_text)

    Directory layout:
        data_dir/
            images/   *.png   (field crop images)
            labels/   *.txt   (one line per image, same stem name)

    Or pass a CSV with columns: image_path, text
    """

    def __init__(
        self,
        data_dir: str | Path,
        processor: TrOCRProcessor,
        max_length: int = 128,
        augment: bool = False,
    ):
        self.processor  = processor
        self.max_length = max_length
        self.augment    = augment
        self.samples    = self._load(Path(data_dir))

    def _load(self, data_dir: Path) -> List[dict]:
        # Try CSV first
        csv_path = data_dir / "ocr_labels.csv"
        if csv_path.exists():
            import csv
            samples = []
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    img_path = Path(row["image_path"])
                    if img_path.exists():
                        samples.append({"image": str(img_path), "text": row["text"]})
            return samples

        # Fall back to images/ + labels/ directory pair
        img_dir = data_dir / "images"
        lbl_dir = data_dir / "labels"
        samples = []
        if img_dir.exists():
            for img_path in sorted(img_dir.glob("*.png")):
                lbl_path = lbl_dir / img_path.with_suffix(".txt").name
                if lbl_path.exists():
                    text = lbl_path.read_text().strip()
                    samples.append({"image": str(img_path), "text": text})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img    = Image.open(sample["image"]).convert("RGB")

        pixel_values = self.processor(
            images=img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            sample["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so loss ignores them
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ── Fine-tuning function ──────────────────────────────────────────────────

def finetune_trocr(
    data_dir: str,
    output_dir: str,
    base_model: str = "microsoft/trocr-base-printed",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 5e-5,
    warmup_steps: int = 500,
):
    """
    Fine-tune TrOCR on certificate field crops.

    data_dir should contain train/ and val/ subdirectories
    each with images/ and labels/ or ocr_labels.csv.
    """
    processor = TrOCRProcessor.from_pretrained(base_model)
    model     = VisionEncoderDecoderModel.from_pretrained(base_model)

    # Config tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.eos_token_id           = processor.tokenizer.sep_token_id
    model.config.max_length             = 128
    model.config.no_repeat_ngram_size   = 3
    model.config.length_penalty         = 2.0
    model.config.num_beams              = 4

    train_ds = CertificateOCRDataset(f"{data_dir}/train", processor, augment=True)
    val_ds   = CertificateOCRDataset(f"{data_dir}/val",   processor, augment=False)

    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        label_ids = pred.label_ids
        pred_ids  = pred.predictions

        # Decode predictions
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": cer}

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        logging_steps=50,
        report_to="mlflow" if True else "none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuned TrOCR saved to {output_dir}")
    return trainer


# ── OCR field crop generator (from synthetic certs) ───────────────────────

def extract_field_crops(
    cert_image_bgr: np.ndarray,
    field_regions: dict,
    output_dir: str | Path,
    cert_id: str,
):
    """
    Given a certificate image and known field bounding boxes
    (from synthetic generation), save individual field crops
    for TrOCR training data.

    field_regions: {field_name: (x, y, w, h)}
    """
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    crops = {}
    for field_name, (x, y, w, h) in field_regions.items():
        crop = cert_image_bgr[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        fname = f"{cert_id}_{field_name}.png"
        cv2.imwrite(str(output_dir / "images" / fname), crop)
        crops[field_name] = fname

    return crops


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("TrOCR module loaded — model download required for full test")
    print("Run: from ml.src.models.ocr.trocr_model import CertificateOCR")
    print("     ocr = CertificateOCR()  # downloads ~400MB on first run")
    print("[OK] trocr_model.py imports cleanly")
