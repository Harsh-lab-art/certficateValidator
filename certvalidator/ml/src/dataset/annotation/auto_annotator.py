"""
ml/src/dataset/annotation/auto_annotator.py

Automatically generates LayoutLMv3 NER annotation files from
synthetic certificates where we know the ground-truth field values
and positions.

Since we generate certificates programmatically, we know:
  - Exactly what text was rendered at which pixel position
  - Which field each text block belongs to

This gives us free, perfectly-labelled training data without any
manual annotation work — a huge advantage over real-world datasets.

For real certificates (Phase 2 later), annotations can be created
using Label Studio and exported as the same JSON format.

Output per certificate:
  {
    "image_path": "ml/data/synthetic/genuine/cert_00001.png",
    "words":  ["DELHI", "TECHNOLOGICAL", "UNIVERSITY", "Rahul", ...],
    "boxes":  [[x1,y1,x2,y2], ...],        <- normalised to 0-1000
    "labels": ["B-INST", "I-INST", "I-INST", "B-NAME", ...]
  }
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract
from PIL import Image
from rich.console import Console
from rich.progress import track

from ml.src.models.layout.layoutlm_extractor import LABEL2ID

console = Console()


# ── Field keyword patterns for heuristic labelling ───────────────────────
# Used when we can't run the full synthetic pipeline with known positions.

FIELD_PATTERNS = {
    "NAME": [
        r"^[A-Z][a-z]+ [A-Z][a-z]+",          # Title case proper name
    ],
    "INST": [
        r"\b(university|institute|college|school|iit|nit|bits)\b",
    ],
    "DEGREE": [
        r"\b(bachelor|master|doctor|b\.?tech|m\.?tech|b\.?sc|m\.?sc|mba|phd)\b",
    ],
    "DATE": [
        r"\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}",
        r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}",
    ],
    "GRADE": [
        r"\b(a\+|a|b\+|b|first division|distinction|pass with merit)\b",
        r"\bcgpa\s*[:=]?\s*\d+\.\d+",
    ],
    "ROLL": [
        r"\b\d{2}[a-z]{2,5}\d{3,5}\b",
        r"\broll\s*(no|number|#)?\s*[:=]?\s*\w+",
    ],
}


@dataclass
class AnnotationSample:
    image_path: str
    words:  List[str]
    boxes:  List[List[int]]    # normalised [0, 1000]
    labels: List[str]
    cert_id: str
    split: str = "train"       # "train" | "val" | "test"


# ── Auto-annotator ────────────────────────────────────────────────────────

class AutoAnnotator:
    """
    Generates NER annotation JSON files from synthetic certificate images.

    Two modes:
      1. Known positions (synthetic) — uses metadata from generation
      2. Heuristic (real certs) — uses regex patterns on OCR output
    """

    def __init__(self, ocr_lang: str = "eng"):
        self.ocr_lang = ocr_lang

    # ------------------------------------------------------------------ #
    #  From synthetic labels CSV                                           #
    # ------------------------------------------------------------------ #

    def annotate_from_csv(
        self,
        labels_csv: str | Path,
        image_root: str | Path,
        output_dir: str | Path,
        train_ratio: float = 0.75,
        val_ratio: float = 0.15,
    ) -> Dict[str, int]:
        """
        Create NER annotations for all certificates in labels.csv.
        Uses Tesseract to get words+boxes, then matches field values
        to assign BIO tags.

        Returns count dict: {train: N, val: N, test: N}
        """
        labels_csv  = Path(labels_csv)
        image_root  = Path(image_root)
        output_dir  = Path(output_dir)

        for split in ["train", "val", "test"]:
            (output_dir / split).mkdir(parents=True, exist_ok=True)

        # Load labels
        rows = []
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                rows.append(row)

        n = len(rows)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        counts = {"train": 0, "val": 0, "test": 0}

        console.print(f"[cyan]Auto-annotating {n} certificates...[/cyan]")

        for i, row in enumerate(track(rows, description="Annotating")):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"

            img_path = image_root / row["filename"]
            if not img_path.exists():
                continue

            # Known field values from synthetic generation
            known_fields = {
                "NAME":       row.get("student_name", ""),
                "INST":       row.get("institution", ""),
                "DEGREE":     row.get("degree", ""),
                "DATE":       row.get("issue_date", ""),
                "GRADE":      row.get("grade", ""),
            }

            sample = self._annotate_image(
                img_path=str(img_path),
                known_fields=known_fields,
                cert_id=img_path.stem,
                split=split,
            )

            if sample and len(sample.words) > 5:
                self._save_sample(sample, output_dir)
                counts[split] += 1

        total = sum(counts.values())
        console.print(
            f"[green]Done.[/green] {total} annotations: "
            f"train={counts['train']} val={counts['val']} test={counts['test']}"
        )
        return counts

    # ------------------------------------------------------------------ #
    #  Core annotation logic                                               #
    # ------------------------------------------------------------------ #

    def _annotate_image(
        self,
        img_path: str,
        known_fields: Dict[str, str],
        cert_id: str,
        split: str,
    ) -> Optional[AnnotationSample]:
        """Run Tesseract on image, assign BIO labels via field matching."""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                return None
            h, w = img.shape[:2]

            pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            data = pytesseract.image_to_data(
                pil,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 6",
            )

            words, boxes, labels = [], [], []
            for i, word in enumerate(data["text"]):
                word = word.strip()
                if not word or int(data["conf"][i]) < 10:
                    continue

                x, y = data["left"][i], data["top"][i]
                bw    = data["width"][i]
                bh    = data["height"][i]

                # Normalise box to [0, 1000]
                box = [
                    int(x / w * 1000),
                    int(y / h * 1000),
                    int((x + bw) / w * 1000),
                    int((y + bh) / h * 1000),
                ]

                label = self._match_label(word, i, data["text"], known_fields)
                words.append(word)
                boxes.append(box)
                labels.append(label)

            return AnnotationSample(
                image_path=img_path,
                words=words,
                boxes=boxes,
                labels=labels,
                cert_id=cert_id,
                split=split,
            )
        except Exception as e:
            console.print(f"[red]Error annotating {img_path}: {e}[/red]")
            return None

    def _match_label(
        self,
        word: str,
        word_idx: int,
        all_words: List[str],
        known_fields: Dict[str, str],
    ) -> str:
        """
        Assign a BIO label to a word by checking if it's part of
        a known field value.

        Strategy:
          1. Build the 3-gram context around the word
          2. Check if the context appears in any field value
          3. If it's the first word of the field span → B-FIELD
          4. If it's a continuation → I-FIELD
          5. Otherwise → O
        """
        word_lower = word.lower()

        for field_type, field_value in known_fields.items():
            if not field_value:
                continue
            fv_words = field_value.lower().split()
            if not fv_words:
                continue

            # Check if this word appears in the field value
            if word_lower not in fv_words and word_lower not in field_value.lower():
                continue

            # Determine B vs I
            if word_lower == fv_words[0] or field_value.lower().startswith(word_lower):
                return f"B-{field_type}"

            # Check if previous word was part of same field
            if word_idx > 0:
                prev_word = all_words[word_idx - 1].strip().lower()
                if prev_word in fv_words:
                    return f"I-{field_type}"

        return "O"

    # ------------------------------------------------------------------ #
    #  Save / load                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _save_sample(sample: AnnotationSample, output_dir: Path):
        out_path = output_dir / sample.split / f"{sample.cert_id}.json"
        with open(out_path, "w") as f:
            json.dump({
                "image_path": sample.image_path,
                "words":      sample.words,
                "boxes":      sample.boxes,
                "labels":     sample.labels,
                "cert_id":    sample.cert_id,
            }, f)

    @staticmethod
    def load_annotation(path: str | Path) -> dict:
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------ #
    #  OCR crop generator for TrOCR training                              #
    # ------------------------------------------------------------------ #

    def generate_ocr_crops(
        self,
        labels_csv: str | Path,
        image_root: str | Path,
        output_dir: str | Path,
        max_samples: int = 5000,
    ) -> int:
        """
        Generate field-level image crops + text labels for TrOCR fine-tuning.
        Each crop is one certificate field (name, date, etc.).

        Writes output_dir/ocr_labels.csv with columns: image_path, text
        """
        labels_csv = Path(labels_csv)
        image_root = Path(image_root)
        output_dir = Path(output_dir)
        crops_dir  = output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        ocr_rows = []
        # Field → approximate vertical position in our synthetic certs (fraction of height)
        # These are rough estimates based on our rendering positions in generate_synthetic.py
        FIELD_VERTICAL = {
            "institution":   0.14,
            "student_name":  0.38,
            "degree":        0.56,
            "issue_date":    0.82,
            "grade":         0.75,
        }
        FIELD_HEIGHT_FRAC = 0.06   # crop height as fraction of image height

        rows = []
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                rows.append(row)

        console.print(f"[cyan]Generating OCR crops from {len(rows)} certificates...[/cyan]")
        count = 0

        for row in track(rows[:max_samples], description="Cropping"):
            img_path = image_root / row["filename"]
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            for field, v_frac in FIELD_VERTICAL.items():
                text = row.get(field, "").strip()
                if not text:
                    continue

                y1 = max(0, int((v_frac - FIELD_HEIGHT_FRAC / 2) * h))
                y2 = min(h, int((v_frac + FIELD_HEIGHT_FRAC / 2) * h))
                crop = img[y1:y2, :, :]
                if crop.size == 0:
                    continue

                crop_name = f"{img_path.stem}_{field}.png"
                cv2.imwrite(str(crops_dir / crop_name), crop)
                ocr_rows.append({
                    "image_path": str(crops_dir / crop_name),
                    "text": text,
                    "field": field,
                })
                count += 1

        # Write CSV
        csv_out = output_dir / "ocr_labels.csv"
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "text", "field"])
            writer.writeheader()
            writer.writerows(ocr_rows)

        console.print(f"[green]Generated {count} OCR crops → {csv_out}[/green]")
        return count


# ── Annotation statistics ─────────────────────────────────────────────────

def annotation_stats(annotation_dir: str | Path) -> Dict:
    annotation_dir = Path(annotation_dir)
    stats = {}

    for split in ["train", "val", "test"]:
        split_dir = annotation_dir / split
        if not split_dir.exists():
            continue
        files = list(split_dir.glob("*.json"))
        label_counts: Dict[str, int] = {}
        total_words = 0

        for fp in files:
            with open(fp) as f:
                sample = json.load(f)
            for label in sample.get("labels", []):
                label_counts[label] = label_counts.get(label, 0) + 1
                total_words += 1

        stats[split] = {
            "files":       len(files),
            "total_words": total_words,
            "label_dist":  label_counts,
        }

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, typer
    app = typer.Typer()

    @app.command()
    def annotate(
        csv:    Path = typer.Option(...,   help="labels.csv from synthetic generator"),
        images: Path = typer.Option(...,   help="Image root (ml/data/synthetic)"),
        out:    Path = typer.Option(Path("ml/data/annotation"), help="Output dir"),
    ):
        """Auto-annotate synthetic certificates for LayoutLMv3 training."""
        annotator = AutoAnnotator()
        counts    = annotator.annotate_from_csv(csv, images, out)
        console.print(counts)

        stats = annotation_stats(out)
        for split, s in stats.items():
            console.print(f"{split}: {s['files']} files | {s['total_words']} words")

    @app.command()
    def crops(
        csv:    Path = typer.Option(...),
        images: Path = typer.Option(...),
        out:    Path = typer.Option(Path("ml/data/ocr_crops")),
    ):
        """Generate field crops for TrOCR fine-tuning."""
        annotator = AutoAnnotator()
        n = annotator.generate_ocr_crops(csv, images, out)
        console.print(f"[green]{n} crops generated[/green]")

    app()
