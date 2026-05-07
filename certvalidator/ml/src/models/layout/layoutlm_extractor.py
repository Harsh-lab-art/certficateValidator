"""
ml/src/models/layout/layoutlm_extractor.py

LayoutLMv3-based certificate field extractor.

LayoutLMv3 is a multimodal document understanding model that jointly
models text, layout (bounding boxes), and visual features. It is
state-of-the-art for document NER tasks — exactly what we need to
identify named fields in a certificate.

NER Label schema (BIO tagging):
  O          — not a field
  B-NAME     — beginning of student name
  I-NAME     — inside student name
  B-INST     — beginning of institution name
  I-INST     — inside institution name
  B-DEGREE   — beginning of degree title
  I-DEGREE   — inside degree title
  B-DATE     — issue date (begin)
  I-DATE     — issue date (inside)
  B-GRADE    — grade / CGPA
  I-GRADE    — inside grade
  B-ROLL     — roll number
  I-ROLL     — inside roll number
  B-DISCIPLINE — branch/discipline
  I-DISCIPLINE

Pipeline:
  1. Run OCR (Tesseract / TrOCR) on full certificate → words + bounding boxes
  2. Feed (words, boxes, image) into LayoutLMv3
  3. Decode NER tags → extract field strings with confidence
  4. Return FieldExtractionResult

Usage:
    extractor = FieldExtractor()
    result = extractor.extract(cert_image_bgr)
    print(result.student_name)   # "Rahul Sharma"
    print(result.field_scores)   # {"student_name": 0.97, ...}
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
import pytesseract

# Configure Tesseract path for Windows
import os as _os
_tesseract_paths = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe".format(
        _os.environ.get("USERNAME", "")
    ),
]
for _tp in _tesseract_paths:
    if _os.path.exists(_tp):
        pytesseract.pytesseract.tesseract_cmd = _tp
        break


# ── Label map ─────────────────────────────────────────────────────────────

LABELS = [
    "O",
    "B-NAME", "I-NAME",
    "B-INST", "I-INST",
    "B-DEGREE", "I-DEGREE",
    "B-DATE", "I-DATE",
    "B-GRADE", "I-GRADE",
    "B-ROLL", "I-ROLL",
    "B-DISCIPLINE", "I-DISCIPLINE",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
NUM_LABELS = len(LABELS)


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class FieldExtractionResult:
    student_name:  Optional[str] = None
    institution:   Optional[str] = None
    degree:        Optional[str] = None
    discipline:    Optional[str] = None
    issue_date:    Optional[str] = None
    grade:         Optional[str] = None
    roll_number:   Optional[str] = None
    raw_text:      Optional[str] = None

    # Per-field confidence 0–1 (from token probability)
    field_scores: Dict[str, float] = field(default_factory=dict)

    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "student_name":  self.student_name,
            "institution":   self.institution,
            "degree":        self.degree,
            "discipline":    self.discipline,
            "issue_date":    self.issue_date,
            "grade":         self.grade,
            "roll_number":   self.roll_number,
            "field_scores":  self.field_scores,
        }

    def flagged_fields(self, threshold: float = 0.70) -> List[str]:
        """Fields whose confidence is below threshold — likely anomalies."""
        return [f for f, s in self.field_scores.items() if s < threshold]

    def overall_confidence(self) -> float:
        if not self.field_scores:
            return 0.0
        return float(np.mean(list(self.field_scores.values())))


# ── OCR helper (Tesseract) ────────────────────────────────────────────────

def _tesseract_words_boxes(
    img_bgr: np.ndarray,
    lang: str = "eng",
) -> Tuple[List[str], List[List[int]]]:
    """
    Run Tesseract OCR and return (words, normalised_boxes).
    Boxes are normalised to [0, 1000] as required by LayoutLMv3.
    """
    h, w = img_bgr.shape[:2]
    pil  = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    data = pytesseract.image_to_data(
        pil, lang=lang,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",
    )

    words, boxes = [], []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        conf = int(data["conf"][i])
        if conf < 10:
            continue

        x, y = data["left"][i], data["top"][i]
        bw, bh = data["width"][i], data["height"][i]

        # Normalise to [0, 1000]
        x1 = int(x / w * 1000)
        y1 = int(y / h * 1000)
        x2 = int((x + bw) / w * 1000)
        y2 = int((y + bh) / h * 1000)

        words.append(word)
        boxes.append([x1, y1, x2, y2])

    return words, boxes


# ── Field extractor ───────────────────────────────────────────────────────

class FieldExtractor:
    """
    LayoutLMv3-based certificate field extraction.

    On first call downloads microsoft/layoutlmv3-base (~900MB).
    Pass model_path to load a fine-tuned checkpoint.
    """

    BASE_MODEL = "microsoft/layoutlmv3-base"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        src = model_path or self.BASE_MODEL
        self.processor = LayoutLMv3Processor.from_pretrained(
            src, apply_ocr=False   # we run our own OCR
        )
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            src,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device).eval()
        # Only use LayoutLMv3 NER if a fine-tuned checkpoint was explicitly provided
        self._is_finetuned = model_path is not None

    # ------------------------------------------------------------------ #
    #  Main extraction                                                     #
    # ------------------------------------------------------------------ #

    def extract(self, img_bgr: np.ndarray) -> FieldExtractionResult:
        """
        Run full field extraction on a certificate image.

        Uses a two-stage approach:
          1. Regex/keyword extractor — fast, works well on structured marksheets
          2. LayoutLMv3 NER — used only if fine-tuned checkpoint is available
        """
        t0 = time.time()

        words, boxes = _tesseract_words_boxes(img_bgr)
        raw_text     = " ".join(words)

        if not words:
            return FieldExtractionResult(
                raw_text="",
                processing_time_ms=(time.time() - t0) * 1000
            )

        # Always try regex extractor first — it understands real marksheet layouts
        result = self._regex_extract(raw_text)
        result.raw_text = raw_text

        # Only use LayoutLMv3 if it was actually fine-tuned (has NER weights)
        # Base pretrained model produces garbage on certificates — skip it
        if self._is_finetuned:
            try:
                ner_result = self._layoutlm_extract(img_bgr, words, boxes)
                # Merge: use NER result only for fields regex missed
                for field_key in ["student_name", "institution", "degree",
                                  "discipline", "issue_date", "grade", "roll_number"]:
                    regex_val = getattr(result, field_key)
                    ner_val   = getattr(ner_result, field_key)
                    if not regex_val and ner_val:
                        setattr(result, field_key, ner_val)
                        result.field_scores[field_key] = ner_result.field_scores.get(field_key, 0.5)
            except Exception as e:
                print(f"[FieldExtractor] LayoutLMv3 failed, using regex only: {e}")

        result.processing_time_ms = (time.time() - t0) * 1000
        return result

    @staticmethod
    def _regex_extract(text: str) -> FieldExtractionResult:
        """
        Regex + keyword extractor for Indian academic marksheets and certificates.
        Handles formats from RGPV, DTU, Mumbai University, Anna University, VTU etc.
        """
        result = FieldExtractionResult()
        scores = {}
        t = text  # full OCR text

        # ── Roll Number ───────────────────────────────────────────────────
        # Matches: 0133CS231142, 2021BCS0123, 18CS001, etc.
        roll_patterns = [
            r"(?:roll\s*(?:no|number|#)?[\s:.-]*)([\w\d]{6,20})",
            r"(?:enrollment\s*(?:no|number)?[\s:.-]*)([\w\d]{6,20})",
            r"(?:regd?\.?\s*(?:no|number)?[\s:.-]*)([\w\d]{6,20})",
            r"\b(\d{2,4}[A-Z]{2,5}\d{3,6})\b",   # e.g. 0133CS231142
            r"\b([A-Z]{2,4}\d{2}\d{3,5})\b",       # e.g. CS21001
        ]
        for pat in roll_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                result.roll_number = m.group(1).strip().upper()
                scores["roll_number"] = 0.90
                break
        if not result.roll_number:
            scores["roll_number"] = 0.0

        # ── Student Name ──────────────────────────────────────────────────
        # Look for NAME: or S/D/W/O pattern (Indian marksheets)
        name_patterns = [
            r"(?:name\s*[:\-]\s*)([A-Z][A-Za-z\s]{3,40})(?:\n|$|S/D|D/O|S/O|W/O)",
            r"(?:student\s*name\s*[:\-]\s*)([A-Z][A-Za-z\s]{3,40})",
            r"(?:candidate\s*name\s*[:\-]\s*)([A-Z][A-Za-z\s]{3,40})",
            r"\bNAME\s*:\s*([A-Z][A-Z\s]{3,40})\b",
        ]
        for pat in name_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                name = re.sub(r"\s+", " ", m.group(1)).strip()
                # Reject if it looks like an institution or has too many words
                words_in_name = name.split()
                if 1 < len(words_in_name) <= 5 and not any(
                    kw in name.lower() for kw in
                    ["university", "institute", "college", "technology", "science"]
                ):
                    result.student_name = name.title()
                    scores["student_name"] = 0.85
                    break
        if not result.student_name:
            scores["student_name"] = 0.0

        # ── Institution ───────────────────────────────────────────────────
        inst_patterns = [
            r"(?:instt?\.?\s*[:\-]\s*)([A-Za-z\s&,\.]{5,80}?)(?:\n|SEMESTER|ROLL|$)",
            r"(?:institution\s*[:\-]\s*)([A-Za-z\s&,\.]{5,80}?)(?:\n|$)",
            r"(?:college\s*[:\-]\s*)([A-Za-z\s&,\.]{5,80}?)(?:\n|$)",
            r"(?:university\s*[:\-]\s*)([A-Za-z\s&,\.]{5,80}?)(?:\n|$)",
            # University name in header (all caps line with "university" or "institute")
            r"^([A-Z][A-Za-z\s&,\.]{10,80}(?:University|Institute|College|Technology)[A-Za-z\s,\.]{0,40})$",
        ]
        for pat in inst_patterns:
            m = re.search(pat, t, re.IGNORECASE | re.MULTILINE)
            if m:
                inst = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(".,")
                if len(inst) > 5:
                    result.institution = inst
                    scores["institution"] = 0.80
                    break
        if not result.institution:
            scores["institution"] = 0.0

        # ── Degree ────────────────────────────────────────────────────────
        degree_patterns = [
            r"(Bachelor\s+of\s+(?:Technology|Science|Engineering|Arts|Commerce|Computer\s+Applications)[^\n]{0,60})",
            r"(Master\s+of\s+(?:Technology|Science|Engineering|Business\s+Administration)[^\n]{0,60})",
            r"(Doctor\s+of\s+Philosophy[^\n]{0,40})",
            r"(B\.?\s*Tech[^\n]{0,60})",
            r"(M\.?\s*Tech[^\n]{0,60})",
            r"(B\.?\s*E\.?[^\n]{0,40})",
        ]
        for pat in degree_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                result.degree = re.sub(r"\s+", " ", m.group(1)).strip()
                scores["degree"] = 0.85
                break
        if not result.degree:
            scores["degree"] = 0.0

        # ── Issue Date ────────────────────────────────────────────────────
        date_patterns = [
            r"(?:date\s*[:\-]\s*)(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",
            r"(?:date\s*[:\-]\s*)(\d{1,2}\s+\w+\s+\d{4})",
            r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
            r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b",
            # Exam month/year like "DECEMBER-2025" or "DECEMBER 2025"
            r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)[\s\-]\d{4})\b",
        ]
        for pat in date_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                result.issue_date = m.group(1).strip()
                scores["issue_date"] = 0.85
                break
        if not result.issue_date:
            scores["issue_date"] = 0.0

        # ── Grade / SGPA / CGPA ───────────────────────────────────────────
        grade_patterns = [
            r"(?:sgpa\s*[:\-]?\s*)(\d+\.\d+)",
            r"(?:cgpa\s*[:\-]?\s*)(\d+\.\d+)",
            r"(?:gpa\s*[:\-]?\s*)(\d+\.\d+)",
            r"(?:result\s*[:\-]\s*)(PASS|FAIL|DISTINCTION|FIRST\s+DIVISION|SECOND\s+DIVISION)",
            r"\b(PASS|FAIL|DISTINCTION)\b",
        ]
        for pat in grade_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                result.grade = m.group(1).strip().upper()
                scores["grade"] = 0.85
                break
        if not result.grade:
            scores["grade"] = 0.0

        # ── Discipline / Branch ───────────────────────────────────────────
        discipline_patterns = [
            r"(Computer\s+Science\s*(?:&|and)?\s*Engineering)",
            r"(Electronics\s*(?:&|and)?\s*Communication(?:\s+Engineering)?)",
            r"(Mechanical\s+Engineering)",
            r"(Civil\s+Engineering)",
            r"(Information\s+Technology)",
            r"(Electrical\s+Engineering)",
            r"(Data\s+Science)",
            r"(Artificial\s+Intelligence)",
        ]
        for pat in discipline_patterns:
            m = re.search(pat, t, re.IGNORECASE)
            if m:
                result.discipline = re.sub(r"\s+", " ", m.group(1)).strip()
                scores["discipline"] = 0.80
                break
        if not result.discipline:
            scores["discipline"] = 0.0

        result.field_scores = scores
        return result

    def _layoutlm_extract(
        self,
        img_bgr: np.ndarray,
        words: List[str],
        boxes: List[List[int]],
    ) -> FieldExtractionResult:
        """Run LayoutLMv3 NER — only called when fine-tuned checkpoint exists."""
        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        raw_text = " ".join(words)

        chunk_size = 480
        all_probs, all_labels_decoded = [], []

        for start in range(0, len(words), chunk_size):
            w_chunk = words[start:start + chunk_size]
            b_chunk = boxes[start:start + chunk_size]

            encoding = self.processor(
                pil, w_chunk, boxes=b_chunk,
                return_tensors="pt", truncation=True,
                max_length=512, padding="max_length",
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)

            logits = outputs.logits[0]
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

            try:
                word_ids = encoding.word_ids(batch_index=0)
            except Exception:
                word_ids = list(range(len(w_chunk)))

            word_label_probs: Dict[int, List] = {}
            for tok_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                word_label_probs.setdefault(word_idx, []).append(probs[tok_idx])

            for word_idx in sorted(word_label_probs):
                mean_prob = np.mean(word_label_probs[word_idx], axis=0)
                label_id  = int(mean_prob.argmax())
                all_labels_decoded.append(ID2LABEL[label_id])
                all_probs.append(float(mean_prob.max()))

        return self._decode_spans(
            words=words[:len(all_labels_decoded)],
            labels=all_labels_decoded,
            probs=all_probs,
            raw_text=raw_text,
        )

    # ------------------------------------------------------------------ #
    #  Span decoder                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decode_spans(
        words: List[str],
        labels: List[str],
        probs: List[float],
        raw_text: str,
    ) -> FieldExtractionResult:
        """
        Convert BIO tag sequence → field strings.
        Groups consecutive B-*/I-* tokens into a single span.
        """
        field_map = {
            "NAME":       "student_name",
            "INST":       "institution",
            "DEGREE":     "degree",
            "DISCIPLINE": "discipline",
            "DATE":       "issue_date",
            "GRADE":      "grade",
            "ROLL":       "roll_number",
        }

        spans: Dict[str, List[Tuple[str, float]]] = {f: [] for f in field_map.values()}
        span_probs: Dict[str, List[float]] = {f: [] for f in field_map.values()}

        current_field = None
        for word, label, prob in zip(words, labels, probs):
            if label.startswith("B-"):
                entity = label[2:]
                field_key = field_map.get(entity)
                if field_key:
                    current_field = field_key
                    spans[field_key].append(word)
                    span_probs[field_key].append(prob)
            elif label.startswith("I-") and current_field:
                spans[current_field].append(word)
                span_probs[current_field].append(prob)
            else:
                current_field = None

        # Build result
        result = FieldExtractionResult(raw_text=raw_text)
        field_scores = {}

        for field_key in field_map.values():
            if spans[field_key]:
                text  = " ".join(spans[field_key])
                score = float(np.mean(span_probs[field_key]))

                # Post-process
                text = FieldExtractor._postprocess(field_key, text)

                setattr(result, field_key, text)
                field_scores[field_key] = round(score, 4)
            else:
                field_scores[field_key] = 0.0

        result.field_scores = field_scores
        return result

    @staticmethod
    def _postprocess(field_key: str, text: str) -> str:
        """Clean up extracted text per field type."""
        text = text.strip()
        if field_key == "student_name":
            # Title case, remove extra whitespace
            text = re.sub(r"\s+", " ", text).title()
        elif field_key == "institution":
            text = text.upper()
        elif field_key == "issue_date":
            # Normalise date separators
            text = re.sub(r"[/\-]", " ", text)
        elif field_key == "grade":
            text = text.upper().replace(" ", "")
        elif field_key == "roll_number":
            text = text.upper().replace(" ", "")
        return text

    # ------------------------------------------------------------------ #
    #  Save / load                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "FieldExtractor":
        return cls(model_path=path, **kwargs)


# ── Training pipeline ─────────────────────────────────────────────────────

class LayoutLMv3NERDataset(torch.utils.data.Dataset):
    """
    Dataset for fine-tuning LayoutLMv3 on certificate NER.

    Each sample is a JSON file with:
      {
        "image_path": "...",
        "words":  ["John", "Doe", ...],
        "boxes":  [[x1,y1,x2,y2], ...],
        "labels": ["B-NAME", "I-NAME", "O", ...]
      }
    """

    def __init__(
        self,
        annotation_dir: str | Path,
        processor: LayoutLMv3Processor,
        max_length: int = 512,
    ):
        self.processor  = processor
        self.max_length = max_length
        self.samples    = self._load(Path(annotation_dir))

    def _load(self, ann_dir: Path) -> List[dict]:
        import json
        samples = []
        for p in sorted(ann_dir.glob("*.json")):
            with open(p) as f:
                samples.append(json.load(f))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        pil = Image.open(s["image_path"]).convert("RGB")

        word_labels = [LABEL2ID.get(l, 0) for l in s["labels"]]

        encoding = self.processor(
            pil,
            s["words"],
            boxes=s["boxes"],
            word_labels=word_labels,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


def finetune_layoutlmv3(
    annotation_dir: str,
    output_dir: str,
    base_model: str = "microsoft/layoutlmv3-base",
    epochs: int = 15,
    batch_size: int = 2,
    lr: float = 2e-5,
):
    """Fine-tune LayoutLMv3 for certificate NER."""
    processor = LayoutLMv3Processor.from_pretrained(base_model, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_ds = LayoutLMv3NERDataset(f"{annotation_dir}/train", processor)
    val_ds   = LayoutLMv3NERDataset(f"{annotation_dir}/val",   processor)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Fine-tuned LayoutLMv3 saved to {output_dir}")
    return trainer


if __name__ == "__main__":
    print("[OK] layoutlm_extractor.py imports cleanly")
    print(f"     NER labels: {LABELS}")
    print(f"     NUM_LABELS: {NUM_LABELS}")
