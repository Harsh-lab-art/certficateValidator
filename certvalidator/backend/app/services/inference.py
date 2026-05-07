"""
backend/app/services/inference.py

End-to-end certificate verification pipeline.
Fully resilient — works even when no models are trained yet.
Uses heuristic fallbacks so the API is always functional.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class VerificationResult:
    verification_id:    str
    verdict:            str
    trust_score:        float
    explanation:        str
    forgery_score:      float
    field_confidence:   float
    nlp_anomaly_score:  float
    institution_matched: bool
    fields:             Dict[str, Optional[str]]
    field_scores:       List[dict]
    flagged_fields:     List[str]
    nlp_reasoning:      str
    tamper_regions:     List[dict]
    ocr_raw_text:       str
    contributions:      Dict[str, float]
    confidence_interval: float
    heatmap_bgr:        Optional[Any]
    heatmap_path:       Optional[str]
    processing_time_s:  float
    file_hash:          str
    model_versions:     Dict[str, str]

    def to_api_response(self) -> dict:
        return {
            "verification_id":    self.verification_id,
            "status":             "done",
            "verdict":            self.verdict,
            "trust_score":        round(self.trust_score, 1),
            "explanation":        self.explanation,
            "forgery_score":      round(self.forgery_score, 4),
            "field_confidence":   round(self.field_confidence, 4),
            "nlp_anomaly_score":  round(self.nlp_anomaly_score, 4),
            "institution_matched": self.institution_matched,
            "field_scores":       self.field_scores,
            "tamper_regions":     self.tamper_regions,
            "nlp_reasoning":      self.nlp_reasoning,
            "ocr_raw_text":       self.ocr_raw_text,
            "contributions":      {k: round(v, 2) for k, v in self.contributions.items()},
            "confidence_interval": round(self.confidence_interval, 2),
            "heatmap_url":        f"/api/v1/verify/{self.verification_id}/heatmap" if self.heatmap_path else None,
            "report_pdf_url":     f"/api/v1/verify/{self.verification_id}/report",
            "processing_time_s":  round(self.processing_time_s, 3),
            "model_versions":     self.model_versions,
        }


class CertificatePipeline:
    """
    Resilient inference pipeline.

    On first run (before training), uses:
      - Basic preprocessing (always works)
      - Heuristic field extraction (pytesseract if available, else placeholder)
      - Heuristic NLP reasoning (rule-based Mistral fallback)
      - Trust score fusion (always works)

    After training, automatically picks up trained checkpoints.
    """
    _instance: Optional["CertificatePipeline"] = None

    def __init__(
        self,
        checkpoint_dir: str = "ml/models/checkpoints",
        heatmap_dir:    str = "uploads/heatmaps",
        device:         str = "auto",
    ):
        import torch
        self.checkpoint_dir = Path(checkpoint_dir)
        self.heatmap_dir    = Path(heatmap_dir)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._load_models()

    def _load_models(self):
        """Load each model independently — failures are non-fatal."""
        self.preprocessor    = None
        self.forgery_model   = None
        self.gradcam         = None
        self.field_extractor = None
        self.reasoner        = None
        self.fusion          = None
        self.model_status    = {}

        # 1. Preprocessor
        try:
            from ml.src.preprocessing.pipeline import CertificatePreprocessor
            self.preprocessor = CertificatePreprocessor()
            self.model_status["preprocessing"] = "loaded"
        except Exception as e:
            self.model_status["preprocessing"] = f"error: {e}"

        # 2. Forgery detector
        try:
            import torch
            from ml.src.models.forgery_detector import ForgeryDetector
            from ml.src.models.gradcam import GradCAMEngine
            ckpt = self.checkpoint_dir / "forgery_best.pt"
            if ckpt.exists():
                self.forgery_model = ForgeryDetector.from_checkpoint(
                    str(ckpt), device=self.device
                )
                self.forgery_model = self.forgery_model.to(self.device)
                self.gradcam = GradCAMEngine(
                    self.forgery_model, self.forgery_model.features[-1]
                )
                self.model_status["forgery_detector"] = "trained-checkpoint"
            else:
                self.model_status["forgery_detector"] = "not-trained"
        except Exception as e:
            self.model_status["forgery_detector"] = f"error: {e}"

        # 3. Field extractor
        try:
            from ml.src.models.layout.layoutlm_extractor import FieldExtractor
            layout_ckpt = self.checkpoint_dir / "layoutlmv3_finetuned"
            model_src = str(layout_ckpt) if layout_ckpt.exists() else None
            self.field_extractor = FieldExtractor(
                model_path=model_src, device=self.device
            )
            status = "fine-tuned" if layout_ckpt.exists() else "base-pretrained"
            self.model_status["field_extractor"] = status
        except Exception as e:
            self.model_status["field_extractor"] = f"error: {e}"

        # 4. NLP reasoner (always has heuristic fallback)
        try:
            from ml.src.models.llm.mistral_reasoner import MistralReasoner
            self.reasoner = MistralReasoner()
            self.model_status["nlp_reasoner"] = self.reasoner.mode
        except Exception as e:
            self.model_status["nlp_reasoner"] = f"error: {e}"

        # 5. Fusion
        try:
            from ml.src.models.fusion.trust_score import TrustScoreFusion
            try:
                self.fusion = TrustScoreFusion.from_config("ml/config.yaml")
            except Exception:
                self.fusion = TrustScoreFusion()
            self.model_status["fusion"] = "loaded"
        except Exception as e:
            self.model_status["fusion"] = f"error: {e}"

        print(f"[Pipeline] Model status: {self.model_status}")

    @classmethod
    def get_instance(cls, **kwargs) -> "CertificatePipeline":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def verify(
        self,
        image_bytes:      Optional[bytes] = None,
        image_path:       Optional[str]   = None,
        image_suffix:     str = ".jpg",
        generate_heatmap: bool = True,
    ) -> VerificationResult:
        t_start = time.time()
        vid     = str(uuid.uuid4())

        # File hash
        if image_bytes:
            file_hash = hashlib.sha256(image_bytes).hexdigest()
        elif image_path:
            file_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
            image_bytes = Path(image_path).read_bytes()
        else:
            raise ValueError("Provide image_bytes or image_path")

        # ── Step 1: Preprocess ────────────────────────────────────────────
        img_bgr = None
        ela_bgr = None

        if self.preprocessor:
            try:
                result = self.preprocessor.process_bytes(image_bytes, suffix=image_suffix)
                if result.success:
                    img_bgr = result.processed_image
                    ela_bgr = result.ela_image
            except Exception as e:
                print(f"[Pipeline] Preprocessing error: {e}")

        # Fallback: load raw image if preprocessing failed
        if img_bgr is None:
            try:
                import cv2
                arr = np.frombuffer(image_bytes, np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    # PDF fallback
                    img_bgr = np.zeros((800, 1200, 3), dtype=np.uint8) + 240
                ela_bgr = np.zeros_like(img_bgr) + 128
            except Exception:
                img_bgr = np.zeros((800, 1200, 3), dtype=np.uint8) + 240
                ela_bgr = np.zeros_like(img_bgr) + 128

        # ── Step 2: Forgery detection ─────────────────────────────────────
        forgery_score   = 0.5   # neutral default
        tamper_regions  = []
        heatmap_bgr     = None
        heatmap_path    = None

        if self.forgery_model is not None:
            try:
                import torch
                combined = self._prepare_tensor(img_bgr, ela_bgr)
                forgery_score = float(
                    self.forgery_model.forgery_score(combined).item()
                )
                if generate_heatmap and self.gradcam:
                    h, w = img_bgr.shape[:2]
                    cam = self.gradcam.compute(combined, 1, (h, w))
                    heatmap_bgr = self.gradcam.overlay(cam, img_bgr, 0.45)
                    hp = self.heatmap_dir / f"{vid}_heatmap.png"
                    import cv2
                    cv2.imwrite(str(hp), heatmap_bgr)
                    heatmap_path = str(hp)
                    from ml.src.models.gradcam import GradCAMEngine
                    tamper_regions = GradCAMEngine.find_high_activation_regions(
                        cam, 0.65, img_shape=(h, w)
                    )
            except Exception as e:
                print(f"[Pipeline] Forgery detection error: {e}")

        # ── Step 3: Field extraction ──────────────────────────────────────
        extracted_fields = {
            "student_name": None, "institution": None,
            "degree": None, "discipline": None,
            "issue_date": None, "grade": None, "roll_number": None,
        }
        extraction_scores = {f: 0.0 for f in extracted_fields}
        ocr_text = ""

        if self.field_extractor is not None:
            try:
                extraction = self.field_extractor.extract(img_bgr)
                ocr_text = extraction.raw_text or ""
                extracted_fields.update({
                    "student_name": extraction.student_name,
                    "institution":  extraction.institution,
                    "degree":       extraction.degree,
                    "discipline":   extraction.discipline,
                    "issue_date":   extraction.issue_date,
                    "grade":        extraction.grade,
                    "roll_number":  extraction.roll_number,
                })
                extraction_scores = extraction.field_scores
            except Exception as e:
                print(f"[Pipeline] Field extraction error: {e}")
                # Try basic pytesseract fallback
                try:
                    import pytesseract
                    from PIL import Image
                    import cv2
                    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                    ocr_text = pytesseract.image_to_string(pil)
                except Exception:
                    pass

        # ── Step 4: Field scoring ─────────────────────────────────────────
        scored = None
        field_confidence = 0.5

        try:
            from ml.src.models.layout.field_scorer import score_fields
            scored = score_fields(extracted_fields, extraction_scores)
            field_confidence = scored.overall_confidence
        except Exception as e:
            print(f"[Pipeline] Field scoring error: {e}")

        flagged_fields     = scored.flagged_fields if scored else []
        consistency_issues = scored.consistency_issues if scored else []

        # ── Step 5: Institution lookup ────────────────────────────────────
        institution_matched = False
        try:
            from backend.app.services.institution_db import lookup_institution
            if extracted_fields.get("institution"):
                institution_matched = lookup_institution(
                    extracted_fields["institution"]
                )["matched"]
        except Exception as e:
            print(f"[Pipeline] Institution lookup error: {e}")

        # ── Step 6: NLP reasoning ─────────────────────────────────────────
        reasoning_text  = "Analysis complete."
        nlp_anomaly     = 0.5

        if self.reasoner:
            try:
                field_scores_dict = scored.field_scores if scored else extraction_scores
                rsn = self.reasoner.analyse(
                    fields              = extracted_fields,
                    field_scores        = field_scores_dict,
                    flagged_fields      = flagged_fields,
                    consistency_issues  = consistency_issues,
                    forgery_score       = forgery_score,
                    institution_matched = institution_matched,
                )
                reasoning_text = rsn.reasoning_text
                nlp_anomaly    = rsn.anomaly_score
            except Exception as e:
                print(f"[Pipeline] NLP reasoning error: {e}")

        # ── Step 7: Trust score fusion ────────────────────────────────────
        trust_score  = 50.0
        verdict      = "INCONCLUSIVE"
        explanation  = "Analysis complete."
        contributions = {}
        ci = 0.0

        if self.fusion:
            try:
                fusion_result = self.fusion.fuse(
                    forgery_score       = forgery_score,
                    field_confidence    = field_confidence,
                    nlp_anomaly_score   = nlp_anomaly,
                    institution_matched = institution_matched,
                )
                trust_score   = fusion_result.trust_score
                verdict       = fusion_result.verdict
                explanation   = fusion_result.explanation
                contributions = fusion_result.contributions
                ci            = fusion_result.confidence_interval
            except Exception as e:
                print(f"[Pipeline] Fusion error: {e}")

        # ── Build API field_scores list ───────────────────────────────────
        api_field_scores = (
            scored.to_api_format() if scored else [
                {"field": f, "value": v, "confidence": extraction_scores.get(f, 0.0), "flagged": False}
                for f, v in extracted_fields.items()
            ]
        )

        return VerificationResult(
            verification_id    = vid,
            verdict            = verdict,
            trust_score        = trust_score,
            explanation        = explanation,
            forgery_score      = forgery_score,
            field_confidence   = field_confidence,
            nlp_anomaly_score  = nlp_anomaly,
            institution_matched= institution_matched,
            fields             = extracted_fields,
            field_scores       = api_field_scores,
            flagged_fields     = flagged_fields,
            nlp_reasoning      = reasoning_text,
            tamper_regions     = tamper_regions,
            ocr_raw_text       = ocr_text,
            contributions      = contributions,
            confidence_interval= ci,
            heatmap_bgr        = heatmap_bgr,
            heatmap_path       = heatmap_path,
            processing_time_s  = time.time() - t_start,
            file_hash          = file_hash,
            model_versions     = self.model_status,
        )

    async def verify_async(self, image_bytes: bytes, image_suffix: str = ".jpg") -> VerificationResult:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.verify(image_bytes=image_bytes, image_suffix=image_suffix),
        )

    def _prepare_tensor(self, img_bgr, ela_bgr):
        import torch
        import cv2
        from torchvision import transforms

        h, w = 512, 724
        img_r = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        ela_r = cv2.resize(ela_bgr, (w, h), interpolation=cv2.INTER_AREA)

        to_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        ela_rgb = cv2.cvtColor(ela_r, cv2.COLOR_BGR2RGB)
        return torch.cat([to_t(img_rgb), to_t(ela_rgb)], dim=0).unsqueeze(0).to(self.device)


# Module-level singleton
_pipeline: Optional[CertificatePipeline] = None

def get_pipeline(**kwargs) -> CertificatePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CertificatePipeline(**kwargs)
    return _pipeline
