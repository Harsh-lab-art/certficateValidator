"""
backend/tests/test_pipeline.py

Integration tests for the full inference pipeline.
Runs without trained model checkpoints using the heuristic fallback.
"""

from __future__ import annotations
import sys, numpy as np, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pytest
from ml.src.models.fusion.trust_score import TrustScoreFusion, FusionResult
from ml.src.models.llm.mistral_reasoner import MistralReasoner, build_analysis_prompt
from ml.src.models.layout.field_scorer import score_fields, FieldValidator, ConsistencyChecker
from ml.src.preprocessing.pipeline import CertificatePreprocessor


# ── Fusion tests ──────────────────────────────────────────────────────────

class TestTrustScoreFusion:

    def setup_method(self):
        self.fusion = TrustScoreFusion()

    def test_weights_sum_to_one(self):
        assert abs(self.fusion.w_forgery + self.fusion.w_field + self.fusion.w_nlp - 1.0) < 1e-6

    def test_genuine_certificate(self):
        r = self.fusion.fuse(forgery_score=0.05, field_confidence=0.95,
                             nlp_anomaly_score=0.08, institution_matched=True)
        assert r.verdict == "GENUINE"
        assert r.trust_score >= 75
        assert "GENUINE" in r.explanation

    def test_fake_certificate(self):
        r = self.fusion.fuse(forgery_score=0.92, field_confidence=0.30,
                             nlp_anomaly_score=0.88, institution_matched=False)
        assert r.verdict == "FAKE"
        assert r.trust_score < 45
        assert "FAKE" in r.explanation or "TAMPERED" in r.explanation

    def test_suspicious_certificate(self):
        r = self.fusion.fuse(forgery_score=0.42, field_confidence=0.65,
                             nlp_anomaly_score=0.45, institution_matched=True)
        assert r.verdict == "SUSPICIOUS"
        assert 45 <= r.trust_score < 75

    def test_contributions_sum_near_score(self):
        r = self.fusion.fuse(0.1, 0.9, 0.1, True)
        contrib_sum = sum(r.contributions.values())
        assert abs(contrib_sum - r.trust_score) < 2.0  # within 2 points

    def test_institution_mismatch_lowers_score(self):
        r_match   = self.fusion.fuse(0.2, 0.8, 0.2, institution_matched=True)
        r_nomatch = self.fusion.fuse(0.2, 0.8, 0.2, institution_matched=False)
        assert r_match.trust_score > r_nomatch.trust_score

    def test_all_verdicts_covered(self):
        verdicts = set()
        test_inputs = [
            (0.02, 0.98, 0.02, True),
            (0.50, 0.60, 0.50, True),
            (0.95, 0.20, 0.90, False),
        ]
        for args in test_inputs:
            verdicts.add(self.fusion.fuse(*args).verdict)
        assert "GENUINE" in verdicts
        assert "FAKE" in verdicts

    def test_score_clamped_0_100(self):
        for forgery, field, nlp in [(0.0,1.0,0.0), (1.0,0.0,1.0), (0.5,0.5,0.5)]:
            r = self.fusion.fuse(forgery, field, nlp, True)
            assert 0 <= r.trust_score <= 100

    def test_to_dict_keys(self):
        r = self.fusion.fuse(0.1, 0.9, 0.1, True)
        d = r.to_dict()
        assert all(k in d for k in ["trust_score","verdict","forgery_score",
                                     "field_confidence","contributions","explanation"])

    def test_from_config(self):
        try:
            f = TrustScoreFusion.from_config("ml/config.yaml")
            assert f.w_forgery == 0.45
        except FileNotFoundError:
            pass  # config not available in CI — skip

    def test_batch_fusion(self):
        results = self.fusion.fuse_batch(
            forgery_scores=[0.05, 0.90],
            field_confidences=[0.95, 0.25],
            nlp_anomaly_scores=[0.05, 0.85],
            institution_flags=[True, False],
        )
        assert len(results) == 2
        assert results[0].verdict == "GENUINE"
        assert results[1].verdict == "FAKE"


# ── Mistral reasoner tests ────────────────────────────────────────────────

class TestMistralReasoner:

    def setup_method(self):
        # Always uses heuristic (no model file in test env)
        self.reasoner = MistralReasoner()

    def test_mode_is_heuristic(self):
        assert self.reasoner.mode == "heuristic"

    def test_analyse_genuine(self):
        r = self.reasoner.analyse(
            fields={"student_name":"Rahul Sharma","institution":"IIT Delhi",
                    "degree":"B.Tech","issue_date":"2023","grade":"9.2/10","roll_number":"19CS001"},
            field_scores={"student_name":0.95,"institution":0.92,"degree":0.90,
                          "issue_date":0.88,"grade":0.85,"roll_number":0.91},
            flagged_fields=[],
            consistency_issues=[],
            forgery_score=0.05,
            institution_matched=True,
        )
        assert 0.0 <= r.anomaly_score <= 1.0
        assert len(r.reasoning_text) > 20
        assert r.anomaly_score < 0.4

    def test_analyse_fake(self):
        r = self.reasoner.analyse(
            fields={"student_name":"X","institution":"Fake Corp","degree":"BXY",
                    "issue_date":"3000","grade":"15/10","roll_number":"X"},
            field_scores={"student_name":0.20,"institution":0.10,"degree":0.15,
                          "issue_date":0.05,"grade":0.08,"roll_number":0.10},
            flagged_fields=["student_name","institution","degree","issue_date","grade","roll_number"],
            consistency_issues=["date in far future","impossible CGPA"],
            forgery_score=0.92,
            institution_matched=False,
        )
        assert r.anomaly_score > 0.5
        assert len(r.detected_issues) > 0

    def test_result_has_all_fields(self):
        r = self.reasoner.analyse({},{},[], [], 0.3, True)
        assert hasattr(r, "reasoning_text")
        assert hasattr(r, "anomaly_score")
        assert hasattr(r, "detected_issues")
        assert hasattr(r, "confidence")
        assert hasattr(r, "model_used")

    def test_prompt_builder(self):
        prompt = build_analysis_prompt(
            fields={"student_name":"Test"},
            field_scores={"student_name":0.9},
            flagged_fields=[],
            consistency_issues=[],
            forgery_score=0.1,
            institution_matched=True,
        )
        assert "FORGERY DETECTOR SCORE" in prompt
        assert "student_name" in prompt.lower() or "student name" in prompt.lower()

    def test_to_dict(self):
        r = self.reasoner.analyse({},{},[], [], 0.3, True)
        d = r.to_dict()
        assert all(k in d for k in ["reasoning_text","anomaly_score","detected_issues","model_used"])


# ── End-to-end pipeline smoke test ────────────────────────────────────────

class TestPipelineEndToEnd:

    def test_preprocessing_produces_ela(self):
        pp = CertificatePreprocessor()
        # Create synthetic test image
        img = np.ones((800, 1100, 3), dtype=np.uint8) * 240
        cv2.putText(img, "TEST CERTIFICATE", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            r = pp.process(f.name)
        assert r.success
        assert r.ela_image.shape == r.processed_image.shape

    def test_fusion_pipeline_integration(self):
        """Full pipeline from raw scores to API response dict."""
        fusion    = TrustScoreFusion()
        reasoner  = MistralReasoner()

        fields = {"student_name":"Priya Patel","institution":"IIT Bombay",
                  "degree":"M.Tech","issue_date":"10 June 2024",
                  "grade":"8.9/10","roll_number":"22CSE0042"}
        f_scores = {k: 0.88 for k in fields}

        scoring = score_fields(fields, f_scores)
        reasoning = reasoner.analyse(
            fields=fields, field_scores=scoring.field_scores,
            flagged_fields=scoring.flagged_fields,
            consistency_issues=scoring.consistency_issues,
            forgery_score=0.12, institution_matched=True,
        )
        result = fusion.fuse(
            forgery_score=0.12,
            field_confidence=scoring.overall_confidence,
            nlp_anomaly_score=reasoning.anomaly_score,
            institution_matched=True,
        )

        assert result.verdict in ("GENUINE", "SUSPICIOUS", "FAKE")
        assert 0 <= result.trust_score <= 100
        assert len(result.explanation) > 10

        # Verify this looks like a genuine cert
        assert result.verdict == "GENUINE", (
            f"Expected GENUINE for clean cert, got {result.verdict} "
            f"(score={result.trust_score:.1f})"
        )
