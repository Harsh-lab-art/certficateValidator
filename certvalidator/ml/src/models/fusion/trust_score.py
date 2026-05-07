"""
ml/src/models/fusion/trust_score.py

Trust score fusion layer.

Combines outputs from all three models into a single 0-100 trust score
and a categorical verdict.

Fusion formula:
  raw_score = (1 - forgery_score)    × W_FORGERY   (0=fake, 1=genuine)
            + field_confidence        × W_FIELD
            + (1 - nlp_anomaly_score) × W_NLP
            + institution_bonus       × W_INST

  trust_score = raw_score × 100   (clamped to [0, 100])

Weights (from ml/config.yaml):
  forgery : 0.45  ← most reliable, trained on forensic signal
  field   : 0.35  ← field extraction quality
  nlp     : 0.20  ← LLM reasoning

Verdict thresholds:
  score >= 75  → GENUINE
  45 <= score < 75 → SUSPICIOUS
  score < 45   → FAKE / TAMPERED

The fusion layer also produces:
  - Per-component contribution breakdown (shown in UI)
  - Confidence interval (uncertainty estimate)
  - Explanation string for each verdict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ── Weights ────────────────────────────────────────────────────────────────

W_FORGERY = 0.45
W_FIELD   = 0.35
W_NLP     = 0.20

# Small bonus for institution database match
INST_MATCH_BONUS = 0.05

# Verdict thresholds
THRESH_GENUINE    = 75.0
THRESH_SUSPICIOUS = 45.0


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    # Primary outputs
    trust_score: float          # 0.0 – 100.0
    verdict: str                # "GENUINE" | "SUSPICIOUS" | "FAKE"

    # Raw sub-scores (each 0–1 where 1 = more genuine)
    forgery_genuine_score: float    # 1 - forgery_detector_output
    field_confidence:      float    # from LayoutLMv3 field scorer
    nlp_genuine_score:     float    # 1 - mistral_anomaly_score
    institution_matched:   bool

    # Component contributions to final score (for UI breakdown)
    contributions: Dict[str, float] = field(default_factory=dict)

    # Confidence interval (±)
    confidence_interval: float = 0.0

    # Human-readable explanation
    explanation: str = ""

    # Raw inputs (for logging)
    raw_forgery_score:  float = 0.0
    raw_field_score:    float = 0.0
    raw_nlp_score:      float = 0.0

    def to_dict(self) -> dict:
        return {
            "trust_score":          round(self.trust_score, 2),
            "verdict":              self.verdict,
            "forgery_score":        round(self.raw_forgery_score, 4),
            "field_confidence":     round(self.raw_field_score, 4),
            "nlp_anomaly_score":    round(self.raw_nlp_score, 4),
            "institution_matched":  self.institution_matched,
            "contributions":        {k: round(v, 4) for k, v in self.contributions.items()},
            "confidence_interval":  round(self.confidence_interval, 2),
            "explanation":          self.explanation,
        }

    def to_api_verdict_summary(self) -> dict:
        """Compact format for API response top-level."""
        return {
            "verdict":     self.verdict,
            "trust_score": round(self.trust_score, 1),
            "explanation": self.explanation,
        }


# ── Fusion engine ──────────────────────────────────────────────────────────

class TrustScoreFusion:
    """
    Weighted ensemble fusion of forgery detector, field scorer, and NLP reasoner.

    Parameters
    ----------
    w_forgery : weight for EfficientNet+ELA forgery score
    w_field   : weight for LayoutLMv3 field confidence
    w_nlp     : weight for Mistral NLP anomaly score
    """

    def __init__(
        self,
        w_forgery: float = W_FORGERY,
        w_field:   float = W_FIELD,
        w_nlp:     float = W_NLP,
        thresh_genuine:    float = THRESH_GENUINE,
        thresh_suspicious: float = THRESH_SUSPICIOUS,
    ):
        total = w_forgery + w_field + w_nlp
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total}"
        self.w_forgery = w_forgery
        self.w_field   = w_field
        self.w_nlp     = w_nlp
        self.thresh_genuine    = thresh_genuine
        self.thresh_suspicious = thresh_suspicious

    def fuse(
        self,
        forgery_score:       float,   # 0=genuine 1=fake (raw from EfficientNet)
        field_confidence:    float,   # 0=bad 1=good  (from LayoutLMv3)
        nlp_anomaly_score:   float,   # 0=clean 1=anomalous (from Mistral)
        institution_matched: bool = True,
        uncertainty_estimates: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        """
        Compute the final trust score.

        All inputs in [0, 1]. Higher forgery_score and nlp_anomaly_score
        mean MORE suspicious, so we invert them before weighting.
        """
        # Convert to "genuine likelihood" for each component
        g_forgery = 1.0 - float(np.clip(forgery_score,     0, 1))
        g_field   =       float(np.clip(field_confidence,   0, 1))
        g_nlp     = 1.0 - float(np.clip(nlp_anomaly_score, 0, 1))

        # Institution bonus (small, not a deciding factor on its own)
        inst_bonus = INST_MATCH_BONUS if institution_matched else 0.0

        # Weighted sum
        raw = (
            g_forgery * self.w_forgery +
            g_field   * self.w_field   +
            g_nlp     * self.w_nlp     +
            inst_bonus
        )
        trust_score = float(np.clip(raw * 100, 0, 100))

        # Verdict
        verdict = self._verdict(trust_score)

        # Per-component contributions (what each model added to the score)
        contributions = {
            "forgery_detector": round(g_forgery * self.w_forgery * 100, 2),
            "field_extractor":  round(g_field   * self.w_field   * 100, 2),
            "nlp_reasoning":    round(g_nlp     * self.w_nlp     * 100, 2),
            "institution_match": round(inst_bonus * 100, 2),
        }

        # Uncertainty: std of the three genuine-likelihood scores
        scores   = [g_forgery, g_field, g_nlp]
        ci       = float(np.std(scores) * 100 * 1.96)   # 95% CI

        explanation = self._build_explanation(
            verdict, trust_score, g_forgery, g_field, g_nlp,
            institution_matched, ci
        )

        return FusionResult(
            trust_score           = round(trust_score, 2),
            verdict               = verdict,
            forgery_genuine_score = g_forgery,
            field_confidence      = g_field,
            nlp_genuine_score     = g_nlp,
            institution_matched   = institution_matched,
            contributions         = contributions,
            confidence_interval   = round(ci, 2),
            explanation           = explanation,
            raw_forgery_score     = forgery_score,
            raw_field_score       = field_confidence,
            raw_nlp_score         = nlp_anomaly_score,
        )

    def _verdict(self, score: float) -> str:
        if score >= self.thresh_genuine:
            return "GENUINE"
        elif score >= self.thresh_suspicious:
            return "SUSPICIOUS"
        else:
            return "FAKE"

    @staticmethod
    def _build_explanation(
        verdict: str,
        score: float,
        g_forgery: float,
        g_field: float,
        g_nlp: float,
        inst_matched: bool,
        ci: float,
    ) -> str:
        weakest = min(
            [("forgery detector", g_forgery),
             ("field extraction", g_field),
             ("NLP reasoning",    g_nlp)],
            key=lambda x: x[1]
        )
        strongest = max(
            [("forgery detector", g_forgery),
             ("field extraction", g_field),
             ("NLP reasoning",    g_nlp)],
            key=lambda x: x[1]
        )

        if verdict == "GENUINE":
            return (
                f"Certificate verified as GENUINE with {score:.1f}/100 trust score "
                f"(±{ci:.1f}). "
                f"Strongest signal: {strongest[0]} ({strongest[1]:.0%} genuine). "
                + (f"Institution matched in reference database. " if inst_matched else "")
                + f"No significant tampering detected."
            )
        elif verdict == "SUSPICIOUS":
            return (
                f"Certificate flagged as SUSPICIOUS with {score:.1f}/100 trust score "
                f"(±{ci:.1f}). "
                f"Primary concern: {weakest[0]} score is low ({weakest[1]:.0%}). "
                + ("" if inst_matched else "Institution not found in reference database. ")
                + f"Manual review recommended."
            )
        else:
            return (
                f"Certificate marked as FAKE / TAMPERED with {score:.1f}/100 trust score "
                f"(±{ci:.1f}). "
                f"Critical failure in {weakest[0]} ({weakest[1]:.0%} genuine). "
                + ("" if inst_matched else "Institution unknown — not in reference database. ")
                + f"Certificate should be rejected."
            )

    # ── Batch fusion ─────────────────────────────────────────────────────

    def fuse_batch(
        self,
        forgery_scores:     list,
        field_confidences:  list,
        nlp_anomaly_scores: list,
        institution_flags:  list,
    ) -> List[FusionResult]:
        return [
            self.fuse(fs, fc, ns, im)
            for fs, fc, ns, im in zip(
                forgery_scores, field_confidences,
                nlp_anomaly_scores, institution_flags
            )
        ]

    # ── Calibration ──────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "ml/config.yaml") -> "TrustScoreFusion":
        """Load weights from ml/config.yaml."""
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        w = cfg["fusion"]["weights"]
        t = cfg["fusion"]["thresholds"]
        return cls(
            w_forgery=w["forgery"],
            w_field=w["field"],
            w_nlp=w["nlp"],
            thresh_genuine=t["genuine"],
            thresh_suspicious=t["suspicious"],
        )


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fusion = TrustScoreFusion()

    test_cases = [
        # (forgery, field_conf, nlp_anomaly, inst_match, expected_verdict)
        (0.05, 0.95, 0.08, True,  "GENUINE"),
        (0.65, 0.70, 0.55, False, "FAKE"),
        (0.35, 0.72, 0.40, True,  "SUSPICIOUS"),
        (0.92, 0.30, 0.88, False, "FAKE"),
        (0.02, 0.98, 0.03, True,  "GENUINE"),
    ]

    print(f"{'Forgery':>10} {'Field':>8} {'NLP':>8} {'Inst':>6} {'Score':>8} {'Verdict':>12} {'Expected':>12}")
    print("─" * 70)
    all_pass = True
    for forgery, field, nlp, inst, expected in test_cases:
        r = fusion.fuse(forgery, field, nlp, inst)
        ok = "✓" if r.verdict == expected else "✗"
        if r.verdict != expected:
            all_pass = False
        print(f"{forgery:10.2f} {field:8.2f} {nlp:8.2f} {str(inst):>6} "
              f"{r.trust_score:8.1f} {r.verdict:>12} {expected:>12} {ok}")

    print()
    if all_pass:
        print("[OK] All fusion test cases passed")
    else:
        print("[WARN] Some test cases failed — check threshold tuning")
