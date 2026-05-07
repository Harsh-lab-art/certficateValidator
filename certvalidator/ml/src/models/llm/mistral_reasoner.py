"""
ml/src/models/llm/mistral_reasoner.py

Mistral-7B GGUF reasoning engine for certificate anomaly analysis.

Why Mistral-7B on the RTX 3050 (4GB VRAM)?
  - Q4_K_M quantisation reduces model to ~4.1GB — fits exactly
  - 20 GPU layers offloaded via llama-cpp-python
  - Remaining layers run on CPU (i5 handles this fine)
  - Inference: ~8-12 seconds per certificate — acceptable for our use case

The LLM does NOT make the final verdict — that is the trust score fusion
layer's job. The LLM contributes:
  1. A natural-language reasoning paragraph (shown in the UI)
  2. An anomaly score (0.0–1.0) used by the fusion layer
  3. A structured list of specific inconsistencies it detected

Model download (run once):
    python -m ml.src.models.llm.mistral_reasoner download

Usage:
    reasoner = MistralReasoner()
    result   = reasoner.analyse(field_extraction_result, scored_fields)
    print(result.reasoning_text)
    print(result.anomaly_score)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

console = Console()


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass
class ReasoningResult:
    reasoning_text:  str              # Human-readable paragraph for UI
    anomaly_score:   float            # 0.0 = clean  1.0 = highly anomalous
    detected_issues: List[str]        # Structured list of anomalies found
    confidence:      float            # LLM's self-reported confidence
    processing_time_ms: float = 0.0
    model_used:      str = "mistral-7b-q4"

    def to_dict(self) -> dict:
        return {
            "reasoning_text":   self.reasoning_text,
            "anomaly_score":    round(self.anomaly_score, 4),
            "detected_issues":  self.detected_issues,
            "confidence":       round(self.confidence, 4),
            "processing_time_ms": round(self.processing_time_ms, 1),
            "model_used":       self.model_used,
        }


# ── Prompt builder ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert forensic document examiner specialising in \
academic certificate authentication. Your task is to analyse extracted certificate \
fields and detect inconsistencies, anomalies, or signs of tampering.

You must respond ONLY with a valid JSON object — no preamble, no markdown, no extra text.

JSON schema:
{
  "reasoning": "<2-4 sentence forensic analysis paragraph>",
  "anomaly_score": <float 0.0-1.0>,
  "issues": ["<issue 1>", "<issue 2>"],
  "confidence": <float 0.0-1.0>
}

anomaly_score guide:
  0.0-0.2  = Certificate appears genuine, no notable anomalies
  0.2-0.4  = Minor inconsistencies, likely genuine
  0.4-0.6  = Suspicious — notable anomalies detected
  0.6-0.8  = Likely tampered — significant inconsistencies
  0.8-1.0  = Almost certainly fake — multiple serious anomalies
"""


def build_analysis_prompt(
    fields: Dict[str, Any],
    field_scores: Dict[str, float],
    flagged_fields: List[str],
    consistency_issues: List[str],
    forgery_score: float,
    institution_matched: bool,
) -> str:
    """Build the user prompt for Mistral from certificate analysis data."""
    flagged_str = (
        ", ".join(flagged_fields) if flagged_fields else "none"
    )
    consistency_str = (
        "; ".join(consistency_issues) if consistency_issues else "none detected"
    )

    lines = [
        "Analyse this certificate for authenticity:\n",
        f"EXTRACTED FIELDS:",
        f"  Student name:  {fields.get('student_name', 'NOT FOUND')}",
        f"  Institution:   {fields.get('institution', 'NOT FOUND')}",
        f"  Degree:        {fields.get('degree', 'NOT FOUND')}",
        f"  Discipline:    {fields.get('discipline', 'NOT FOUND')}",
        f"  Issue date:    {fields.get('issue_date', 'NOT FOUND')}",
        f"  Grade / CGPA:  {fields.get('grade', 'NOT FOUND')}",
        f"  Roll number:   {fields.get('roll_number', 'NOT FOUND')}",
        f"\nFIELD CONFIDENCE SCORES (0-1, lower = more suspicious):",
    ]
    for fname, score in field_scores.items():
        flag = " ← LOW CONFIDENCE" if score < 0.6 else ""
        lines.append(f"  {fname:20s}: {score:.2f}{flag}")

    lines += [
        f"\nFLAGGED FIELDS: {flagged_str}",
        f"CROSS-FIELD CONSISTENCY ISSUES: {consistency_str}",
        f"FORGERY DETECTOR SCORE: {forgery_score:.3f} (0=genuine, 1=fake)",
        f"INSTITUTION DATABASE MATCH: {'YES' if institution_matched else 'NO — unknown institution'}",
        f"\nBased on all evidence above, provide your forensic analysis as JSON.",
    ]

    return "\n".join(lines)


# ── Mistral reasoner ───────────────────────────────────────────────────────

class MistralReasoner:
    """
    Mistral-7B GGUF reasoning engine.

    Tries to load llama-cpp-python first (GPU accelerated).
    Falls back to a rule-based heuristic engine if llama.cpp
    is not installed or the model file is missing — so the
    rest of the pipeline stays functional during development.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_gpu_layers: int = 20,
        n_threads: int = 8,
        context_length: int = 4096,
        max_new_tokens: int = 300,
        temperature: float = 0.1,
    ):
        self.model_path    = model_path or self._default_model_path()
        self.n_gpu_layers  = n_gpu_layers
        self.n_threads     = n_threads
        self.ctx_len       = context_length
        self.max_tokens    = max_new_tokens
        self.temperature   = temperature
        self._llm          = None
        self._mode         = "unloaded"

        self._load()

    def _default_model_path(self) -> str:
        candidates = [
            "ml/models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "ml/models/llm/mistral-7b-q4.gguf",
            Path.home() / ".cache" / "certvalidator" / "mistral-7b-q4.gguf",
        ]
        for p in candidates:
            if Path(p).exists():
                return str(p)
        return "ml/models/llm/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    def _load(self):
        if not Path(self.model_path).exists():
            console.print(
                f"[yellow]Mistral model not found at {self.model_path}.[/yellow]\n"
                f"[yellow]Run: python -m ml.src.models.llm.mistral_reasoner download[/yellow]\n"
                f"[yellow]Falling back to rule-based reasoner.[/yellow]"
            )
            self._mode = "heuristic"
            return

        try:
            from llama_cpp import Llama
            self._llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                n_ctx=self.ctx_len,
                verbose=False,
            )
            self._mode = "llm"
            console.print(f"[green]Mistral-7B loaded ({self.n_gpu_layers} GPU layers)[/green]")
        except ImportError:
            console.print(
                "[yellow]llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python[cublas] "
                "Falling back to rule-based reasoner.[/yellow]"
            )
            self._mode = "heuristic"
        except Exception as e:
            console.print(f"[yellow]Mistral load error: {e}. Using heuristic fallback.[/yellow]")
            self._mode = "heuristic"

    # ------------------------------------------------------------------ #
    #  Main analysis                                                       #
    # ------------------------------------------------------------------ #

    def analyse(
        self,
        fields: Dict[str, Any],
        field_scores: Dict[str, float],
        flagged_fields: List[str],
        consistency_issues: List[str],
        forgery_score: float = 0.0,
        institution_matched: bool = True,
    ) -> ReasoningResult:
        """
        Analyse a certificate and return forensic reasoning + anomaly score.
        Dispatches to LLM or heuristic fallback based on what's available.
        """
        t0 = time.time()

        if self._mode == "llm":
            result = self._llm_analyse(
                fields, field_scores, flagged_fields,
                consistency_issues, forgery_score, institution_matched
            )
        else:
            result = self._heuristic_analyse(
                fields, field_scores, flagged_fields,
                consistency_issues, forgery_score, institution_matched
            )

        result.processing_time_ms = (time.time() - t0) * 1000
        return result

    # ------------------------------------------------------------------ #
    #  LLM path                                                            #
    # ------------------------------------------------------------------ #

    def _llm_analyse(self, fields, field_scores, flagged, consistency_issues,
                     forgery_score, institution_matched) -> ReasoningResult:
        prompt = build_analysis_prompt(
            fields, field_scores, flagged,
            consistency_issues, forgery_score, institution_matched
        )

        full_prompt = (
            f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
            f"{prompt} [/INST]"
        )

        response = self._llm(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "[INST]"],
        )

        raw = response["choices"][0]["text"].strip()
        return self._parse_llm_output(raw)

    def _parse_llm_output(self, raw: str) -> ReasoningResult:
        """Parse JSON from LLM output, with robust fallback on parse errors."""
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        try:
            data = json.loads(raw)
            return ReasoningResult(
                reasoning_text  = str(data.get("reasoning", raw)),
                anomaly_score   = float(data.get("anomaly_score", 0.5)),
                detected_issues = list(data.get("issues", [])),
                confidence      = float(data.get("confidence", 0.7)),
                model_used      = "mistral-7b-q4",
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Extract what we can from raw text
            score_match = re.search(r'"anomaly_score"\s*:\s*([\d.]+)', raw)
            score = float(score_match.group(1)) if score_match else 0.5

            return ReasoningResult(
                reasoning_text  = raw[:500] if raw else "Analysis unavailable.",
                anomaly_score   = score,
                detected_issues = [],
                confidence      = 0.5,
                model_used      = "mistral-7b-q4-parse-fallback",
            )

    # ------------------------------------------------------------------ #
    #  Heuristic fallback (no LLM required)                               #
    # ------------------------------------------------------------------ #

    def _heuristic_analyse(
        self, fields, field_scores, flagged,
        consistency_issues, forgery_score, institution_matched
    ) -> ReasoningResult:
        """
        Rule-based reasoning that mimics what the LLM would say.
        Used during development when Mistral isn't downloaded yet.
        Produces the same interface as the LLM path.
        """
        issues     = list(flagged) + consistency_issues
        n_issues   = len(issues)

        # Compute anomaly score from available signals
        base_score = forgery_score * 0.4
        if not institution_matched:
            base_score += 0.15
        field_penalty = sum(
            max(0, 0.6 - s) * 0.5
            for s in field_scores.values()
        ) / max(len(field_scores), 1)
        base_score += field_penalty

        anomaly_score = min(1.0, base_score)

        # Build reasoning paragraph
        student  = fields.get("student_name") or "the student"
        inst     = fields.get("institution")  or "an unknown institution"
        degree   = fields.get("degree")       or "a degree"
        date     = fields.get("issue_date")   or "an unspecified date"

        if anomaly_score < 0.2:
            reasoning = (
                f"The certificate for {student} from {inst} presents no significant "
                f"anomalies. All extracted fields are internally consistent, the "
                f"institution matches the reference database, and the forgery detector "
                f"reports a low tamper probability ({forgery_score:.1%}). "
                f"The {degree} issued on {date} appears genuine."
            )
        elif anomaly_score < 0.45:
            reasoning = (
                f"Minor inconsistencies were detected in the certificate for {student}. "
                + (f"The following fields show reduced confidence: {', '.join(flagged)}. " if flagged else "")
                + (f"Cross-field issues: {'; '.join(consistency_issues)}. " if consistency_issues else "")
                + f"The forgery detector score is {forgery_score:.1%}. "
                f"Overall the certificate is likely genuine but warrants closer inspection."
            )
        elif anomaly_score < 0.7:
            reasoning = (
                f"Significant anomalies detected in the certificate for {student}. "
                + (f"Flagged fields: {', '.join(flagged)}. " if flagged else "")
                + (f"Consistency problems: {'; '.join(consistency_issues)}. " if consistency_issues else "")
                + (f"Institution '{inst}' not found in the reference database. " if not institution_matched else "")
                + f"Forgery detector confidence: {forgery_score:.1%}. "
                f"This certificate is suspicious and likely tampered."
            )
        else:
            reasoning = (
                f"Multiple serious anomalies indicate this certificate is almost certainly "
                f"forged or tampered. The forgery detector reports {forgery_score:.1%} "
                f"tamper probability. "
                + (f"All following fields were flagged: {', '.join(flagged)}. " if flagged else "")
                + (f"Critical consistency failures: {'; '.join(consistency_issues)}. " if consistency_issues else "")
                + (f"The institution '{inst}' is not registered in our database. " if not institution_matched else "")
                + "This certificate should be rejected."
            )

        return ReasoningResult(
            reasoning_text  = reasoning,
            anomaly_score   = round(anomaly_score, 4),
            detected_issues = issues[:10],
            confidence      = 0.75,
            model_used      = "heuristic-fallback",
        )

    @property
    def mode(self) -> str:
        return self._mode


# ── Model downloader ───────────────────────────────────────────────────────

def download_mistral(output_dir: str = "ml/models/llm"):
    """
    Download Mistral-7B-Instruct Q4_K_M GGUF from HuggingFace.
    File size ~4.1GB — requires internet connection on first run.
    """
    import urllib.request

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    url  = (
        "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        "/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    )
    dest = out / "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

    if dest.exists():
        console.print(f"[green]Model already exists at {dest}[/green]")
        return str(dest)

    console.print(f"[cyan]Downloading Mistral-7B Q4_K_M (~4.1GB)...[/cyan]")
    console.print(f"[cyan]Destination: {dest}[/cyan]")

    def progress(block, block_size, total):
        done = block * block_size
        pct  = min(100, int(done / total * 100)) if total > 0 else 0
        bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}% ({done/1e9:.2f}/{total/1e9:.2f}GB)", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=progress)
    print()
    console.print(f"[green]Downloaded to {dest}[/green]")
    return str(dest)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download_mistral()
    else:
        # Quick demo
        reasoner = MistralReasoner()
        print(f"Mode: {reasoner.mode}")

        result = reasoner.analyse(
            fields={"student_name":"Rahul Sharma","institution":"DTU","degree":"B.Tech","issue_date":"15 May 2023","grade":"A+ / 9.6/10","roll_number":"19DTU1234"},
            field_scores={"student_name":0.95,"institution":0.92,"degree":0.90,"issue_date":0.88,"grade":0.85,"roll_number":0.91},
            flagged_fields=[],
            consistency_issues=[],
            forgery_score=0.07,
            institution_matched=True,
        )
        print(f"\nAnomaly score : {result.anomaly_score}")
        print(f"Reasoning     : {result.reasoning_text}")
        print(f"Issues        : {result.detected_issues}")
        print(f"Model used    : {result.model_used}")
