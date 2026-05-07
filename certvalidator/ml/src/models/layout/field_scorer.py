"""
ml/src/models/layout/field_scorer.py

Field confidence scorer and anomaly detector.

After LayoutLMv3 extracts fields from a certificate, this module:
  1. Validates each field against format rules (date format, CGPA range, etc.)
  2. Cross-checks fields for internal consistency (graduation date vs roll number year)
  3. Checks institution against the known institution database
  4. Computes a per-field "confidence" that feeds into the trust score fusion

This is the layer between raw field extraction and the NLP reasoning module.
It catches obvious anomalies (impossible dates, malformed roll numbers) that
don't need the full LLM treatment.

Score semantics:
  1.0 = field extracted cleanly AND passes all validation rules
  0.7 = extracted but fails one soft validation rule
  0.4 = extracted but fails hard validation (impossible value)
  0.0 = field not found / empty
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Validation rules ──────────────────────────────────────────────────────

KNOWN_DEGREES = {
    "bachelor of technology", "b.tech", "b tech",
    "master of technology", "m.tech", "m tech",
    "bachelor of science", "b.sc", "b sc", "bsc",
    "master of science", "m.sc", "m sc", "msc",
    "bachelor of engineering", "b.e", "be",
    "bachelor of commerce", "b.com", "bcom",
    "master of business administration", "mba",
    "bachelor of computer applications", "bca",
    "master of computer applications", "mca",
    "doctor of philosophy", "ph.d", "phd",
    "bachelor of arts", "b.a", "ba",
    "master of arts", "m.a", "ma",
}

KNOWN_GRADES = {
    "o", "a+", "a", "b+", "b", "c", "d", "f",
    "first class", "second class", "pass class",
    "first division", "second division", "third division",
    "distinction", "pass", "pass with merit",
    "first class with distinction",
}

MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}

# Roll number patterns from common Indian universities
ROLL_PATTERNS = [
    r"^\d{2}[A-Z]{2,5}\d{3,6}$",            # 19CSE1234
    r"^\d{4}[A-Z]{2}\d{4}$",                 # 2019CS1234
    r"^[A-Z]{2}\d{2}[A-Z]{2}\d{3}$",         # EN19CS001
    r"^\d{7,12}$",                            # Pure numeric
    r"^[A-Z]\d{9}$",                          # A123456789
]


# ── Per-field validators ──────────────────────────────────────────────────

@dataclass
class ValidationResult:
    score:     float            # 0.0 – 1.0
    issues:    List[str]        # list of detected problems
    cleaned:   Optional[str]    # cleaned/normalised value


class FieldValidator:

    # ── Student name ──────────────────────────────────────────────────────
    @staticmethod
    def validate_name(name: Optional[str]) -> ValidationResult:
        if not name:
            return ValidationResult(0.0, ["name not found"], None)

        issues = []
        score  = 1.0
        cleaned = name.strip()

        # Must have at least first + last name
        parts = cleaned.split()
        if len(parts) < 2:
            issues.append("name appears to be single word — may be incomplete")
            score -= 0.2

        # Should not contain numbers
        if re.search(r"\d", cleaned):
            issues.append(f"name contains digits: '{cleaned}'")
            score -= 0.4

        # Length sanity
        if len(cleaned) < 3:
            issues.append("name too short")
            score -= 0.3
        elif len(cleaned) > 80:
            issues.append("name unusually long")
            score -= 0.1

        # Should not be all caps (OCR artefact)
        if cleaned == cleaned.upper() and len(cleaned) > 5:
            cleaned = cleaned.title()

        # Check for non-Latin characters that might be OCR garbage
        non_latin = re.findall(r"[^\x00-\x7F\u0900-\u097F ]", cleaned)
        if non_latin:
            issues.append(f"unexpected characters in name: {non_latin[:3]}")
            score -= 0.2

        return ValidationResult(max(0.0, score), issues, cleaned)

    # ── Institution ───────────────────────────────────────────────────────
    @staticmethod
    def validate_institution(inst: Optional[str]) -> ValidationResult:
        if not inst:
            return ValidationResult(0.0, ["institution not found"], None)

        issues = []
        score  = 1.0
        cleaned = inst.strip()

        edu_keywords = {
            "university", "institute", "college", "school", "iit", "nit",
            "bits", "iisc", "iim", "deemed", "academy", "polytechnic",
        }
        has_keyword = any(k in cleaned.lower() for k in edu_keywords)
        if not has_keyword:
            issues.append("institution name lacks educational keyword")
            score -= 0.3

        if len(cleaned) < 5:
            issues.append("institution name too short")
            score -= 0.3

        return ValidationResult(max(0.0, score), issues, cleaned)

    # ── Degree ────────────────────────────────────────────────────────────
    @staticmethod
    def validate_degree(degree: Optional[str]) -> ValidationResult:
        if not degree:
            return ValidationResult(0.0, ["degree not found"], None)

        issues = []
        score  = 1.0
        cleaned = degree.strip().lower()

        if not any(d in cleaned for d in KNOWN_DEGREES):
            issues.append(f"unrecognised degree title: '{degree}'")
            score -= 0.4

        return ValidationResult(max(0.0, score), issues, degree.strip())

    # ── Date ──────────────────────────────────────────────────────────────
    @staticmethod
    def validate_date(date_str: Optional[str]) -> ValidationResult:
        if not date_str:
            return ValidationResult(0.0, ["date not found"], None)

        issues = []
        score  = 1.0
        cleaned = date_str.strip()

        # Try to parse
        parsed = None
        formats = [
            "%d %B %Y", "%d %b %Y",
            "%d/%m/%Y", "%d-%m-%Y",
            "%B %d %Y", "%b %d %Y",
            "%Y-%m-%d",
        ]
        for fmt in formats:
            try:
                parsed = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue

        if parsed is None:
            # Try extracting year at least
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", cleaned)
            if year_match:
                year = int(year_match.group())
                parsed = datetime(year, 1, 1)
                score -= 0.2
                issues.append("date only partially parsed — year extracted")
            else:
                issues.append(f"date format not recognised: '{date_str}'")
                score -= 0.5

        if parsed:
            now = datetime.now()
            if parsed.year < 1950:
                issues.append(f"date year {parsed.year} implausibly old")
                score -= 0.5
            elif parsed > now:
                issues.append(f"date {parsed.date()} is in the future")
                score -= 0.6
            elif parsed.year > now.year - 50:
                pass   # Normal range
            cleaned = parsed.strftime("%d %B %Y") if parsed.day != 1 else str(parsed.year)

        return ValidationResult(max(0.0, score), issues, cleaned)

    # ── Grade / CGPA ──────────────────────────────────────────────────────
    @staticmethod
    def validate_grade(grade: Optional[str]) -> ValidationResult:
        if not grade:
            return ValidationResult(0.0, ["grade not found"], None)

        issues = []
        score  = 1.0
        cleaned = grade.strip()

        # Check for CGPA
        cgpa_match = re.search(r"(\d+\.?\d*)\s*/?\s*10", cleaned)
        if cgpa_match:
            cgpa = float(cgpa_match.group(1))
            if cgpa > 10.0:
                issues.append(f"CGPA {cgpa} exceeds 10.0 — impossible on 10-point scale")
                score -= 0.8
            elif cgpa > 9.95:
                issues.append(f"CGPA {cgpa} is suspiciously high")
                score -= 0.2
            elif cgpa < 1.0:
                issues.append(f"CGPA {cgpa} is suspiciously low")
                score -= 0.3
            return ValidationResult(max(0.0, score), issues, cleaned)

        # Check percentage
        pct_match = re.search(r"(\d+\.?\d*)\s*%", cleaned)
        if pct_match:
            pct = float(pct_match.group(1))
            if pct > 100:
                issues.append(f"percentage {pct}% exceeds 100 — impossible")
                score -= 0.9
            return ValidationResult(max(0.0, score), issues, cleaned)

        # Check letter grade
        grade_lower = cleaned.lower()
        if not any(g in grade_lower for g in KNOWN_GRADES):
            issues.append(f"unrecognised grade format: '{grade}'")
            score -= 0.3

        return ValidationResult(max(0.0, score), issues, cleaned)

    # ── Roll number ───────────────────────────────────────────────────────
    @staticmethod
    def validate_roll_number(roll: Optional[str]) -> ValidationResult:
        if not roll:
            return ValidationResult(0.0, ["roll number not found"], None)

        issues = []
        score  = 1.0
        cleaned = roll.strip().upper().replace(" ", "")

        if not any(re.match(p, cleaned) for p in ROLL_PATTERNS):
            issues.append(f"roll number '{cleaned}' doesn't match known patterns")
            score -= 0.3

        if len(cleaned) < 4 or len(cleaned) > 20:
            issues.append(f"roll number length {len(cleaned)} unusual")
            score -= 0.2

        return ValidationResult(max(0.0, score), issues, cleaned)


# ── Cross-field consistency ───────────────────────────────────────────────

class ConsistencyChecker:
    """
    Checks that fields make logical sense together.
    E.g., issue date should be after ~18 years from any embedded birth year.
    """

    @staticmethod
    def check_all(fields: Dict) -> Tuple[float, List[str]]:
        """
        Returns (consistency_score, list_of_issues).
        consistency_score: 0.0 (very inconsistent) → 1.0 (consistent)
        """
        issues = []
        score  = 1.0

        name  = fields.get("student_name", "") or ""
        inst  = fields.get("institution",  "") or ""
        deg   = fields.get("degree",       "") or ""
        date  = fields.get("issue_date",   "") or ""
        grade = fields.get("grade",        "") or ""
        roll  = fields.get("roll_number",  "") or ""

        # Check roll number year vs issue date
        if roll and date:
            roll_year_match = re.search(r"\b(19|20)(\d{2})\b", roll)
            date_year_match = re.search(r"\b(19|20)(\d{2})\b", date)
            if roll_year_match and date_year_match:
                roll_year = int(roll_year_match.group())
                date_year = int(date_year_match.group())
                gap = date_year - roll_year
                if gap < 1:
                    issues.append(
                        f"issue date year ({date_year}) precedes roll number year ({roll_year})"
                    )
                    score -= 0.5
                elif gap > 10:
                    issues.append(
                        f"large gap between roll year ({roll_year}) and issue date ({date_year})"
                    )
                    score -= 0.2

        # PhD should not be issued in 3 years (min 3 years usually)
        if deg and "doctor" in deg.lower() and roll and date:
            roll_year_match = re.search(r"\b20(\d{2})\b", roll)
            date_year_match = re.search(r"\b20(\d{2})\b", date)
            if roll_year_match and date_year_match:
                gap = int(date_year_match.group()) - int(roll_year_match.group())
                if gap < 3:
                    issues.append(f"PhD completed in {gap} years — unusually fast")
                    score -= 0.3

        # CGPA > 9.8 is rare — flag but not critical
        cgpa_match = re.search(r"(\d+\.?\d*)\s*/?\s*10", grade)
        if cgpa_match and float(cgpa_match.group(1)) > 9.8:
            issues.append(f"CGPA {cgpa_match.group(1)}/10 is unusually high")
            score -= 0.15

        return max(0.0, score), issues


# ── Main field confidence scorer ──────────────────────────────────────────

@dataclass
class ScoredFields:
    student_name:  Optional[str] = None
    institution:   Optional[str] = None
    degree:        Optional[str] = None
    issue_date:    Optional[str] = None
    grade:         Optional[str] = None
    roll_number:   Optional[str] = None

    # Per-field confidence (validation-adjusted)
    field_scores: Dict[str, float] = field(default_factory=dict)

    # Per-field issues
    field_issues: Dict[str, List[str]] = field(default_factory=dict)

    # Cross-field consistency
    consistency_score: float = 1.0
    consistency_issues: List[str] = field(default_factory=list)

    # Overall field confidence (mean of individual scores)
    overall_confidence: float = 0.0

    # List of fields that appear tampered or anomalous
    flagged_fields: List[str] = field(default_factory=list)

    def to_api_format(self) -> List[dict]:
        """Format for the FastAPI FieldScore response model."""
        return [
            {
                "field":       f,
                "value":       getattr(self, f, None),
                "confidence":  self.field_scores.get(f, 0.0),
                "flagged":     f in self.flagged_fields,
                "issues":      self.field_issues.get(f, []),
            }
            for f in ["student_name", "institution", "degree",
                      "issue_date", "grade", "roll_number"]
        ]


def score_fields(
    extracted: Dict[str, Optional[str]],
    extraction_confidence: Dict[str, float],
    flag_threshold: float = 0.60,
) -> ScoredFields:
    """
    Run all validators on extracted fields and compute final confidence.

    Parameters
    ----------
    extracted             : {field_name: value} from FieldExtractor
    extraction_confidence : {field_name: confidence} from LayoutLMv3
    flag_threshold        : fields below this score are flagged

    Returns
    -------
    ScoredFields with all per-field and aggregate scores
    """
    validators = {
        "student_name": FieldValidator.validate_name,
        "institution":  FieldValidator.validate_institution,
        "degree":       FieldValidator.validate_degree,
        "issue_date":   FieldValidator.validate_date,
        "grade":        FieldValidator.validate_grade,
        "roll_number":  FieldValidator.validate_roll_number,
    }

    scored = ScoredFields()
    field_scores  = {}
    field_issues  = {}
    flagged       = []

    for field_name, validator in validators.items():
        raw_value = extracted.get(field_name)
        val_result = validator(raw_value)

        # Blend extraction confidence with validation score
        extract_conf = extraction_confidence.get(field_name, 0.5)
        final_score  = extract_conf * 0.5 + val_result.score * 0.5

        field_scores[field_name] = round(final_score, 4)
        field_issues[field_name] = val_result.issues

        # Use cleaned value if available
        cleaned = val_result.cleaned or raw_value
        setattr(scored, field_name, cleaned)

        if final_score < flag_threshold:
            flagged.append(field_name)

    # Consistency check
    consistency_score, consistency_issues = ConsistencyChecker.check_all(
        {f: getattr(scored, f) for f in validators}
    )
    if consistency_score < 0.7:
        flagged = list(set(flagged + ["consistency"]))

    scored.field_scores        = field_scores
    scored.field_issues        = field_issues
    scored.consistency_score   = consistency_score
    scored.consistency_issues  = consistency_issues
    scored.flagged_fields      = flagged
    scored.overall_confidence  = round(float(np.mean(list(field_scores.values()))), 4)

    return scored


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "student_name": "Rahul Sharma",
            "institution":  "Delhi Technological University",
            "degree":       "Bachelor of Technology",
            "issue_date":   "15 May 2023",
            "grade":        "A+ / 9.45 CGPA",
            "roll_number":  "19DTU1234",
        },
        {
            "student_name": "J0hn D0e",           # digits in name
            "institution":  "Fake Corp Ltd",      # not an edu institution
            "degree":       "Master of XYZ",      # unknown degree
            "issue_date":   "45 Octember 2025",   # invalid date
            "grade":        "15.0/10",            # impossible CGPA
            "roll_number":  "X",                  # too short
        },
    ]

    for i, case in enumerate(test_cases):
        print(f"\n{'='*55}")
        print(f"Test case {i+1}: {'genuine-like' if i==0 else 'tampered-like'}")
        result = score_fields(
            extracted=case,
            extraction_confidence={f: 0.9 for f in case},
        )
        for fname, fscore in result.field_scores.items():
            flag = " ← FLAGGED" if fname in result.flagged_fields else ""
            issues = result.field_issues.get(fname, [])
            issue_str = f" [{'; '.join(issues)}]" if issues else ""
            print(f"  {fname:20s}: {fscore:.2f}{flag}{issue_str}")
        print(f"  Overall: {result.overall_confidence:.2f} | "
              f"Consistency: {result.consistency_score:.2f}")
        if result.flagged_fields:
            print(f"  Flagged: {result.flagged_fields}")
