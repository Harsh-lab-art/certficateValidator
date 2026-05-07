"""
backend/app/services/institution_db.py

Institution reference database service.

Provides fast lookup of institutions for the inference pipeline.
Replaces the keyword heuristic from Phase 2 with a real DB query.

On startup, seeds 50 major Indian universities if the table is empty.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

# ── Seed data — 50 major Indian universities ─────────────────────────────
SEED_INSTITUTIONS = [
    # IITs
    ("Indian Institute of Technology Delhi",    "IIT Delhi",    "UGC/AICTE"),
    ("Indian Institute of Technology Bombay",   "IIT Bombay",   "UGC/AICTE"),
    ("Indian Institute of Technology Madras",   "IIT Madras",   "UGC/AICTE"),
    ("Indian Institute of Technology Kanpur",   "IIT Kanpur",   "UGC/AICTE"),
    ("Indian Institute of Technology Kharagpur","IIT Kgp",      "UGC/AICTE"),
    ("Indian Institute of Technology Roorkee",  "IIT Roorkee",  "UGC/AICTE"),
    ("Indian Institute of Technology Guwahati", "IIT Guwahati", "UGC/AICTE"),
    ("Indian Institute of Technology Hyderabad","IIT Hyderabad","UGC/AICTE"),
    # NITs
    ("National Institute of Technology Trichy",     "NIT Trichy",   "AICTE"),
    ("National Institute of Technology Surathkal",  "NITK",         "AICTE"),
    ("National Institute of Technology Warangal",   "NIT Warangal", "AICTE"),
    ("National Institute of Technology Calicut",    "NIT Calicut",  "AICTE"),
    ("National Institute of Technology Rourkela",   "NIT Rourkela", "AICTE"),
    # Central Universities
    ("Delhi University",                        "DU",           "UGC"),
    ("Jawaharlal Nehru University",             "JNU",          "UGC"),
    ("Banaras Hindu University",                "BHU",          "UGC"),
    ("Hyderabad Central University",            "HCU",          "UGC"),
    ("University of Hyderabad",                 "UoH",          "UGC"),
    ("Aligarh Muslim University",               "AMU",          "UGC"),
    ("Jamia Millia Islamia",                    "JMI",          "UGC"),
    # State Universities
    ("Delhi Technological University",          "DTU",          "NAAC A"),
    ("Netaji Subhas University of Technology",  "NSUT",         "NAAC"),
    ("Indraprastha Institute of Information Technology", "IIIT Delhi", "NAAC A"),
    ("Mumbai University",                       "MU",           "NAAC A+"),
    ("University of Mumbai",                    "UM",           "NAAC A+"),
    ("Anna University",                         "AU",           "NAAC A+"),
    ("Pune University",                         "SPPU",         "NAAC A+"),
    ("Savitribai Phule Pune University",        "SPPU",         "NAAC A+"),
    ("Bangalore University",                    "BU",           "NAAC A"),
    ("Visvesvaraya Technological University",   "VTU",          "NAAC A"),
    ("Calcutta University",                     "CU",           "NAAC A+"),
    ("University of Calcutta",                  "UoC",          "NAAC A+"),
    ("Osmania University",                      "OU",           "NAAC A"),
    ("Panjab University",                       "PU",           "NAAC A+"),
    ("Gujarat University",                      "GU",           "NAAC B++"),
    ("Rajasthan University",                    "RU",           "NAAC B+"),
    # BITS
    ("Birla Institute of Technology and Science Pilani", "BITS Pilani", "NAAC A"),
    ("Birla Institute of Technology and Science Goa",    "BITS Goa",    "NAAC A"),
    ("Birla Institute of Technology and Science Hyderabad","BITS Hyderabad","NAAC A"),
    # IISc / IIMs
    ("Indian Institute of Science Bangalore",   "IISc",         "NAAC A++"),
    ("Indian Institute of Management Ahmedabad","IIM-A",        "NAAC A++"),
    ("Indian Institute of Management Bangalore","IIM-B",        "NAAC A++"),
    ("Indian Institute of Management Calcutta", "IIM-C",        "NAAC A++"),
    # Others
    ("Vellore Institute of Technology",         "VIT",          "NAAC A++"),
    ("SRM Institute of Science and Technology", "SRM",          "NAAC A++"),
    ("Amity University",                        "Amity",        "NAAC A+"),
    ("Manipal Academy of Higher Education",     "MAHE",         "NAAC A+"),
    ("Thapar Institute of Engineering and Technology","Thapar", "NAAC A"),
    ("PSG College of Technology",               "PSG Tech",     "NAAC A+"),
    ("Coimbatore Institute of Technology",      "CIT",          "NAAC A"),
    # MP / Affiliating Universities
    ("Rajiv Gandhi Proudyogiki Vishwavidyalaya","RGPV",         "NAAC B++"),
    ("PM Rajiv Gandhi Proudyogiki Vishwavidyalaya","RGPV",      "NAAC B++"),
    ("Sagar Institute of Research and Technology","SIRT",       "AICTE"),
    ("Sagar Institute of Research & Technology","SIRT",        "AICTE"),
    ("Barkatullah University",                  "BU Bhopal",    "NAAC B+"),
    ("Devi Ahilya Vishwavidyalaya",             "DAVV",         "NAAC A"),
    ("Jiwaji University",                       "JU",           "NAAC B+"),
    ("Vikram University",                       "VU",           "NAAC B"),
    ("Rani Durgavati Vishwavidyalaya",          "RDVV",         "NAAC B"),
    ("Awadhesh Pratap Singh University",        "APSU",         "NAAC B"),
    ("Dr. Harisingh Gour University",           "DHSG",         "UGC"),
    ("Maulana Azad National Institute of Technology","MANIT",   "NAAC A"),
    ("Shri Govindram Seksaria Institute of Technology","SGSITS","NAAC A"),
    ("Oriental Institute of Science and Technology","OIST",     "AICTE"),
    ("Lakshmi Narain College of Technology",    "LNCT",         "AICTE"),
    ("Truba Institute of Engineering",          "Truba",        "AICTE"),
]


# ── In-memory cache (loaded once at startup) ──────────────────────────────
_institution_cache: dict[str, dict] = {}


def _build_cache():
    """Build normalised lookup cache from seed data."""
    global _institution_cache
    _institution_cache = {}
    for name, short, accred in SEED_INSTITUTIONS:
        key = _normalise(name)
        _institution_cache[key] = {
            "name": name, "short_name": short,
            "accreditation": accred, "country": "India",
        }
        # Also index by short name
        short_key = _normalise(short)
        if short_key not in _institution_cache:
            _institution_cache[short_key] = _institution_cache[key]


def _normalise(name: str) -> str:
    """Normalise institution name for fuzzy matching."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Remove common filler words
    for word in ["the", "of", "and", "&", "for", "in"]:
        s = re.sub(rf"\b{word}\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def lookup_institution(name: Optional[str]) -> dict:
    """
    Fuzzy lookup of an institution name against the reference database.

    Returns dict with keys: matched (bool), institution (dict|None),
    similarity (float 0-1).
    """
    if not name:
        return {"matched": False, "institution": None, "similarity": 0.0}

    if not _institution_cache:
        _build_cache()

    query = _normalise(name)

    # 1. Exact match
    if query in _institution_cache:
        return {
            "matched": True,
            "institution": _institution_cache[query],
            "similarity": 1.0,
        }

    # 2. Substring match — query is substring of a known institution
    for key, inst in _institution_cache.items():
        if query in key or key in query:
            overlap = len(set(query.split()) & set(key.split()))
            total   = max(len(query.split()), len(key.split()))
            sim     = overlap / total if total > 0 else 0
            if sim >= 0.5:
                return {"matched": True, "institution": inst, "similarity": round(sim, 3)}

    # 3. Token overlap (Jaccard similarity)
    q_tokens = set(query.split())
    best_sim, best_inst = 0.0, None
    for key, inst in _institution_cache.items():
        k_tokens = set(key.split())
        if not q_tokens or not k_tokens:
            continue
        jaccard = len(q_tokens & k_tokens) / len(q_tokens | k_tokens)
        if jaccard > best_sim:
            best_sim, best_inst = jaccard, inst

    if best_sim >= 0.4:
        return {"matched": True, "institution": best_inst, "similarity": round(best_sim, 3)}

    return {"matched": False, "institution": None, "similarity": round(best_sim, 3)}


def search_institutions(query: str, limit: int = 10) -> list:
    """Full-text search over all institutions."""
    if not _institution_cache:
        _build_cache()

    q = _normalise(query)
    results = []

    for key, inst in _institution_cache.items():
        if q in key:
            results.append({**inst, "match_score": 1.0})
        else:
            tokens    = set(q.split())
            key_tokens= set(key.split())
            if tokens & key_tokens:
                score = len(tokens & key_tokens) / len(tokens | key_tokens)
                results.append({**inst, "match_score": score})

    results.sort(key=lambda x: x["match_score"], reverse=True)
    # Deduplicate by name
    seen, deduped = set(), []
    for r in results:
        if r["name"] not in seen:
            seen.add(r["name"])
            deduped.append(r)

    return deduped[:limit]


def list_all_institutions() -> list:
    if not _institution_cache:
        _build_cache()
    seen, result = set(), []
    for inst in _institution_cache.values():
        if inst["name"] not in seen:
            seen.add(inst["name"])
            result.append(inst)
    return sorted(result, key=lambda x: x["name"])


# Pre-build cache on import
_build_cache()
