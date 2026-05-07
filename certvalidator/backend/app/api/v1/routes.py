"""
backend/app/api/v1/routes.py
Windows-compatible, resilient. No module-level hash calls.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from backend.app.core.config import get_settings
from backend.app.core.auth import (
    create_access_token, verify_password, hash_password,
    get_current_user_optional, require_auth,
)
from backend.app.services.institution_db import (
    lookup_institution, search_institutions, list_all_institutions,
)

cfg = get_settings()

# In-memory result store
_results: dict = {}

# ── Pydantic models ────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class VerificationResponse(BaseModel):
    verification_id:     str
    status:              str
    verdict:             Optional[str]   = None
    trust_score:         Optional[float] = None
    explanation:         Optional[str]   = None
    processing_time_s:   Optional[float] = None
    forgery_score:       Optional[float] = None
    field_confidence:    Optional[float] = None
    nlp_anomaly_score:   Optional[float] = None
    institution_matched: Optional[bool]  = None
    field_scores:        list  = []
    tamper_regions:      list  = []
    nlp_reasoning:       Optional[str]   = None
    ocr_raw_text:        Optional[str]   = None
    contributions:       dict  = {}
    confidence_interval: Optional[float] = None
    heatmap_url:         Optional[str]   = None
    report_pdf_url:      Optional[str]   = None
    model_versions:      dict  = {}


# ── Auth router ────────────────────────────────────────────────────────────

auth_router = APIRouter(prefix="/auth", tags=["Auth"])

# LAZY user store — only hashed when first login happens, NOT at import time
_users: dict = {}

def _get_users() -> dict:
    """Initialize user store lazily to avoid bcrypt crash at import."""
    global _users
    if not _users:
        _users = {
            "admin@certvalidator.ai": {
                "id":              str(uuid.uuid4()),
                "email":           "admin@certvalidator.ai",
                "hashed_password": hash_password("admin123"),
                "role":            "admin",
            }
        }
    return _users


@auth_router.post("/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    users = _get_users()
    user  = users.get(form.username)
    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": user["id"], "role": user["role"]})
    return TokenResponse(access_token=token, expires_in=cfg.jwt_expire_min * 60)


@auth_router.post("/register", status_code=201)
async def register(email: str = Form(...), password: str = Form(...),
                   full_name: str = Form(...)):
    users = _get_users()
    if email in users:
        raise HTTPException(400, "Email already registered")
    uid = str(uuid.uuid4())
    users[email] = {
        "id": uid, "email": email,
        "hashed_password": hash_password(password),
        "role": "verifier", "full_name": full_name,
    }
    token = create_access_token({"sub": uid, "role": "verifier"})
    return {"id": uid, "email": email, "access_token": token}


@auth_router.get("/me")
async def me(user: dict = Depends(require_auth)):
    return user


# ── Verify router ──────────────────────────────────────────────────────────

verify_router = APIRouter(prefix="/verify", tags=["Verification"])

ALLOWED_TYPES = {
    "image/jpeg", "image/png", "image/tiff",
    "application/pdf", "image/bmp",
    "image/jpg", "application/octet-stream",
}


@verify_router.post("", response_model=VerificationResponse, status_code=202)
async def submit(
    request:          Request,
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    user:             Optional[dict] = Depends(get_current_user_optional),
):
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file received")
    if len(data) > cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {cfg.max_upload_mb}MB limit")

    original_name = file.filename or "upload.jpg"
    suffix = Path(original_name).suffix.lower() or ".jpg"

    cfg.upload_path().mkdir(parents=True, exist_ok=True)
    vid = str(uuid.uuid4())
    upload_path = cfg.upload_path() / f"{vid}{suffix}"
    upload_path.write_bytes(data)

    _results[vid] = {"status": "processing", "verification_id": vid}
    background_tasks.add_task(_run_inline, vid, data, suffix)

    return VerificationResponse(verification_id=vid, status="processing")


async def _run_inline(vid: str, data: bytes, suffix: str):
    try:
        from backend.app.services.inference import get_pipeline
        from backend.app.services.report_generator import generate_pdf_report

        pipeline = get_pipeline()
        result   = await pipeline.verify_async(data, suffix)
        api_resp = result.to_api_response()
        api_resp["status"] = "done"

        try:
            report_path = cfg.report_path() / f"{vid}_report.pdf"
            generate_pdf_report(result, str(report_path))
            api_resp["report_pdf_url"] = f"/api/v1/verify/{vid}/report"
        except Exception as e:
            print(f"[Routes] PDF generation failed (non-fatal): {e}")

        _results[vid] = api_resp

    except Exception as exc:
        import traceback
        traceback.print_exc()
        _results[vid] = {
            "verification_id": vid,
            "status": "error",
            "detail": str(exc),
        }


@verify_router.get("/{verification_id}", response_model=VerificationResponse)
async def get_result(verification_id: str):
    cached = _results.get(verification_id)
    if cached is None:
        raise HTTPException(404, "Verification not found — poll again shortly")
    if cached.get("status") == "processing":
        return VerificationResponse(verification_id=verification_id, status="processing")
    if cached.get("status") == "error":
        raise HTTPException(500, cached.get("detail", "Verification failed"))
    try:
        return VerificationResponse(**{
            k: v for k, v in cached.items()
            if k in VerificationResponse.model_fields
        })
    except Exception:
        return VerificationResponse(
            verification_id=verification_id, status="done",
            verdict=cached.get("verdict"),
            trust_score=cached.get("trust_score"),
            explanation=cached.get("explanation"),
            field_scores=cached.get("field_scores", []),
            nlp_reasoning=cached.get("nlp_reasoning"),
        )


@verify_router.get("/{verification_id}/heatmap")
async def get_heatmap(verification_id: str):
    path = cfg.heatmap_path() / f"{verification_id}_heatmap.png"
    if not path.exists():
        raise HTTPException(404, "Heatmap not available")
    return FileResponse(str(path), media_type="image/png",
                        filename=f"heatmap_{verification_id[:8]}.png")


@verify_router.get("/{verification_id}/report")
async def get_report(verification_id: str):
    path = cfg.report_path() / f"{verification_id}_report.pdf"
    if not path.exists():
        raise HTTPException(404, "Report not yet generated")
    return FileResponse(str(path), media_type="application/pdf",
                        filename=f"certvalidator_report_{verification_id[:8]}.pdf")


@verify_router.get("", summary="Verification history")
async def history(limit: int = 20, offset: int = 0,
                  user: Optional[dict] = Depends(get_current_user_optional)):
    completed = [v for v in _results.values()
                 if isinstance(v, dict) and v.get("verdict")]
    return {"total": len(completed), "results": completed[offset: offset + limit]}


# ── Institution router ─────────────────────────────────────────────────────

inst_router = APIRouter(prefix="/institutions", tags=["Institutions"])


@inst_router.get("/search")
async def search(q: str, limit: int = 10):
    return {"query": q, "total": 0, "results": search_institutions(q, limit)}


@inst_router.get("")
async def list_inst():
    return {"institutions": list_all_institutions()}


@inst_router.get("/lookup")
async def lookup(name: str):
    return lookup_institution(name)


@inst_router.post("", status_code=201)
async def register_institution(
    name:          str = Form(...),
    short_name:    Optional[str] = Form(None),
    country:       str = Form("India"),
    accreditation: Optional[str] = Form(None),
    seal_image:    Optional[UploadFile] = File(None),
    user:          dict = Depends(require_auth),
):
    logo_hash = None
    if seal_image:
        logo_hash = hashlib.sha256(await seal_image.read()).hexdigest()
    return {
        "id": str(uuid.uuid4()), "name": name,
        "short_name": short_name, "country": country,
        "accreditation": accreditation, "logo_hash": logo_hash,
    }


# ── Stats router ───────────────────────────────────────────────────────────

stats_router = APIRouter(prefix="/stats", tags=["Stats"])


@stats_router.get("")
async def get_stats():
    completed  = [v for v in _results.values()
                  if isinstance(v, dict) and v.get("verdict")]
    total      = len(completed)
    genuine    = sum(1 for r in completed if r.get("verdict") == "GENUINE")
    fake       = sum(1 for r in completed if r.get("verdict") == "FAKE")
    suspicious = sum(1 for r in completed if r.get("verdict") == "SUSPICIOUS")
    avg_score  = (sum(r.get("trust_score", 0) for r in completed) / total
                  if total else 0.0)
    return {
        "total_verifications": total,
        "genuine":             genuine,
        "fake":                fake,
        "suspicious":          suspicious,
        "avg_trust_score":     round(avg_score, 1),
        "fake_rate":           round(fake / total * 100, 1) if total else 0,
    }
