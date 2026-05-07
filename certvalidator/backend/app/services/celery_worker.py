"""
backend/app/services/celery_worker.py

Celery task queue for asynchronous certificate verification.

Architecture:
  FastAPI ──POST /verify──► Celery queue (Redis)
                                    │
                              Celery worker
                                    │
                            CertificatePipeline
                                    │
                            Save to PostgreSQL
                                    │
                         GET /verify/{id} polls result

Why Celery?
  The inference pipeline takes 5-30 seconds (OCR + LayoutLM + Mistral).
  A synchronous endpoint would timeout on slow connections.
  Celery decouples submission from execution.

Start worker:
    celery -A backend.app.services.celery_worker worker \
           --loglevel=info --concurrency=2

Monitor:
    celery -A backend.app.services.celery_worker flower
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from celery import Celery
from celery.utils.log import get_task_logger

from backend.app.core.config import get_settings

cfg    = get_settings()
logger = get_task_logger(__name__)

# ── Celery app ────────────────────────────────────────────────────────────
celery_app = Celery(
    "certvalidator",
    broker=cfg.redis_url,
    backend=cfg.redis_url.replace("/0", "/1"),
)

celery_app.conf.update(
    task_serializer       = "json",
    result_serializer     = "json",
    accept_content        = ["json"],
    result_expires        = 3600,          # keep results 1 hour
    task_track_started    = True,
    task_acks_late        = True,          # re-queue on worker crash
    worker_prefetch_multiplier = 1,        # one task at a time per worker
    task_soft_time_limit  = 120,           # 2 min soft limit
    task_time_limit       = 180,           # 3 min hard limit
    broker_connection_retry_on_startup = True,
)


# ── Main verification task ────────────────────────────────────────────────

@celery_app.task(
    bind=True,
    name="certvalidator.verify_certificate",
    max_retries=2,
    default_retry_delay=5,
)
def verify_certificate_task(
    self,
    verification_id: str,
    file_path: str,
    file_suffix: str = ".jpg",
    user_id: Optional[str] = None,
) -> dict:
    """
    Main Celery task for certificate verification.

    Parameters
    ----------
    verification_id : UUID string for this verification run
    file_path       : path to the uploaded file on disk
    file_suffix     : file extension (.jpg, .png, .pdf)
    user_id         : requesting user's UUID (for audit log)

    Returns
    -------
    dict : full verification result (stored in Redis + PostgreSQL)
    """
    logger.info(f"Starting verification {verification_id}")

    try:
        # Update state to PROGRESS
        self.update_state(
            state="PROGRESS",
            meta={"step": "preprocessing", "verification_id": verification_id},
        )

        # Import here to avoid loading models at import time
        from backend.app.services.inference import get_pipeline

        pipeline = get_pipeline()
        image_bytes = Path(file_path).read_bytes()

        self.update_state(
            state="PROGRESS",
            meta={"step": "running_inference", "verification_id": verification_id},
        )

        result = pipeline.verify(
            image_bytes=image_bytes,
            image_suffix=file_suffix,
            generate_heatmap=True,
        )

        self.update_state(
            state="PROGRESS",
            meta={"step": "saving_results", "verification_id": verification_id},
        )

        # Persist to PostgreSQL (sync, via SQLAlchemy sync session)
        _persist_result(verification_id, result, user_id)

        # Generate PDF report
        _generate_pdf(verification_id, result)

        api_response = result.to_api_response()
        api_response["status"] = "done"

        logger.info(
            f"Verification {verification_id} complete: "
            f"{result.verdict} {result.trust_score:.1f}"
        )
        return api_response

    except Exception as exc:
        logger.error(f"Verification {verification_id} failed: {exc}")
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {
                "verification_id": verification_id,
                "status": "error",
                "detail": str(exc),
            }


# ── Persistence helpers ───────────────────────────────────────────────────

def _persist_result(verification_id: str, result, user_id: Optional[str]):
    """
    Save verification result to PostgreSQL using a sync session.
    Celery workers use sync SQLAlchemy (not async).
    """
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from backend.app.models.schema import Verification, Certificate
        import hashlib

        # Sync DB URL (replace asyncpg with psycopg2)
        sync_url = cfg.database_url.replace(
            "postgresql+asyncpg://", "postgresql://"
        )
        engine  = create_engine(sync_url, pool_pre_ping=True)

        with Session(engine) as session:
            # Upsert Certificate record
            cert = Certificate(
                id                   = uuid.UUID(verification_id),
                upload_filename      = f"{verification_id}.jpg",
                file_hash            = result.file_hash,
                extracted_name       = result.fields.get("student_name"),
                extracted_institution= result.fields.get("institution"),
                extracted_degree     = result.fields.get("degree"),
                extracted_issue_date = result.fields.get("issue_date"),
                extracted_grade      = result.fields.get("grade"),
                extracted_roll_number= result.fields.get("roll_number"),
            )
            session.merge(cert)

            # Verification record
            verif = Verification(
                id               = uuid.UUID(str(uuid.uuid4())),
                certificate_id   = uuid.UUID(verification_id),
                verdict          = result.verdict,
                trust_score      = result.trust_score,
                forgery_score    = result.forgery_score,
                field_confidence = result.field_confidence,
                nlp_anomaly_score= result.nlp_anomaly_score,
                institution_match= result.institution_matched,
                field_scores     = result.field_scores,
                tamper_regions   = result.tamper_regions,
                nlp_reasoning    = result.nlp_reasoning,
                ocr_raw_text     = result.ocr_raw_text,
                heatmap_path     = result.heatmap_path,
                processing_time_s= result.processing_time_s,
                model_versions   = result.model_versions,
                requested_by     = uuid.UUID(user_id) if user_id else None,
            )
            session.add(verif)
            session.commit()

    except Exception as e:
        logger.warning(f"DB persist failed (non-fatal): {e}")


def _generate_pdf(verification_id: str, result):
    """Generate and save PDF report as a background step."""
    try:
        from backend.app.services.report_generator import generate_pdf_report
        report_path = Path(cfg.report_dir) / f"{verification_id}_report.pdf"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_pdf_report(result, str(report_path))
        result.report_pdf_path = str(report_path)
    except Exception as e:
        logger.warning(f"PDF generation failed (non-fatal): {e}")


# ── Type hint fix ─────────────────────────────────────────────────────────
from typing import Optional
