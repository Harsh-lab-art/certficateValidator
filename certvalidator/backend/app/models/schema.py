"""
backend/app/models/schema.py

SQLAlchemy ORM models for CertValidator.

Tables:
  - institutions    → known issuing institutions (the ground-truth DB)
  - certificates    → uploaded certificate records
  - verifications   → verification run results with full score breakdown
  - users           → API users (for JWT auth)
  - audit_log       → every verification request (for compliance)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, func
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Institutions (ground truth reference)
# ---------------------------------------------------------------------------

class Institution(Base):
    __tablename__ = "institutions"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name        = Column(String(300), nullable=False, index=True)
    short_name  = Column(String(50))
    country     = Column(String(100), default="India")
    accreditation_body = Column(String(200))        # e.g. "NAAC A+", "UGC"
    is_active   = Column(Boolean, default=True)
    domain      = Column(String(200))               # official website domain
    logo_hash   = Column(String(64))                # SHA-256 of official seal
    created_at  = Column(DateTime, server_default=func.now())
    updated_at  = Column(DateTime, server_default=func.now(), onupdate=func.now())

    certificates = relationship("Certificate", back_populates="institution_ref")

    def __repr__(self):
        return f"<Institution {self.name}>"


# ---------------------------------------------------------------------------
# Certificates (uploaded documents)
# ---------------------------------------------------------------------------

class Certificate(Base):
    __tablename__ = "certificates"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    upload_filename = Column(String(500), nullable=False)
    file_hash       = Column(String(64), nullable=False, index=True)   # SHA-256 of raw upload
    file_size_bytes = Column(Integer)
    mime_type       = Column(String(100))

    # Extracted fields (populated after OCR + LayoutLM)
    extracted_name        = Column(String(300))
    extracted_institution = Column(String(300))
    extracted_degree      = Column(String(300))
    extracted_discipline  = Column(String(300))
    extracted_roll_number = Column(String(100))
    extracted_issue_date  = Column(String(100))
    extracted_grade       = Column(String(50))
    extracted_cgpa        = Column(Float)

    # Link to known institution (null if unrecognised)
    institution_id = Column(UUID(as_uuid=True), ForeignKey("institutions.id"), nullable=True)
    institution_ref = relationship("Institution", back_populates="certificates")

    # Blockchain anchor
    content_hash   = Column(String(64))   # SHA-256 of canonical fields
    ipfs_cid       = Column(String(100))  # IPFS content identifier (if anchored)
    chain_tx_hash  = Column(String(100))  # Polygon tx hash (if anchored)

    uploaded_by  = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at   = Column(DateTime, server_default=func.now())

    verifications = relationship("Verification", back_populates="certificate")

    def __repr__(self):
        return f"<Certificate {self.id} — {self.extracted_name}>"


# ---------------------------------------------------------------------------
# Verifications (each analysis run)
# ---------------------------------------------------------------------------

class Verification(Base):
    __tablename__ = "verifications"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    certificate_id = Column(UUID(as_uuid=True), ForeignKey("certificates.id"), nullable=False)
    certificate    = relationship("Certificate", back_populates="verifications")

    # Top-level verdict
    verdict        = Column(String(20), nullable=False)   # "GENUINE" | "FAKE" | "SUSPICIOUS" | "INCONCLUSIVE"
    trust_score    = Column(Float, nullable=False)         # 0.0 – 100.0

    # Sub-scores (each model's contribution)
    forgery_score          = Column(Float)   # EfficientNet+ELA — 0=genuine, 1=fake
    field_confidence       = Column(Float)   # LayoutLM extraction confidence
    nlp_anomaly_score      = Column(Float)   # Mistral reasoning anomaly
    institution_match      = Column(Boolean) # DB lookup result

    # Detailed breakdown stored as JSON
    field_scores           = Column(JSON)    # {name: 0.95, date: 0.40, grade: 0.99, ...}
    tamper_regions         = Column(JSON)    # [{x,y,w,h,confidence}, ...] from GradCAM
    nlp_reasoning          = Column(Text)    # Mistral's reasoning paragraph
    ocr_raw_text           = Column(Text)    # Full OCR output

    # Paths to generated assets
    heatmap_path           = Column(String(500))   # GradCAM overlay image path
    report_pdf_path        = Column(String(500))   # generated PDF report path

    # Performance
    processing_time_s      = Column(Float)
    model_versions         = Column(JSON)    # {ocr: "v1.2", forgery: "v1.0", ...}

    requested_by   = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at     = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<Verification {self.id} — {self.verdict} {self.trust_score:.1f}>"


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email         = Column(String(300), unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    full_name     = Column(String(300))
    role          = Column(String(50), default="verifier")   # "admin" | "verifier" | "institution"
    is_active     = Column(Boolean, default=True)
    api_key       = Column(String(64), unique=True, index=True)
    rate_limit_rpm = Column(Integer, default=60)             # requests per minute

    created_at    = Column(DateTime, server_default=func.now())
    last_login    = Column(DateTime)

    def __repr__(self):
        return f"<User {self.email}>"


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog(Base):
    __tablename__ = "audit_log"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id         = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action          = Column(String(100), nullable=False)   # "verify" | "upload" | "login"
    resource_type   = Column(String(50))                    # "certificate" | "verification"
    resource_id     = Column(String(100))
    ip_address      = Column(String(45))
    user_agent      = Column(String(500))
    status          = Column(String(20))                    # "success" | "error"
    error_message   = Column(Text)
    created_at      = Column(DateTime, server_default=func.now(), index=True)
