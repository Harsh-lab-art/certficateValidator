"""initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2026-04-14
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id",               UUID(as_uuid=True), primary_key=True),
        sa.Column("email",            sa.String(300),  nullable=False, unique=True),
        sa.Column("hashed_password",  sa.String(200),  nullable=False),
        sa.Column("full_name",        sa.String(300)),
        sa.Column("role",             sa.String(50),   server_default="verifier"),
        sa.Column("is_active",        sa.Boolean(),    server_default="true"),
        sa.Column("api_key",          sa.String(64),   unique=True),
        sa.Column("rate_limit_rpm",   sa.Integer(),    server_default="60"),
        sa.Column("created_at",       sa.DateTime(),   server_default=sa.func.now()),
        sa.Column("last_login",       sa.DateTime()),
    )
    op.create_index("ix_users_email",   "users",  ["email"])
    op.create_index("ix_users_api_key", "users",  ["api_key"])

    op.create_table(
        "institutions",
        sa.Column("id",                  UUID(as_uuid=True), primary_key=True),
        sa.Column("name",                sa.String(300), nullable=False),
        sa.Column("short_name",          sa.String(50)),
        sa.Column("country",             sa.String(100), server_default="India"),
        sa.Column("accreditation_body",  sa.String(200)),
        sa.Column("is_active",           sa.Boolean(),   server_default="true"),
        sa.Column("domain",              sa.String(200)),
        sa.Column("logo_hash",           sa.String(64)),
        sa.Column("created_at",          sa.DateTime(),  server_default=sa.func.now()),
        sa.Column("updated_at",          sa.DateTime(),  server_default=sa.func.now()),
    )
    op.create_index("ix_institutions_name", "institutions", ["name"])

    op.create_table(
        "certificates",
        sa.Column("id",                   UUID(as_uuid=True), primary_key=True),
        sa.Column("upload_filename",       sa.String(500), nullable=False),
        sa.Column("file_hash",             sa.String(64),  nullable=False),
        sa.Column("file_size_bytes",       sa.Integer()),
        sa.Column("mime_type",             sa.String(100)),
        sa.Column("extracted_name",        sa.String(300)),
        sa.Column("extracted_institution", sa.String(300)),
        sa.Column("extracted_degree",      sa.String(300)),
        sa.Column("extracted_discipline",  sa.String(300)),
        sa.Column("extracted_roll_number", sa.String(100)),
        sa.Column("extracted_issue_date",  sa.String(100)),
        sa.Column("extracted_grade",       sa.String(50)),
        sa.Column("extracted_cgpa",        sa.Float()),
        sa.Column("institution_id",        UUID(as_uuid=True),
                  sa.ForeignKey("institutions.id"), nullable=True),
        sa.Column("content_hash",          sa.String(64)),
        sa.Column("ipfs_cid",              sa.String(100)),
        sa.Column("chain_tx_hash",         sa.String(100)),
        sa.Column("uploaded_by",           UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at",            sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_certificates_file_hash",    "certificates", ["file_hash"])
    op.create_index("ix_certificates_content_hash", "certificates", ["content_hash"])

    op.create_table(
        "verifications",
        sa.Column("id",                  UUID(as_uuid=True), primary_key=True),
        sa.Column("certificate_id",      UUID(as_uuid=True),
                  sa.ForeignKey("certificates.id"), nullable=False),
        sa.Column("verdict",             sa.String(20),  nullable=False),
        sa.Column("trust_score",         sa.Float(),     nullable=False),
        sa.Column("forgery_score",       sa.Float()),
        sa.Column("field_confidence",    sa.Float()),
        sa.Column("nlp_anomaly_score",   sa.Float()),
        sa.Column("institution_match",   sa.Boolean()),
        sa.Column("field_scores",        sa.JSON()),
        sa.Column("tamper_regions",      sa.JSON()),
        sa.Column("nlp_reasoning",       sa.Text()),
        sa.Column("ocr_raw_text",        sa.Text()),
        sa.Column("heatmap_path",        sa.String(500)),
        sa.Column("report_pdf_path",     sa.String(500)),
        sa.Column("processing_time_s",   sa.Float()),
        sa.Column("model_versions",      sa.JSON()),
        sa.Column("requested_by",        UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at",          sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_verifications_cert",    "verifications", ["certificate_id"])
    op.create_index("ix_verifications_verdict", "verifications", ["verdict"])

    op.create_table(
        "audit_log",
        sa.Column("id",            UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id",       UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action",        sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(50)),
        sa.Column("resource_id",   sa.String(100)),
        sa.Column("ip_address",    sa.String(45)),
        sa.Column("user_agent",    sa.String(500)),
        sa.Column("status",        sa.String(20)),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at",    sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("verifications")
    op.drop_table("certificates")
    op.drop_table("institutions")
    op.drop_table("users")
