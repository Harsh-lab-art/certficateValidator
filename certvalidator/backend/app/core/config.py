"""backend/app/core/config.py — Windows-compatible settings"""
from __future__ import annotations
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings

# ── Auto-detect Tesseract on any machine ──────────────────────────────────
def _configure_tesseract():
    """
    Find tesseract.exe on the current machine and configure pytesseract.
    Checks (in order):
      1. Already in PATH (works on Linux/Mac/Windows if installed normally)
      2. Common Windows install locations
      3. TESSERACT_CMD environment variable (user can set this manually)
    """
    # If user set an env var, use that
    env_cmd = os.environ.get("TESSERACT_CMD", "")
    if env_cmd and os.path.exists(env_cmd):
        _apply(env_cmd)
        return

    # Already in PATH?
    if shutil.which("tesseract"):
        return  # pytesseract will find it automatically

    # Common Windows install locations (not hardcoded paths — built dynamically)
    candidates = []
    for prog_dir in [
        os.environ.get("PROGRAMFILES", r"C:\Program Files"),
        os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs"),
        os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Programs"),
        os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local"),
    ]:
        if prog_dir:
            candidates.append(os.path.join(prog_dir, "Tesseract-OCR", "tesseract.exe"))

    for path in candidates:
        if os.path.exists(path):
            _apply(path)
            return

    # Not found — pytesseract will raise a clear error when OCR is attempted
    print("[Config] Tesseract not found. Install from: "
          "https://github.com/UB-Mannheim/tesseract/wiki")


def _apply(tesseract_exe: str):
    """Set tesseract path in pytesseract and add its folder to PATH."""
    folder = str(Path(tesseract_exe).parent)
    if folder not in os.environ.get("PATH", ""):
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe
    except ImportError:
        pass


_configure_tesseract()


class Settings(BaseSettings):
    app_name:    str = "CertValidator"
    app_version: str = "0.3.0"
    debug:       bool = False
    environment: str = "development"

    database_url: str = Field(
        default="postgresql+asyncpg://certval:certval@localhost:5432/certvalidator",
        alias="DATABASE_URL",
    )

    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    jwt_secret:     str = Field(default="CHANGE_ME_32_CHAR_SECRET_KEY_HERE_!", alias="JWT_SECRET_KEY")
    jwt_algorithm:  str = "HS256"
    jwt_expire_min: int = 1440

    upload_dir:    str = "uploads"
    heatmap_dir:   str = "uploads/heatmaps"
    report_dir:    str = "uploads/reports"
    max_upload_mb: int = 20

    checkpoint_dir: str = "ml/models/checkpoints"

    # CORS — open by default for dev
    cors_origins: List[str] = ["*"]

    rate_limit_per_min: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        populate_by_name = True
        extra = "ignore"

    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def heatmap_path(self) -> Path:
        p = Path(self.heatmap_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def report_path(self) -> Path:
        p = Path(self.report_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
