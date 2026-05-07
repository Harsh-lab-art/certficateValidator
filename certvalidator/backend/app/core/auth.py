"""
backend/app/core/auth.py

JWT token creation/validation and API key authentication.
Used by all protected endpoints.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Header, status
from fastapi.security import OAuth2PasswordBearer
from backend.app.core.config import get_settings

pwd_ctx = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    cfg = get_settings()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=cfg.jwt_expire_min)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, cfg.jwt_secret, algorithm=cfg.jwt_algorithm)


def decode_token(token: str) -> dict:
    cfg = get_settings()
    try:
        return jwt.decode(token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI dependencies ──────────────────────────────────────────────────

async def get_current_user_optional(
    token: Optional[str] = Depends(oauth2_scheme),
    x_api_key: Optional[str] = Header(default=None),
) -> Optional[dict]:
    """
    Returns user dict if authenticated, None if not.
    Accepts both JWT Bearer token and X-API-Key header.
    """
    if x_api_key:
        # API key auth (for Chrome extension)
        # TODO: look up in DB
        if x_api_key == "dev-api-key":
            return {"id": "dev", "role": "verifier"}
        return None

    if token:
        try:
            return decode_token(token)
        except HTTPException:
            return None

    return None


async def require_auth(
    user: Optional[dict] = Depends(get_current_user_optional),
) -> dict:
    """Dependency that raises 401 if not authenticated."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def require_admin(user: dict = Depends(require_auth)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
