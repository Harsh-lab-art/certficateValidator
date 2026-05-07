"""backend/app/main.py — Windows-compatible, graceful model loading"""
from __future__ import annotations
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app.core.config import get_settings
from backend.app.api.v1.routes import auth_router, verify_router, inst_router, stats_router

cfg = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Gracefully attempt model loading — never crash startup
    try:
        from backend.app.services.inference import get_pipeline
        get_pipeline(checkpoint_dir=cfg.checkpoint_dir)
        print("[CertValidator] Pipeline loaded successfully.")
    except Exception as e:
        print(f"[CertValidator] Pipeline not loaded (models not trained yet): {e}")
        print("[CertValidator] Upload will use heuristic fallback — fully functional.")
    yield

app = FastAPI(
    title=cfg.app_name, version=cfg.app_version,
    description="AI-powered academic certificate authenticity validator.",
    lifespan=lifespan, docs_url="/docs", redoc_url="/redoc",
)

# CORS — allow all origins during development (fixes Windows localhost issues)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.time()
    r  = await call_next(request)
    r.headers["X-Process-Time"] = f"{(time.time()-t0)*1000:.1f}ms"
    return r

PREFIX = "/api/v1"
app.include_router(auth_router,   prefix=PREFIX)
app.include_router(verify_router, prefix=PREFIX)
app.include_router(inst_router,   prefix=PREFIX)
app.include_router(stats_router,  prefix=PREFIX)

@app.get("/health", tags=["System"])
async def health():
    try:
        from backend.app.services.inference import _pipeline
        models_loaded = _pipeline is not None
    except Exception:
        models_loaded = False
    return {
        "status": "ok",
        "version": cfg.app_version,
        "timestamp": time.time(),
        "models_loaded": models_loaded,
    }

@app.exception_handler(Exception)
async def err(request: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )
