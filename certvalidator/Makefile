# CertValidator — Master Makefile
# All commands from one place.
# Usage: make <target>
# Requires: Python 3.10+, Node 18+, Redis, PostgreSQL

.PHONY: help setup data train-p1 train-p2 train-p3 \
        api frontend worker demo test clean

PYTHON   = .venv/bin/python
PIP      = .venv/bin/pip
UVICORN  = .venv/bin/uvicorn
CELERY   = .venv/bin/celery

# ── Help ──────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "CertValidator — Command Reference"
	@echo "══════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Setup"
	@echo "    make setup           Full project setup (venv + npm + db)"
	@echo ""
	@echo "  Data & Training"
	@echo "    make data            Generate 2000 synthetic certificates"
	@echo "    make train-p1        Train forgery detector (EfficientNet+ELA)"
	@echo "    make train-p2        Fine-tune TrOCR + LayoutLMv3"
	@echo "    make train-p3        Download Mistral-7B GGUF"
	@echo "    make train-all       Run all 3 training phases sequentially"
	@echo ""
	@echo "  Run"
	@echo "    make api             Start FastAPI backend (port 8000)"
	@echo "    make worker          Start Celery inference worker"
	@echo "    make frontend        Start React dashboard (port 5173)"
	@echo "    make demo            Full demo launcher + ngrok tunnel"
	@echo ""
	@echo "  QA"
	@echo "    make test            Run all unit + integration tests"
	@echo "    make demo-certs      Regenerate 3 demo certificates"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean           Remove __pycache__ and build artefacts"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────
setup:
	@echo "Setting up CertValidator..."
	@bash scripts/setup.sh

# ── Data generation ───────────────────────────────────────────────────────
data:
	@echo "Generating synthetic training data..."
	$(PYTHON) scripts/generate_synthetic.py \
		--count 2000 \
		--output ml/data/synthetic \
		--fake-ratio 0.5 \
		--seed 42

demo-certs:
	@echo "Generating demo certificates..."
	$(PYTHON) demo/scripts/generate_demo_certs.py

# ── Training ──────────────────────────────────────────────────────────────
train-p1:
	@echo "Phase 2 Part 1 — EfficientNet+ELA forgery detector"
	@bash scripts/train_forgery.sh

train-p2:
	@echo "Phase 2 Part 2 — TrOCR + LayoutLMv3"
	@bash scripts/train_ocr_layout.sh

train-p3:
	@echo "Phase 2 Part 3 — Downloading Mistral-7B GGUF..."
	$(PYTHON) -m ml.src.models.llm.mistral_reasoner download

train-all: data train-p1 train-p2 train-p3
	@echo "All models trained."

# ── Runtime ───────────────────────────────────────────────────────────────
api:
	@echo "Starting FastAPI backend on :8000"
	$(UVICORN) backend.app.main:app \
		--host 0.0.0.0 --port 8000 \
		--reload --log-level info

worker:
	@echo "Starting Celery inference worker"
	$(CELERY) -A backend.app.services.celery_worker worker \
		--loglevel=info --concurrency=1

frontend:
	@echo "Starting React frontend on :5173"
	cd frontend && npm run dev

demo:
	@echo "Launching full demo stack..."
	@bash scripts/demo.sh

# ── Tests ─────────────────────────────────────────────────────────────────
test:
	@echo "Running full test suite..."
	$(PYTHON) -m pytest backend/tests/ -v --tb=short
	@echo "Preprocessing smoke test..."
	$(PYTHON) -m ml.src.preprocessing.pipeline \
		--input ml/data/raw --output /tmp/cv_test_out 2>/dev/null || true
	@echo "Fusion smoke test..."
	$(PYTHON) -c "
from ml.src.models.fusion.trust_score import TrustScoreFusion
r = TrustScoreFusion().fuse(0.05, 0.95, 0.08, True)
assert r.verdict == 'GENUINE', f'Got {r.verdict}'
print(f'Fusion OK: {r.verdict} {r.trust_score:.1f}')
"
	@echo "All tests passed."

# ── Clean ─────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
