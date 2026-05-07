#!/usr/bin/env bash
# scripts/setup.sh
# Full project setup from scratch on Ubuntu/Debian with an RTX 3050.
# Run once after cloning: bash scripts/setup.sh

set -euo pipefail
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $*"; }
error() { echo -e "${RED}[ERR]${NC}  $*"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ── System dependencies ────────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.10 python3.10-venv python3-pip \
    tesseract-ocr tesseract-ocr-eng \
    libsm6 libxext6 libxrender-dev libglib2.0-0 \
    poppler-utils \
    postgresql postgresql-contrib redis-server \
    nodejs npm \
    git curl wget
ok "System packages installed."

# ── Python virtual environment ─────────────────────────────────────────────
info "Creating Python venv..."
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools -q
ok "Venv ready at .venv/"

# ── Backend dependencies ───────────────────────────────────────────────────
info "Installing backend dependencies..."
pip install -r backend/requirements.txt -q
ok "Backend deps installed."

# ── ML dependencies ────────────────────────────────────────────────────────
info "Installing ML dependencies (this may take a few minutes)..."
pip install -r ml/requirements.txt -q
ok "ML deps installed."

# ── Environment file ───────────────────────────────────────────────────────
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    info "Created backend/.env — edit DATABASE_URL and JWT_SECRET_KEY before running."
fi

# ── PostgreSQL setup ───────────────────────────────────────────────────────
info "Setting up PostgreSQL..."
sudo service postgresql start
sudo -u postgres psql -c "CREATE USER certval WITH PASSWORD 'certval';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE certvalidator OWNER certval;" 2>/dev/null || true
ok "Database 'certvalidator' ready."

# ── Alembic migrations ─────────────────────────────────────────────────────
info "Running database migrations..."
cd backend
DATABASE_URL="postgresql://certval:certval@localhost:5432/certvalidator" \
    alembic upgrade head
cd "$ROOT"
ok "Schema migrated."

# ── Redis ──────────────────────────────────────────────────────────────────
info "Starting Redis..."
sudo service redis-server start
ok "Redis running."

# ── Frontend ───────────────────────────────────────────────────────────────
info "Installing frontend dependencies..."
cd frontend
npm install -q
cd "$ROOT"
ok "Frontend deps installed."

# ── ML data directories ────────────────────────────────────────────────────
mkdir -p ml/data/{raw,processed,synthetic,annotated,ela}
mkdir -p ml/models/{checkpoints,llm}
ok "ML directories ready."

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  CertValidator setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate venv:        source .venv/bin/activate"
echo "  2. Generate training data: python scripts/generate_synthetic.py --count 1000"
echo "  3. Preprocess raw data:    python -m ml.src.preprocessing.pipeline --input ml/data/raw --output ml/data/processed"
echo "  4. Start backend:          uvicorn backend.app.main:app --reload --port 8000"
echo "  5. Start frontend:         cd frontend && npm run dev"
echo ""
