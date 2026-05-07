#!/usr/bin/env bash
# scripts/demo.sh
# Hackathon demo launcher — starts all services and opens ngrok tunnel
# Run on the morning of May 7: bash scripts/demo.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

source .venv/bin/activate 2>/dev/null || true

# ── Kill any stale processes ──────────────────────────────────────────────
pkill -f "uvicorn backend" 2>/dev/null || true
pkill -f "celery.*certvalidator" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

# ── Check Redis ────────────────────────────────────────────────────────────
if ! redis-cli ping &>/dev/null; then
    info "Starting Redis..."
    sudo service redis-server start
    sleep 1
fi
ok "Redis running"

# ── Start Celery worker ────────────────────────────────────────────────────
info "Starting Celery worker..."
celery -A backend.app.services.celery_worker worker \
    --loglevel=warning --concurrency=1 \
    --logfile=logs/celery.log &
CELERY_PID=$!
sleep 2
ok "Celery worker started (PID $CELERY_PID)"

# ── Start FastAPI backend ──────────────────────────────────────────────────
info "Starting FastAPI backend on :8000 ..."
mkdir -p logs
uvicorn backend.app.main:app \
    --host 0.0.0.0 --port 8000 \
    --workers 1 \
    --log-level warning \
    --access-log \
    > logs/backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to be ready
for i in $(seq 1 20); do
    if curl -sf http://localhost:8000/health &>/dev/null; then
        ok "Backend ready (PID $BACKEND_PID)"
        break
    fi
    sleep 2
    if [ $i -eq 20 ]; then
        echo "Backend failed to start — check logs/backend.log"
        cat logs/backend.log | tail -20
        exit 1
    fi
done

# ── Start React frontend ───────────────────────────────────────────────────
info "Starting React frontend on :5173 ..."
cd frontend
npm run dev -- --host > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
sleep 3
ok "Frontend started (PID $FRONTEND_PID)"

# ── ngrok tunnel ──────────────────────────────────────────────────────────
if command -v ngrok &>/dev/null; then
    info "Opening ngrok tunnel for backend..."
    ngrok http 8000 --log=stdout > logs/ngrok.log 2>&1 &
    sleep 3
    NGROK_URL=$(curl -sf http://localhost:4040/api/tunnels 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['tunnels'][0]['public_url'])" 2>/dev/null || echo "")
    if [ -n "$NGROK_URL" ]; then
        ok "ngrok tunnel: $NGROK_URL"
        # Update Chrome extension to point at ngrok
        sed -i "s|http://localhost:8000|$NGROK_URL|g" \
            chrome-extension/src/popup.js \
            chrome-extension/src/content.js 2>/dev/null || true
        echo ""
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}  Share this URL with judges: $NGROK_URL${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    fi
else
    warn "ngrok not found — install from https://ngrok.com for public URL"
    warn "Judges can access via local network: http://$(hostname -I | awk '{print $1}'):8000"
fi

# ── Print demo flow ────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  CertValidator — Hackathon Demo Running${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Dashboard:  http://localhost:5173"
echo "  API docs:   http://localhost:8000/docs"
echo "  API health: http://localhost:8000/health"
echo ""
echo "  Demo flow:"
echo "   1. Open http://localhost:5173 in browser"
echo "   2. Upload genuine_demo.png → watch GENUINE verdict"
echo "   3. Upload tampered_demo.png → watch FAKE verdict + heatmap"
echo "   4. Open Chrome extension → select region → instant verdict"
echo "   5. Show judges the PDF report download"
echo ""
echo "  Logs: logs/backend.log | logs/celery.log | logs/frontend.log"
echo ""
echo "  Press Ctrl+C to stop all services"
echo ""

# ── Wait and cleanup ──────────────────────────────────────────────────────
trap "pkill -f 'uvicorn backend'; pkill -f 'celery.*certvalidator'; pkill -f 'vite'; pkill -f 'ngrok'; echo 'All services stopped.'" EXIT
wait
