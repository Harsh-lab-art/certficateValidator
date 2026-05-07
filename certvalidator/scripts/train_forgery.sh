#!/usr/bin/env bash
# scripts/train_forgery.sh
# Complete Phase 2 Part 1 pipeline:
#   1. Generate synthetic data (if not exists)
#   2. Preprocess + ELA
#   3. Train forgery detector
#   4. Export to ONNX
#   5. Evaluate on test set

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }

SYNTHETIC_COUNT=${SYNTHETIC_COUNT:-2000}
BATCH_SIZE=${BATCH_SIZE:-8}
EPOCHS=${EPOCHS:-30}

# ── Step 1: Generate synthetic data ───────────────────────────────────────
if [ ! -f "ml/data/synthetic/labels.csv" ]; then
    info "Generating ${SYNTHETIC_COUNT} synthetic certificates..."
    python scripts/generate_synthetic.py \
        --count   "$SYNTHETIC_COUNT" \
        --output  ml/data/synthetic \
        --fake-ratio 0.5 \
        --seed 42
    ok "Synthetic data generated"
else
    ok "Synthetic data already exists — skipping generation"
fi

# ── Step 2: Preprocess raw data (if any) ──────────────────────────────────
if [ -d "ml/data/raw" ] && [ "$(ls -A ml/data/raw 2>/dev/null)" ]; then
    info "Preprocessing raw certificates..."
    python -m ml.src.preprocessing.pipeline \
        --input  ml/data/raw \
        --output ml/data/processed \
        --ela    ml/data/ela \
        --workers 4
    ok "Preprocessing complete"
fi

# ── Step 3: Preprocess synthetic data for ELA ─────────────────────────────
if [ ! -d "ml/data/ela/genuine" ]; then
    info "Generating ELA maps for synthetic data..."
    python -m ml.src.preprocessing.pipeline \
        --input  ml/data/synthetic \
        --output ml/data/synthetic_processed \
        --ela    ml/data/ela \
        --workers 4
    ok "ELA maps generated"
else
    ok "ELA maps already exist — skipping"
fi

# ── Step 4: Train ─────────────────────────────────────────────────────────
info "Starting forgery detector training..."
info "  Batch size : $BATCH_SIZE"
info "  Epochs     : $EPOCHS"
info "  Device     : $(python -c 'import torch; print(\"cuda\" if torch.cuda.is_available() else \"cpu\")')"

python -m ml.src.training.train_forgery train \
    --data    ml/data/synthetic \
    --ela-dir ml/data/ela \
    --out     ml/models/checkpoints \
    --epochs  "$EPOCHS" \
    --batch   "$BATCH_SIZE" \
    --workers 4 \
    --amp     true \
    --patience 6

ok "Training complete"

# ── Step 5: Export to ONNX ────────────────────────────────────────────────
if [ -f "ml/models/checkpoints/forgery_best.pt" ]; then
    info "Exporting to ONNX..."
    python -m ml.src.training.train_forgery export \
        --checkpoint ml/models/checkpoints/forgery_best.pt \
        --output     ml/models/checkpoints/forgery_best.onnx
    ok "ONNX model saved"
fi

# ── Step 6: Evaluate ──────────────────────────────────────────────────────
info "Running test set evaluation..."
python -m ml.src.training.evaluate \
    --checkpoint ml/models/checkpoints/forgery_best.pt \
    --data       ml/data/synthetic \
    --out        ml/models/logs

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Phase 2 Part 1 complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Artifacts:"
echo "  ml/models/checkpoints/forgery_best.pt    ← PyTorch checkpoint"
echo "  ml/models/checkpoints/forgery_best.onnx  ← ONNX export"
echo "  ml/models/checkpoints/forgery_metrics.json"
echo "  ml/models/logs/eval_plots.png"
echo "  ml/models/logs/eval_results.json"
