#!/usr/bin/env bash
# scripts/train_ocr_layout.sh
# Phase 2 Part 2 pipeline:
#   1. Generate OCR field crops from synthetic certs
#   2. Auto-annotate for LayoutLMv3 NER
#   3. Fine-tune TrOCR on field crops
#   4. Fine-tune LayoutLMv3 on NER annotations

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
info() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}   $*"; }

TROCR_EPOCHS=${TROCR_EPOCHS:-10}
LAYOUT_EPOCHS=${LAYOUT_EPOCHS:-15}
BATCH_SIZE=${BATCH_SIZE:-2}

# ── Step 1: Ensure synthetic data exists ──────────────────────────────────
if [ ! -f "ml/data/synthetic/labels.csv" ]; then
    info "Generating synthetic data first..."
    python scripts/generate_synthetic.py \
        --count 2000 --output ml/data/synthetic --fake-ratio 0.5
fi

# ── Step 2: Generate OCR crops ────────────────────────────────────────────
if [ ! -f "ml/data/ocr_crops/ocr_labels.csv" ]; then
    info "Generating OCR training crops..."
    python -m ml.src.dataset.annotation.auto_annotator crops \
        --csv    ml/data/synthetic/labels.csv \
        --images ml/data/synthetic \
        --out    ml/data/ocr_crops
    ok "OCR crops generated"
fi

# ── Step 3: Auto-annotate for LayoutLMv3 ─────────────────────────────────
if [ ! -d "ml/data/annotation/train" ]; then
    info "Auto-annotating certificates for NER training..."
    python -m ml.src.dataset.annotation.auto_annotator annotate \
        --csv    ml/data/synthetic/labels.csv \
        --images ml/data/synthetic \
        --out    ml/data/annotation
    ok "NER annotations generated"
fi

# ── Step 4: Fine-tune TrOCR ───────────────────────────────────────────────
info "Fine-tuning TrOCR on certificate field crops..."
python - << PYEOF
from ml.src.models.ocr.trocr_model import finetune_trocr
finetune_trocr(
    data_dir="ml/data/ocr_crops",
    output_dir="ml/models/checkpoints/trocr_finetuned",
    epochs=${TROCR_EPOCHS},
    batch_size=${BATCH_SIZE},
)
PYEOF
ok "TrOCR fine-tuning complete"

# ── Step 5: Fine-tune LayoutLMv3 ─────────────────────────────────────────
info "Fine-tuning LayoutLMv3 for certificate NER..."
python - << PYEOF
from ml.src.models.layout.layoutlm_extractor import finetune_layoutlmv3
finetune_layoutlmv3(
    annotation_dir="ml/data/annotation",
    output_dir="ml/models/checkpoints/layoutlmv3_finetuned",
    epochs=${LAYOUT_EPOCHS},
    batch_size=${BATCH_SIZE},
)
PYEOF
ok "LayoutLMv3 fine-tuning complete"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Phase 2 Part 2 complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Artifacts:"
echo "  ml/models/checkpoints/trocr_finetuned/       ← fine-tuned TrOCR"
echo "  ml/models/checkpoints/layoutlmv3_finetuned/  ← fine-tuned LayoutLMv3"
echo "  ml/data/annotation/                          ← NER training data"
echo "  ml/data/ocr_crops/                           ← OCR training crops"
