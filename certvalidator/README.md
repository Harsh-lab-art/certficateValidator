# CertValidator — AI-Based Academic Certificate Authenticity Validator

> Hackathon 2026 · May 7-8  
> **Problem Statement #5** — AI-Based Academic Certificate Authenticity Validator

---

## Quick Start (Hackathon Day)

```bash
# 1. Clone and enter
git clone <repo> && cd certvalidator

# 2. Full setup (creates venv, installs all deps, sets up DB)
bash scripts/setup.sh

# 3. Generate training data + train all models
make train-all          # takes ~2-4 hrs on RTX 3050

# 4. Launch everything (backend + celery + frontend + ngrok)
make demo
```

That's it. Dashboard at `http://localhost:5173`.

---

## Project Structure

```
certvalidator/
├── backend/                    FastAPI REST API
│   ├── app/
│   │   ├── main.py             Entry point — mounts all routers
│   │   ├── api/v1/routes.py    All endpoints (auth/verify/institutions/stats)
│   │   ├── core/
│   │   │   ├── config.py       Pydantic settings from .env
│   │   │   ├── auth.py         JWT + bcrypt + API key auth
│   │   │   └── database.py     Async SQLAlchemy + asyncpg
│   │   ├── models/schema.py    ORM: certificates, verifications, institutions
│   │   └── services/
│   │       ├── inference.py    CertificatePipeline — chains all 7 steps
│   │       ├── celery_worker.py Async task queue
│   │       ├── institution_db.py 50 seeded Indian universities + fuzzy lookup
│   │       └── report_generator.py ReportLab forensic PDF
│   ├── migrations/             Alembic DB migrations
│   ├── tests/                  pytest test suite
│   └── requirements.txt
│
├── ml/                         All deep learning code
│   ├── config.yaml             Single source of truth for hyperparameters
│   ├── src/
│   │   ├── preprocessing/
│   │   │   └── pipeline.py     Deskew · denoise · ELA generation
│   │   ├── augmentation/
│   │   │   └── augmentor.py    Certificate-specific augmentation
│   │   ├── dataset/
│   │   │   ├── certificate_dataset.py  PyTorch dataset (6-channel RGB+ELA)
│   │   │   └── annotation/
│   │   │       └── auto_annotator.py   Auto NER annotation from synthetic data
│   │   ├── models/
│   │   │   ├── forgery_detector.py     EfficientNet-B4 (6-channel, ELA input)
│   │   │   ├── gradcam.py              GradCAM heatmap engine
│   │   │   ├── ocr/trocr_model.py     TrOCR fine-tuning
│   │   │   ├── layout/
│   │   │   │   ├── layoutlm_extractor.py  LayoutLMv3 NER field extractor
│   │   │   │   └── field_scorer.py    Validation + cross-field consistency
│   │   │   ├── llm/
│   │   │   │   └── mistral_reasoner.py Mistral-7B GGUF reasoning engine
│   │   │   └── fusion/
│   │   │       └── trust_score.py     Weighted ensemble → 0-100 score
│   │   ├── training/
│   │   │   ├── train_forgery.py       Full training loop (AMP, early stop)
│   │   │   └── evaluate.py            Evaluation + ROC/PR curves
│   │   └── utils/
│   │       ├── config.py              YAML config loader
│   │       └── ela_analysis.py        ELA utilities + visualisation
│   └── requirements.txt
│
├── frontend/                   React + Vite dashboard
│   └── src/
│       ├── App.jsx             Router
│       ├── components/
│       │   ├── TrustScoreRing.jsx   Animated SVG score gauge
│       │   ├── VerdictBadge.jsx     GENUINE/FAKE/SUSPICIOUS pill
│       │   ├── DropZone.jsx         Drag-and-drop upload
│       │   ├── ProcessingSteps.jsx  Animated 8-step pipeline indicator
│       │   ├── FieldBreakdown.jsx   Per-field confidence bars
│       │   ├── HeatmapViewer.jsx    GradCAM overlay with opacity slider
│       │   ├── ContributionChart.jsx Recharts radial breakdown
│       │   └── Layout.jsx           Sidebar + live stats
│       ├── hooks/
│       │   ├── useVerification.js   Upload → poll → result state machine
│       │   └── useStats.js          Live dashboard counter poller
│       ├── pages/
│       │   ├── HomePage.jsx         Dashboard with stats + feature cards
│       │   ├── VerifyPage.jsx       Upload flow
│       │   ├── ResultPage.jsx       Full forensic result
│       │   └── HistoryPage.jsx      Past verifications list
│       └── utils/api.js            Axios client with auth interceptors
│
├── chrome-extension/           Manifest V3 extension
│   ├── manifest.json
│   ├── popup.html              Score ring + bars + field table
│   └── src/
│       ├── popup.js            Animated score ring, sub-score bars
│       ├── content.js          Region selector overlay + inline badge
│       └── background.js       captureVisibleTab + message routing
│
├── demo/
│   ├── certificates/           3 ready-to-use demo certs
│   │   ├── genuine_demo.png    → should score 85-95, GENUINE
│   │   ├── name_tampered.png   → should score 15-35, FAKE
│   │   └── grade_tampered.png  → should score 20-40, FAKE
│   └── scripts/
│       └── generate_demo_certs.py
│
├── presentation/
│   ├── CertValidator_Hackathon.pptx   10-slide deck (ready to present)
│   └── build_deck.js
│
├── scripts/
│   ├── setup.sh               One-shot project setup
│   ├── generate_synthetic.py  Synthetic certificate generator
│   ├── train_forgery.sh       Phase 2 Part 1 pipeline
│   ├── train_ocr_layout.sh    Phase 2 Part 2 pipeline
│   └── demo.sh                Full hackathon demo launcher
│
├── Makefile                   Single entry point for all commands
├── ml/config.yaml             All hyperparameters
├── backend/.env.example        Environment template
└── README.md                  This file
```

---

## Architecture

```
Input (upload / Chrome extension screen-select / URL)
        ↓
Preprocessing  → deskew · CLAHE denoise · gamma · ELA
        ↓
┌─────────────────────────────────────────────────────────┐
│                    Deep Learning Core                     │
│  EfficientNet-B4+ELA  │  LayoutLMv3    │  Mistral-7B    │
│  (forgery score)      │  (fields)      │  (reasoning)   │
└─────────────────────────────────────────────────────────┘
        ↓
Trust Score Fusion
  score = (1-forgery)×0.45 + field_conf×0.35 + (1-nlp_anomaly)×0.20 + inst_bonus
        ↓
FastAPI → PostgreSQL + Redis (Celery)
        ↓
React dashboard · GradCAM heatmap · PDF report · Chrome popup
```

---

## Running Each Component

### Backend
```bash
source .venv/bin/activate
uvicorn backend.app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Celery Worker (for async inference)
```bash
celery -A backend.app.services.celery_worker worker --loglevel=info --concurrency=1
```

### Frontend
```bash
cd frontend && npm run dev
# Dashboard: http://localhost:5173
```

### Chrome Extension
1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select `chrome-extension/` folder
4. Click the shield icon → drag region → instant verdict

---

## Training

```bash
# Generate 2000 synthetic certificates (genuine + fake)
make data

# Train Phase 2 Part 1: EfficientNet-B4 forgery detector
make train-p1

# Train Phase 2 Part 2: TrOCR + LayoutLMv3
make train-p2

# Download Mistral-7B GGUF (4.1 GB)
make train-p3

# Or run everything at once
make train-all
```

### Hardware requirements
- RTX 3050 (4 GB VRAM) or better — all models fit with AMP + Q4 quantisation
- 16 GB RAM recommended
- 20 GB free disk space for models + data

---

## Demo Flow (for judges)

1. **Open** `http://localhost:5173`
2. **Upload** `demo/certificates/genuine_demo.png` → watch **GENUINE** with score ~87
3. **Upload** `demo/certificates/name_tampered.png` → watch **FAKE** with GradCAM highlighting the name region
4. **Upload** `demo/certificates/grade_tampered.png` → watch **FAKE** with field validator flagging impossible CGPA
5. **Chrome Extension** → select any certificate on screen → inline verdict badge appears
6. **Export PDF** → download the full forensic report

---

## Environment Variables

Copy `backend/.env.example` to `backend/.env` and fill in:

```env
DATABASE_URL=postgresql+asyncpg://certval:certval@localhost:5432/certvalidator
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your-32-char-secret-here
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Forgery detection | EfficientNet-B4 + ELA (6-channel, PyTorch) |
| OCR | TrOCR-base fine-tuned (HuggingFace) |
| Field extraction | LayoutLMv3-base + NER head |
| NLP reasoning | Mistral-7B-Instruct Q4_K_M (llama.cpp) |
| Backend | FastAPI + Celery + Redis + PostgreSQL |
| Frontend | React 18 + Vite + Tailwind + Framer Motion |
| Extension | Chrome Manifest V3 |
| Reports | ReportLab PDF |
| Training | PyTorch 2.2 + HuggingFace Transformers + MLflow |
