"""
train.py  —  CertValidator Training Pipeline
=============================================

USAGE
-----
1. Put your certificates in the training_data/ folder:

   training_data/
       genuine/        <- real certificates (label = 0)
       fake/           <- tampered/fake certificates (label = 1)

2. Run:  python train.py
   Or:   python train.py --epochs 20 --batch 4
         python train.py --status
         python train.py --test cert.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "training_data"
GENUINE_DIR = DATA_DIR / "genuine"
FAKE_DIR    = DATA_DIR / "fake"
PROCESSED   = DATA_DIR / "processed"
ELA_DIR     = DATA_DIR / "ela"
LABELS_CSV  = DATA_DIR / "labels.csv"
CKPT_DIR    = ROOT / "ml" / "models" / "checkpoints"
METRICS_JSON= CKPT_DIR / "forgery_metrics.json"
SUPPORTED   = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".pdf"}


def _print(msg, color=""):
    colors = {"green":"\033[92m","red":"\033[91m","yellow":"\033[93m",
              "cyan":"\033[96m","bold":"\033[1m","":""}
    reset = "\033[0m" if color else ""
    print(f"{colors.get(color,'')}{msg}{reset}")


def _count(folder):
    if not folder.exists():
        return 0
    return sum(1 for f in folder.iterdir()
               if f.suffix.lower() in SUPPORTED and f.is_file())


def _setup():
    for d in [GENUINE_DIR, FAKE_DIR, PROCESSED, ELA_DIR, CKPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def validate_data():
    n_genuine = _count(GENUINE_DIR)
    n_fake    = _count(FAKE_DIR)
    _print("\n=== Training Data Summary ===", "bold")
    _print(f"  Genuine : {n_genuine}  ({GENUINE_DIR})",
           "green" if n_genuine >= 10 else "red")
    _print(f"  Fake    : {n_fake}  ({FAKE_DIR})",
           "green" if n_fake >= 10 else "red")
    if n_genuine == 0 and n_fake == 0:
        _print("\n[ERROR] No training data found!", "red")
        _print(f"  Add images to:", "yellow")
        _print(f"    {GENUINE_DIR}  <- real certificates", "yellow")
        _print(f"    {FAKE_DIR}  <- fake certificates", "yellow")
        sys.exit(1)
    if n_genuine < 2 or n_fake < 2:
        _print("\n[ERROR] Need at least 2 genuine AND 2 fake certificates.", "red")
        sys.exit(1)
    return n_genuine, n_fake


def preprocess_all(force=False):
    sys.path.insert(0, str(ROOT))
    from ml.src.preprocessing.pipeline import CertificatePreprocessor
    preprocessor = CertificatePreprocessor()
    samples = []

    for label_int, label_name, folder in [
        (0, "genuine", GENUINE_DIR),
        (1, "fake",    FAKE_DIR),
    ]:
        files = [f for f in sorted(folder.iterdir())
                 if f.suffix.lower() in SUPPORTED and f.is_file()]
        _print(f"\nPreprocessing {len(files)} {label_name} certificates...", "cyan")

        for img_path in files:
            out_name = f"{label_name}_{img_path.stem}.png"
            out_path = PROCESSED / out_name
            ela_path = ELA_DIR    / out_name

            if out_path.exists() and ela_path.exists() and not force:
                samples.append({
                    "filename": f"processed/{out_name}",
                    "label": label_int, "tamper_type": "" if label_int == 0 else "manual",
                    "student_name":"","institution":"","degree":"",
                    "issue_date":"","grade":"","cgpa":"0.0",
                })
                continue

            try:
                result = preprocessor.process(img_path)
                if not result.success:
                    _print(f"  [SKIP] {img_path.name}: {result.error}", "yellow")
                    continue
                cv2.imwrite(str(out_path), result.processed_image)
                cv2.imwrite(str(ela_path), result.ela_image)
                samples.append({
                    "filename": f"processed/{out_name}",
                    "label": label_int, "tamper_type": "" if label_int == 0 else "manual",
                    "student_name":"","institution":"","degree":"",
                    "issue_date":"","grade":"","cgpa":"0.0",
                })
                print(f"  OK {img_path.name}")
            except Exception as e:
                _print(f"  [ERROR] {img_path.name}: {e}", "red")

    return samples


def write_labels(samples):
    fieldnames = ["filename","label","tamper_type","student_name",
                  "institution","degree","issue_date","grade","cgpa"]
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    n_g = sum(1 for s in samples if s["label"] == 0)
    n_f = sum(1 for s in samples if s["label"] == 1)
    _print(f"\nLabels CSV: {len(samples)} samples ({n_g} genuine, {n_f} fake)", "green")


def train_model(epochs=30, batch_size=4, lr=3e-4, patience=8, workers=0):
    sys.path.insert(0, str(ROOT))
    import torch
    from ml.src.training.train_forgery import ForgeryTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"\n=== Starting Training ===", "bold")
    _print(f"  Device  : {device}", "cyan")
    if torch.cuda.is_available():
        _print(f"  GPU     : {torch.cuda.get_device_name(0)}", "cyan")
    _print(f"  Epochs  : {epochs} | Batch: {batch_size} | LR: {lr}", "cyan")

    trainer = ForgeryTrainer(
        data_root=str(DATA_DIR),
        out_dir=str(CKPT_DIR),
        ela_dir=str(ELA_DIR),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        num_workers=workers,
        amp=torch.cuda.is_available(),
    )
    results = trainer.train()

    _print(f"\n=== Training Complete ===", "bold")
    _print(f"  Best val AUC : {results['best_val_auc']:.4f}", "green")
    _print(f"  Test AUC     : {results['test']['auc']:.4f}", "green")
    _print(f"  Test Accuracy: {results['test']['acc']:.4f}", "green")
    _print(f"\n  Model: {CKPT_DIR / 'forgery_best.pt'}", "green")
    _print(f"  Restart the API server to use the trained model.", "yellow")
    return results


def test_certificate(image_path):
    sys.path.insert(0, str(ROOT))
    path = Path(image_path)
    if not path.exists():
        _print(f"[ERROR] File not found: {image_path}", "red")
        sys.exit(1)

    _print(f"\n=== Testing Certificate: {path.name} ===", "bold")

    img = cv2.imread(str(path))
    if img is None:
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            _print(f"[ERROR] Cannot load: {e}", "red")
            sys.exit(1)

    ckpt_path = CKPT_DIR / "forgery_best.pt"

    if ckpt_path.exists():
        import torch
        from ml.src.models.forgery_detector import ForgeryDetector
        from ml.src.preprocessing.pipeline import CertificatePreprocessor
        from torchvision import transforms

        _print("  Using trained CNN model...", "green")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = ForgeryDetector.from_checkpoint(str(ckpt_path), device=str(device))
        model.eval()

        preprocessor = CertificatePreprocessor()
        result = preprocessor.process(path)
        img_bgr = result.processed_image
        ela_bgr = result.ela_image

        h, w = 512, 724
        img_r = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
        ela_r = cv2.resize(ela_bgr, (w, h), interpolation=cv2.INTER_AREA)
        to_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        combined = torch.cat([
            to_t(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)),
            to_t(cv2.cvtColor(ela_r, cv2.COLOR_BGR2RGB)),
        ], dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            probs, _ = model.predict(combined)
            forgery_score = float(probs[0, 1].item())
            genuine_prob  = float(probs[0, 0].item())

        _print(f"\n  CNN Result:", "bold")
        _print(f"    Genuine : {genuine_prob:.1%}")
        _print(f"    Fake    : {forgery_score:.1%}")
    else:
        _print("  No trained model — using ELA analysis...", "yellow")
        from ml.src.utils.ela_scorer import score_ela
        ela_result = score_ela(img)
        forgery_score = ela_result.forgery_score
        _print(f"\n  ELA forgery score: {forgery_score:.4f}")

    from ml.src.models.fusion.trust_score import TrustScoreFusion
    fusion = TrustScoreFusion()
    fr = fusion.fuse(
        forgery_score=forgery_score,
        field_confidence=0.7,
        nlp_anomaly_score=forgery_score * 0.8,
        institution_matched=True,
    )

    color = {"GENUINE":"green","SUSPICIOUS":"yellow","FAKE":"red"}.get(fr.verdict,"")
    _print(f"\n{'='*50}", "bold")
    _print(f"  VERDICT     : {fr.verdict}", color)
    _print(f"  Trust Score : {fr.trust_score:.1f} / 100", color)
    _print(f"  Explanation : {fr.explanation}")
    _print(f"{'='*50}\n", "bold")


def show_status():
    _print("\n=== CertValidator Status ===", "bold")
    _print(f"\nTraining Data:", "bold")
    _print(f"  Genuine : {_count(GENUINE_DIR)} certificates  ({GENUINE_DIR})")
    _print(f"  Fake    : {_count(FAKE_DIR)} certificates  ({FAKE_DIR})")

    ckpt = CKPT_DIR / "forgery_best.pt"
    _print(f"\nModel:", "bold")
    if ckpt.exists():
        try:
            import torch
            d = torch.load(str(ckpt), map_location="cpu")
            _print(f"  Status  : TRAINED  (epoch {d.get('epoch','?')}, "
                   f"val AUC={d.get('val_auc',0):.4f})", "green")
        except Exception:
            _print(f"  Status  : TRAINED (checkpoint exists)", "green")
        _print(f"  Path    : {ckpt}")
    else:
        _print(f"  Status  : NOT TRAINED", "red")
        _print(f"  Run 'python train.py' to train", "yellow")

    if METRICS_JSON.exists():
        with open(METRICS_JSON) as f:
            m = json.load(f)
        test = m.get("test", {})
        _print(f"\nLast Results:", "bold")
        _print(f"  Test AUC={test.get('auc',0):.4f}  "
               f"F1={test.get('f1',0):.4f}  "
               f"Acc={test.get('acc',0):.4f}")

    _print(f"\nAPI Server:", "bold")
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        _print(f"  Status : RUNNING  (http://localhost:8000)", "green")
    except Exception:
        _print(f"  Status : NOT RUNNING", "red")


def main():
    global DATA_DIR, GENUINE_DIR, FAKE_DIR, PROCESSED, ELA_DIR, LABELS_CSV, CKPT_DIR

    parser = argparse.ArgumentParser(description="CertValidator Training Pipeline")
    parser.add_argument("--data",    default=str(DATA_DIR))
    parser.add_argument("--out",     default=str(CKPT_DIR))
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--patience",type=int,   default=8)
    parser.add_argument("--workers", type=int,   default=0)
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--status",  action="store_true")
    parser.add_argument("--test",    type=str,   default=None)
    parser.add_argument("--preprocess-only", action="store_true")
    args = parser.parse_args()

    if args.data != str(DATA_DIR):
        DATA_DIR    = Path(args.data)
        GENUINE_DIR = DATA_DIR / "genuine"
        FAKE_DIR    = DATA_DIR / "fake"
        PROCESSED   = DATA_DIR / "processed"
        ELA_DIR     = DATA_DIR / "ela"
        LABELS_CSV  = DATA_DIR / "labels.csv"
    if args.out != str(CKPT_DIR):
        CKPT_DIR = Path(args.out)

    if args.status:
        show_status()
        return

    if args.test:
        test_certificate(args.test)
        return

    _print("\n" + "="*55, "bold")
    _print("  CertValidator — Training Pipeline", "bold")
    _print("="*55 + "\n", "bold")

    _setup()
    validate_data()

    _print(f"\nStep 1/3: Preprocessing...", "cyan")
    samples = preprocess_all(force=args.force)
    if not samples:
        _print("[ERROR] No images preprocessed.", "red")
        sys.exit(1)

    write_labels(samples)

    if args.preprocess_only:
        _print("\nDone. Run without --preprocess-only to train.", "green")
        return

    _print(f"\nStep 2/3: Training...", "cyan")
    train_model(
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, patience=args.patience, workers=args.workers,
    )
    _print(f"\nStep 3/3: Done!", "green")
    _print(f"\nRestart the API server to load the trained model.\n")


if __name__ == "__main__":
    main()
