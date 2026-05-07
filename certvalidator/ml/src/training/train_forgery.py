"""
ml/src/training/train_forgery.py

Full training pipeline for the EfficientNet-B4 forgery detector.

Features:
  - Automatic Mixed Precision (AMP) — halves VRAM on RTX 3050
  - Discriminative fine-tuning: backbone LR = head LR / 10
  - Cosine annealing LR scheduler with linear warmup
  - Early stopping on val AUC
  - Per-epoch MLflow metric logging
  - Checkpoint saves: best model + last model
  - Label smoothing loss for better calibration
  - Full evaluation: AUC, F1, confusion matrix, per-tamper-type accuracy

Usage:
    python -m ml.src.training.train_forgery \
        --data    ml/data/synthetic \
        --ela-dir ml/data/ela \
        --out     ml/models/checkpoints \
        --epochs  30 \
        --batch   8
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    precision_score, recall_score, classification_report,
)
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn
from rich.table import Table

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ml.src.models.forgery_detector import ForgeryDetector
from ml.src.dataset.certificate_dataset import build_dataloaders

console = Console()
app = typer.Typer()


# ── Loss ──────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy with label smoothing for better calibration."""
    def __init__(self, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth * log_probs).sum(dim=-1).mean()


# ── Warmup scheduler wrapper ───────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self._base_lrs     = [g["lr"] for g in optimizer.param_groups]
        self._cosine       = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr)
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            scale = self.current_epoch / max(1, self.warmup_epochs)
            for g, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                g["lr"] = base_lr * scale
        else:
            self._cosine.step()

    def get_last_lr(self) -> list:
        return [g["lr"] for g in self.optimizer.param_groups]


# ── One epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    model: ForgeryDetector,
    loader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[GradScaler],
    device: torch.device,
    phase: str = "train",
) -> Dict:
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_probs, all_preds, all_tamper = [], [], [], []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            combined = batch["combined"].to(device, non_blocking=True)
            labels   = batch["label"].to(device, non_blocking=True)
            tampers  = batch["tamper_type"]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(scaler is not None)):
                logits = model(combined)
                loss   = criterion(logits, labels)

            if is_train and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            elif is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs = torch.softmax(logits.detach().float(), dim=-1)
            preds = probs.argmax(dim=-1)

            total_loss  += loss.item() * labels.size(0)
            all_labels  .extend(labels.cpu().numpy().tolist())
            all_probs   .extend(probs[:, 1].cpu().numpy().tolist())
            all_preds   .extend(preds.cpu().numpy().tolist())
            all_tamper  .extend(tampers)

    n = len(all_labels)
    avg_loss = total_loss / n

    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    probs_arr  = np.array(all_probs)

    # Metrics
    auc = roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.5
    f1  = f1_score(labels_arr, preds_arr, zero_division=0)
    acc = (labels_arr == preds_arr).mean()
    prec= precision_score(labels_arr, preds_arr, zero_division=0)
    rec = recall_score(labels_arr, preds_arr, zero_division=0)

    # Per-tamper-type accuracy (only on fake samples in val/test)
    tamper_acc = {}
    for tamper, label, pred in zip(all_tamper, all_labels, all_preds):
        if tamper and label == 1:
            tamper_acc.setdefault(tamper, {"correct": 0, "total": 0})
            tamper_acc[tamper]["total"]   += 1
            tamper_acc[tamper]["correct"] += int(pred == 1)

    return {
        "loss": avg_loss, "auc": auc, "f1": f1,
        "acc": acc, "precision": prec, "recall": rec,
        "tamper_accuracy": {
            k: v["correct"] / v["total"]
            for k, v in tamper_acc.items()
        },
        "confusion_matrix": confusion_matrix(labels_arr, preds_arr).tolist(),
    }


# ── Main trainer ──────────────────────────────────────────────────────────

class ForgeryTrainer:
    def __init__(
        self,
        data_root: str,
        out_dir: str,
        ela_dir: Optional[str] = None,
        epochs: int = 30,
        batch_size: int = 8,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        label_smoothing: float = 0.05,
        dropout: float = 0.3,
        amp: bool = True,
        patience: int = 6,
        img_size: tuple = (512, 724),
        num_workers: int = 0,   # Windows: keep at 0 to avoid multiprocessing errors with mixed file types
        experiment_name: str = "certvalidator_forgery",
    ):
        self.data_root   = Path(data_root)
        self.out_dir     = Path(out_dir)
        self.ela_dir     = Path(ela_dir) if ela_dir else None
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.weight_decay= weight_decay
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.dropout     = dropout
        self.amp         = amp and torch.cuda.is_available()
        self.patience    = patience
        self.img_size    = img_size
        self.num_workers = num_workers
        self.experiment  = experiment_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict:
        console.print(f"\n[cyan]Device:[/cyan] {self.device}")
        if torch.cuda.is_available():
            console.print(f"[cyan]GPU:[/cyan] {torch.cuda.get_device_name(0)} "
                          f"| VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # ── Auto-update labels.csv before loading ─────────────────────────
        try:
            import sys, os
            sys.path.insert(0, os.getcwd())
            from scripts.update_labels import update_labels
            update_labels(self.data_root, verbose=True)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not auto-update labels.csv: {e}[/yellow]")

        # ── Dataloaders ───────────────────────────────────────────────────
        console.print("[cyan]Loading datasets...[/cyan]")
        loaders = build_dataloaders(
            root=self.data_root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            img_size=self.img_size,
            ela_dir=self.ela_dir,
        )

        # ── Model ─────────────────────────────────────────────────────────
        model = ForgeryDetector(dropout=self.dropout, pretrained=True).to(self.device)
        params = model.count_parameters()
        console.print(f"[cyan]Model:[/cyan] EfficientNet-B4 | "
                      f"{params['total']:,} params | {params['trainable']:,} trainable")

        # ── Loss + Optimiser ──────────────────────────────────────────────
        criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)

        # Discriminative LR: backbone gets 10x lower LR than head
        param_groups = model.parameter_groups(
            lr_backbone=self.lr / 10,
            lr_head=self.lr,
        )
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)
        scheduler = WarmupCosineScheduler(optimizer, self.warmup_epochs, self.epochs)
        scaler    = GradScaler() if self.amp else None

        # ── MLflow ────────────────────────────────────────────────────────
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.experiment)
            mlflow.start_run()
            mlflow.log_params({
                "epochs": self.epochs, "batch_size": self.batch_size,
                "lr": self.lr, "weight_decay": self.weight_decay,
                "dropout": self.dropout, "amp": self.amp,
                "label_smoothing": self.label_smoothing,
                "img_size": self.img_size,
            })

        # ── Training loop ─────────────────────────────────────────────────
        best_auc     = 0.0
        patience_ctr = 0
        history      = []

        console.print(f"\n[cyan]Training for {self.epochs} epochs...[/cyan]\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_metrics = run_epoch(
                model, loaders["train"], criterion,
                optimizer, scaler, self.device, phase="train"
            )
            val_metrics = run_epoch(
                model, loaders["val"], criterion,
                None, None, self.device, phase="val"
            )
            scheduler.step()

            elapsed = time.time() - t0
            val_auc = val_metrics["auc"]

            # Log to MLflow
            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "train_loss": train_metrics["loss"],
                    "train_auc":  train_metrics["auc"],
                    "train_f1":   train_metrics["f1"],
                    "val_loss":   val_metrics["loss"],
                    "val_auc":    val_metrics["auc"],
                    "val_f1":     val_metrics["f1"],
                    "lr_head":    optimizer.param_groups[1]["lr"],
                }, step=epoch)

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val":   val_metrics,
                "lr":    optimizer.param_groups[1]["lr"],
            }
            history.append(epoch_record)

            # Print row
            is_best = val_auc > best_auc
            marker  = "[green]★[/green]" if is_best else " "
            console.print(
                f"Ep {epoch:3d}/{self.epochs} {marker} | "
                f"train loss={train_metrics['loss']:.4f} auc={train_metrics['auc']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} auc={val_auc:.4f} f1={val_metrics['f1']:.4f} | "
                f"{elapsed:.0f}s"
            )

            # Checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "num_classes": 2,
                "dropout": self.dropout,
                "history": history,
            }
            torch.save(ckpt, self.out_dir / "forgery_last.pt")

            if is_best:
                best_auc     = val_auc
                patience_ctr = 0
                torch.save(ckpt, self.out_dir / "forgery_best.pt")
                console.print(f"  [green]New best val AUC: {best_auc:.4f}[/green]")
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    console.print(f"\n[yellow]Early stopping at epoch {epoch} "
                                  f"(no improvement for {self.patience} epochs)[/yellow]")
                    break

        # ── Final evaluation on test set ──────────────────────────────────
        console.print("\n[cyan]Loading best checkpoint for test evaluation...[/cyan]")
        best_ckpt = torch.load(self.out_dir / "forgery_best.pt", map_location=self.device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        test_metrics = run_epoch(
            model, loaders["test"], criterion,
            None, None, self.device, phase="test"
        )

        self._print_test_report(test_metrics)

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "test_auc": test_metrics["auc"],
                "test_f1":  test_metrics["f1"],
                "test_acc": test_metrics["acc"],
            })
            mlflow.end_run()

        # ── Save metrics JSON ─────────────────────────────────────────────
        metrics_path = self.out_dir / "forgery_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({
                "best_val_auc": best_auc,
                "test": test_metrics,
                "history": [{
                    "epoch": r["epoch"],
                    "train_loss": r["train"]["loss"],
                    "train_auc":  r["train"]["auc"],
                    "val_loss":   r["val"]["loss"],
                    "val_auc":    r["val"]["auc"],
                } for r in history],
            }, f, indent=2)

        console.print(f"\n[green]Training complete.[/green] "
                      f"Best val AUC: {best_auc:.4f} | "
                      f"Test AUC: {test_metrics['auc']:.4f}")
        console.print(f"Checkpoints saved to {self.out_dir}")

        return {"best_val_auc": best_auc, "test": test_metrics}

    def _print_test_report(self, metrics: Dict):
        table = Table(title="Test Set Results", style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("AUC",       f"{metrics['auc']:.4f}")
        table.add_row("F1",        f"{metrics['f1']:.4f}")
        table.add_row("Accuracy",  f"{metrics['acc']:.4f}")
        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall",    f"{metrics['recall']:.4f}")
        console.print(table)

        if metrics.get("tamper_accuracy"):
            t_table = Table(title="Per-Tamper-Type Detection Rate", style="yellow")
            t_table.add_column("Tamper type")
            t_table.add_column("Detection rate")
            for k, v in sorted(metrics["tamper_accuracy"].items()):
                color = "green" if v >= 0.85 else "yellow" if v >= 0.65 else "red"
                t_table.add_row(k, f"[{color}]{v:.1%}[/{color}]")
            console.print(t_table)

        cm = np.array(metrics["confusion_matrix"])
        console.print(f"\n[bold]Confusion matrix[/bold]")
        console.print(f"              Pred genuine  Pred fake")
        console.print(f"True genuine       {cm[0,0]:5d}      {cm[0,1]:5d}")
        console.print(f"True fake          {cm[1,0]:5d}      {cm[1,1]:5d}")


# ── ONNX export ───────────────────────────────────────────────────────────

def export_onnx(
    checkpoint_path: str,
    output_path: str,
    img_size: tuple = (512, 724),
    opset: int = 17,
):
    """Export the best model checkpoint to ONNX for FastAPI inference."""
    import torch.onnx

    device = torch.device("cpu")   # export on CPU for portability
    model  = ForgeryDetector.from_checkpoint(checkpoint_path, device=str(device))
    model.eval()

    dummy = torch.randn(1, 6, img_size[0], img_size[1])

    torch.onnx.export(
        model, dummy, output_path,
        opset_version=opset,
        input_names=["combined"],
        output_names=["logits"],
        dynamic_axes={
            "combined": {0: "batch_size", 2: "height", 3: "width"},
            "logits":   {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    console.print(f"[green]ONNX model exported to {output_path}[/green]")

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        out  = sess.run(None, {"combined": dummy.numpy()})
        console.print(f"[green]ONNX verification passed — output shape: {out[0].shape}[/green]")
    except ImportError:
        console.print("[yellow]onnxruntime not installed — skipping verification[/yellow]")


# ── CLI ───────────────────────────────────────────────────────────────────

@app.command()
def train(
    data:     Path = typer.Option(..., help="Root of synthetic dataset (has labels.csv)"),
    out:      Path = typer.Option(Path("ml/models/checkpoints"), help="Checkpoint output dir"),
    ela_dir:  Optional[Path] = typer.Option(None, help="ELA image directory"),
    epochs:   int   = typer.Option(30,   help="Training epochs"),
    batch:    int   = typer.Option(8,    help="Batch size"),
    lr:       float = typer.Option(3e-4, help="Peak learning rate"),
    workers:  int   = typer.Option(0,    help="DataLoader workers (keep 0 on Windows)"),
    amp:      bool  = typer.Option(True, help="Use AMP (recommended on RTX 3050)"),
    patience: int   = typer.Option(6,    help="Early stopping patience"),
):
    """Train the EfficientNet-B4 forgery detector."""
    trainer = ForgeryTrainer(
        data_root=str(data),
        out_dir=str(out),
        ela_dir=str(ela_dir) if ela_dir else None,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        num_workers=workers,
        amp=amp,
        patience=patience,
    )
    trainer.train()


@app.command()
def export(
    checkpoint: Path = typer.Option(..., help="Path to forgery_best.pt"),
    output:     Path = typer.Option(Path("ml/models/checkpoints/forgery_best.onnx")),
    opset:      int  = typer.Option(17),
):
    """Export trained model to ONNX."""
    export_onnx(str(checkpoint), str(output), opset=opset)


if __name__ == "__main__":
    app()
