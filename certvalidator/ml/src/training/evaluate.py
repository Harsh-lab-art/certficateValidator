"""
ml/src/training/evaluate.py

Standalone evaluation script — run after training to get the full
metrics report on the test set, including per-tamper-type breakdown
and a saved confusion matrix plot.

Usage:
    python -m ml.src.training.evaluate \
        --checkpoint ml/models/checkpoints/forgery_best.pt \
        --data       ml/data/synthetic \
        --out        ml/models/logs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    classification_report,
)
import typer
from rich.console import Console
from rich.table import Table

from ml.src.models.forgery_detector import ForgeryDetector
from ml.src.dataset.certificate_dataset import build_dataloaders

console = Console()
app = typer.Typer()


# ── Full evaluation ────────────────────────────────────────────────────────

def evaluate_model(
    model: ForgeryDetector,
    loader,
    device: torch.device,
    out_dir: Optional[Path] = None,
) -> Dict:
    model.eval()
    all_labels, all_probs, all_tampers = [], [], []

    with torch.no_grad():
        for batch in loader:
            combined = batch["combined"].to(device)
            labels   = batch["label"].to(device)
            probs_b  = torch.softmax(model(combined).float(), dim=-1)

            all_labels .extend(labels.cpu().numpy().tolist())
            all_probs  .extend(probs_b[:, 1].cpu().numpy().tolist())
            all_tampers.extend(batch["tamper_type"])

    labels_arr = np.array(all_labels)
    probs_arr  = np.array(all_probs)
    preds_arr  = (probs_arr >= 0.5).astype(int)

    # Core metrics
    auc  = roc_auc_score(labels_arr, probs_arr)
    ap   = average_precision_score(labels_arr, probs_arr)
    report_str = classification_report(labels_arr, preds_arr,
                                       target_names=["genuine", "fake"])
    cm = confusion_matrix(labels_arr, preds_arr)

    # Per-tamper accuracy
    tamper_stats: Dict[str, Dict] = {}
    for t, label, pred in zip(all_tampers, all_labels, preds_arr):
        if not t or label == 0:
            continue
        tamper_stats.setdefault(t, {"correct": 0, "total": 0})
        tamper_stats[t]["total"]   += 1
        tamper_stats[t]["correct"] += int(pred == 1)

    tamper_acc = {
        k: {"rate": v["correct"] / v["total"], **v}
        for k, v in tamper_stats.items()
    }

    # Optimal threshold (maximise F1)
    precisions, recalls, thresholds = precision_recall_curve(labels_arr, probs_arr)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_thresh = float(thresholds[np.argmax(f1s[:-1])])
    preds_opt   = (probs_arr >= best_thresh).astype(int)

    results = {
        "auc": float(auc),
        "average_precision": float(ap),
        "optimal_threshold": best_thresh,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_str,
        "tamper_accuracy": tamper_acc,
        "n_samples": len(labels_arr),
        "n_genuine": int((labels_arr == 0).sum()),
        "n_fake":    int((labels_arr == 1).sum()),
    }

    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_plots(labels_arr, probs_arr, cm, out_dir)
        json_path = out_dir / "eval_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[green]Results saved to {json_path}[/green]")

    return results


def _save_plots(
    labels: np.ndarray,
    probs:  np.ndarray,
    cm:     np.ndarray,
    out_dir: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0f172a")

    # ── ROC curve ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    ax = axes[0]
    ax.set_facecolor("#1e293b")
    ax.plot(fpr, tpr, color="#6366f1", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR", color="#94a3b8")
    ax.set_ylabel("TPR", color="#94a3b8")
    ax.set_title("ROC curve", color="#e2e8f0")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
    ax.tick_params(colors="#64748b")

    # ── Precision-Recall ──────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    ax = axes[1]
    ax.set_facecolor("#1e293b")
    ax.plot(rec, prec, color="#4ade80", lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall", color="#94a3b8")
    ax.set_ylabel("Precision", color="#94a3b8")
    ax.set_title("Precision-recall", color="#e2e8f0")
    ax.legend(facecolor="#1e293b", labelcolor="#e2e8f0")
    ax.tick_params(colors="#64748b")

    # ── Confusion matrix ──────────────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#1e293b")
    disp = ConfusionMatrixDisplay(cm, display_labels=["Genuine", "Fake"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion matrix", color="#e2e8f0")
    ax.tick_params(colors="#64748b")

    plt.tight_layout()
    fig.savefig(out_dir / "eval_plots.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    console.print(f"[green]Plots saved to {out_dir / 'eval_plots.png'}[/green]")


def print_results(results: Dict):
    table = Table(title="Evaluation Results", style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("AUC",                f"{results['auc']:.4f}")
    table.add_row("Average precision",  f"{results['average_precision']:.4f}")
    table.add_row("Optimal threshold",  f"{results['optimal_threshold']:.3f}")
    table.add_row("Samples",            str(results['n_samples']))
    table.add_row("Genuine",            str(results['n_genuine']))
    table.add_row("Fake",               str(results['n_fake']))
    console.print(table)

    console.print("\n[bold]Classification report[/bold]")
    console.print(results["classification_report"])

    if results.get("tamper_accuracy"):
        t_table = Table(title="Per-Tamper Detection Rate", style="yellow")
        t_table.add_column("Tamper type")
        t_table.add_column("Rate")
        t_table.add_column("Correct / Total")
        for k, v in sorted(results["tamper_accuracy"].items()):
            rate  = v["rate"]
            color = "green" if rate >= 0.85 else "yellow" if rate >= 0.65 else "red"
            t_table.add_row(k, f"[{color}]{rate:.1%}[/{color}]",
                            f"{v['correct']}/{v['total']}")
        console.print(t_table)


# ── CLI ───────────────────────────────────────────────────────────────────

@app.command()
def main(
    checkpoint: Path = typer.Option(...,  help="Path to forgery_best.pt"),
    data:       Path = typer.Option(...,  help="Dataset root with labels.csv"),
    out:        Path = typer.Option(Path("ml/models/logs"), help="Output directory for plots/JSON"),
    batch:      int  = typer.Option(8),
    workers:    int  = typer.Option(4),
):
    """Evaluate the forgery detector on the test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Loading checkpoint: {checkpoint}[/cyan]")

    model = ForgeryDetector.from_checkpoint(str(checkpoint), device=str(device))
    model.to(device)

    loaders = build_dataloaders(str(data), batch_size=batch, num_workers=workers)
    results = evaluate_model(model, loaders["test"], device, out_dir=out)
    print_results(results)


if __name__ == "__main__":
    app()
