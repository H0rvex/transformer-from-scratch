"""Plotting helpers for metrics (optional matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(fpr: list[float], tpr: list[float], out_path: Path, auc_val: float) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_curve(
    epochs: list[int],
    train_loss: list[float],
    val_metric: list[float],
    out_path: Path,
    y_label: str,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label="train loss")
    ax.plot(epochs, val_metric, label=y_label)
    ax.set_xlabel("epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_classifier_plots(metrics: dict[str, Any], assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    cm = metrics.get("confusion_matrix")
    if isinstance(cm, list):
        plot_confusion_matrix(cm, ["neg", "pos"], assets_dir / "confusion_matrix.png")
    roc = metrics.get("roc_curve")
    if isinstance(roc, dict) and "fpr" in roc and "tpr" in roc:
        auc_val = float(metrics.get("roc_auc", 0.0))
        plot_roc_curve(roc["fpr"], roc["tpr"], assets_dir / "roc.png", auc_val)
