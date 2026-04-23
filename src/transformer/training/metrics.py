"""Classification and LM metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score.shape[1] == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
            out["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            out["auc_trapz"] = float(auc(fpr, tpr))
        except ValueError:
            out["roc_auc"] = float("nan")
    return out


def lm_loss_and_perplexity(logits: Tensor, targets: Tensor) -> tuple[Tensor, float]:
    """Cross-entropy over flattened sequence; mean perplexity exp(loss)."""
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
    ppl = math.exp(min(20.0, loss.item()))
    return loss, ppl


def bits_per_token(loss_nats: float) -> float:
    return loss_nats / math.log(2.0)
