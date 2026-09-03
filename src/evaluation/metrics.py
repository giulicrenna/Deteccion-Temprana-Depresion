"""Métricas de evaluación: accuracy, F1 macro, AUC, kappa, classification report."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def compute(y_true: list[int], y_pred: list[int], y_proba: list[list[float]] | None = None) -> dict[str, Any]:
    """Devuelve un dict con las métricas estándar del proyecto."""
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    if y_proba is not None:
        try:
            arr = np.asarray(y_proba)
            if arr.ndim == 2 and arr.shape[1] > 1:
                out["auc_ovr_macro"] = float(
                    roc_auc_score(y_true, arr, multi_class="ovr", average="macro")
                )
        except Exception:
            pass
    return out


__all__ = ["compute"]
