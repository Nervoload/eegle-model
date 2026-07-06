"""Classification metrics shared by torch and sklearn experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_names: list[str],
    y_score: np.ndarray | None = None,
    positive_label: str | None = None,
) -> dict[str, Any]:
    """Return JSON-serializable classification metrics."""

    labels = list(range(len(label_names)))
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
    }
    if y_score is not None:
        metrics.update(_score_metrics(y_true, y_score, label_names, positive_label=positive_label))
    return metrics


def _score_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: list[str],
    *,
    positive_label: str | None,
) -> dict[str, Any]:
    y_score = np.asarray(y_score)
    if len(label_names) == 2:
        positive_idx = _positive_index(label_names, positive_label)
        positive_score = y_score[:, positive_idx] if y_score.ndim == 2 else y_score
        y_binary = (np.asarray(y_true) == positive_idx).astype("int64")
        return {
            "positive_label": label_names[positive_idx],
            "positive_prevalence": float(y_binary.mean()),
            "roc_auc": _safe_float_metric(roc_auc_score, y_binary, positive_score),
            "average_precision": _safe_float_metric(
                average_precision_score,
                y_binary,
                positive_score,
            ),
        }
    if y_score.ndim == 2 and y_score.shape[1] == len(label_names):
        return {
            "roc_auc_ovr_weighted": _safe_float_metric(
                roc_auc_score,
                y_true,
                y_score,
                multi_class="ovr",
                average="weighted",
            )
        }
    return {}


def _positive_index(label_names: list[str], positive_label: str | None) -> int:
    if positive_label is not None:
        normalized = positive_label.lower()
        for idx, label in enumerate(label_names):
            if label.lower() == normalized:
                return idx
    for idx, label in enumerate(label_names):
        if "target" in label.lower() and "non" not in label.lower():
            return idx
    return len(label_names) - 1


def _safe_float_metric(metric_func: Any, *args: Any, **kwargs: Any) -> float | None:
    try:
        return float(metric_func(*args, **kwargs))
    except ValueError:
        return None
