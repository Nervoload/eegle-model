"""End-to-end experiment runner."""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from eegworkbench.config import ExperimentConfig, save_config, to_plain_data
from eegworkbench.data import load_eeg_data, standardize_trials
from eegworkbench.evaluation import classification_metrics
from eegworkbench.models import build_model, model_framework
from eegworkbench.models.sklearn_models import build_sklearn_model
from eegworkbench.training.torch_loop import fit_torch_classifier, predict_torch_probabilities


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Run a train/validation/test experiment and write artifacts to disk."""

    run_dir = _make_run_dir(config.output_dir, config.experiment_name)
    save_config(config, run_dir / "config.resolved.yml")

    bundle = load_eeg_data(config.dataset)
    X = standardize_trials(bundle.X, config.dataset.standardize)
    y = bundle.y
    splits = _split_data(
        X,
        y,
        valid_size=config.training.valid_size,
        test_size=config.training.test_size,
        seed=config.training.seed,
    )

    framework = config.model.framework or model_framework(config.model.name)
    if framework == "sklearn" or model_framework(config.model.name) == "sklearn":
        result = _run_sklearn_experiment(config, bundle, splits, run_dir)
    elif framework == "torch":
        result = _run_torch_experiment(config, bundle, splits, run_dir)
    else:
        raise ValueError(f"Unsupported model framework: {framework!r}")

    summary = {
        "run_dir": str(run_dir),
        "dataset": {
            "source": bundle.source,
            "n_trials": bundle.n_trials,
            "n_chans": bundle.n_chans,
            "n_times": bundle.n_times,
            "labels": bundle.label_names,
        },
        "model": to_plain_data(config.model),
        **result,
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


def _run_torch_experiment(
    config: ExperimentConfig,
    bundle: Any,
    splits: dict[str, DataSplit],
    run_dir: Path,
) -> dict[str, Any]:
    model = build_model(
        config.model,
        n_chans=bundle.n_chans,
        n_outputs=bundle.n_outputs,
        n_times=bundle.n_times,
    )
    history = fit_torch_classifier(
        model,
        (splits["train"].X, splits["train"].y),
        (splits["valid"].X, splits["valid"].y),
        training=config.training,
        run_dir=run_dir,
    )
    y_score = predict_torch_probabilities(model, splits["test"].X, training=config.training)
    y_pred = y_score.argmax(axis=1)
    metrics = classification_metrics(
        splits["test"].y,
        y_pred,
        label_names=bundle.label_names,
        y_score=y_score,
    )
    _write_json(run_dir / "metrics.json", metrics)
    _write_predictions_csv(run_dir / "predictions.csv", splits["test"], y_pred, y_score, bundle)
    return {
        "framework": "torch",
        "history": history,
        "test_metrics": metrics,
        "model_path": str(run_dir / "model.pt"),
    }


def _run_sklearn_experiment(
    config: ExperimentConfig,
    bundle: Any,
    splits: dict[str, DataSplit],
    run_dir: Path,
) -> dict[str, Any]:
    try:
        import joblib
    except ImportError:
        joblib = None

    estimator = build_sklearn_model(config.model.name)
    X_train = np.concatenate([splits["train"].X, splits["valid"].X], axis=0)
    y_train = np.concatenate([splits["train"].y, splits["valid"].y], axis=0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(splits["test"].X)
    y_score = _predict_sklearn_scores(estimator, splits["test"].X)
    metrics = classification_metrics(
        splits["test"].y,
        y_pred,
        label_names=bundle.label_names,
        y_score=y_score,
    )
    _write_json(run_dir / "metrics.json", metrics)
    _write_predictions_csv(run_dir / "predictions.csv", splits["test"], y_pred, y_score, bundle)
    model_path = run_dir / "model.joblib"
    if joblib is not None:
        joblib.dump(estimator, model_path)
    return {
        "framework": "sklearn",
        "test_metrics": metrics,
        "model_path": str(model_path) if joblib is not None else None,
    }


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    *,
    valid_size: float,
    test_size: float,
    seed: int,
) -> dict[str, DataSplit]:
    train_valid_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=test_size,
        random_state=seed,
        stratify=_safe_stratify(y),
    )
    relative_valid = valid_size / max(1e-9, 1.0 - test_size)
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=relative_valid,
        random_state=seed,
        stratify=_safe_stratify(y[train_valid_idx]),
    )
    return {
        "train": DataSplit(X[train_idx], y[train_idx], train_idx),
        "valid": DataSplit(X[valid_idx], y[valid_idx], valid_idx),
        "test": DataSplit(X[test_idx], y[test_idx], test_idx),
    }


def _predict_sklearn_scores(estimator: Any, X: np.ndarray) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X))
    return None


def _write_predictions_csv(
    path: Path,
    split: DataSplit,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    bundle: Any,
) -> None:
    metadata_rows = _metadata_rows(bundle.metadata, split.indices)
    score_columns = _score_columns(y_score, bundle.label_names)
    metadata_keys = sorted({key for row in metadata_rows for key in row})
    fieldnames = [
        "row_index",
        "y_true",
        "y_true_label",
        "y_pred",
        "y_pred_label",
        *score_columns,
        *[f"metadata_{key}" for key in metadata_keys],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_pos, original_idx in enumerate(split.indices):
            row = {
                "row_index": int(original_idx),
                "y_true": int(split.y[row_pos]),
                "y_true_label": bundle.label_names[int(split.y[row_pos])],
                "y_pred": int(y_pred[row_pos]),
                "y_pred_label": bundle.label_names[int(y_pred[row_pos])],
            }
            if y_score is not None:
                if y_score.ndim == 1:
                    row["score"] = float(y_score[row_pos])
                else:
                    for class_idx, label in enumerate(bundle.label_names):
                        row[f"score_{_safe_column_name(label)}"] = float(y_score[row_pos, class_idx])
            metadata = metadata_rows[row_pos] if row_pos < len(metadata_rows) else {}
            for key in metadata_keys:
                row[f"metadata_{key}"] = metadata.get(key)
            writer.writerow(row)


def _metadata_rows(metadata: Any, indices: np.ndarray) -> list[dict[str, Any]]:
    if metadata is None:
        return [{} for _ in indices]
    if hasattr(metadata, "iloc"):
        selected = metadata.iloc[indices]
        return [
            {str(key): _json_safe(value) for key, value in row.items()}
            for row in selected.to_dict(orient="records")
        ]
    return [{} for _ in indices]


def _score_columns(y_score: np.ndarray | None, label_names: list[str]) -> list[str]:
    if y_score is None:
        return []
    if y_score.ndim == 1:
        return ["score"]
    return [f"score_{_safe_column_name(label)}" for label in label_names]


def _safe_column_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _safe_stratify(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    return y if counts.size and counts.min() >= 2 else None


def _make_run_dir(output_dir: str, experiment_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in experiment_name)
    run_dir = Path(output_dir) / f"{timestamp}-{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
