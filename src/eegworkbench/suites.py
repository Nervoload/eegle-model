"""Suite runner for repeatable groups of EEG experiments."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from eegworkbench.config import (
    apply_cli_overrides,
    experiment_config_from_mapping,
    load_config,
    to_plain_data,
)
from eegworkbench.training import run_experiment


def run_suite(
    suite_path: str | Path,
    *,
    subjects: list[int] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    output_dir: str | None = None,
    include_disabled: bool = False,
    continue_on_error: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    suite_file = Path(suite_path)
    suite = _load_yaml(suite_file)
    suite_name = suite.get("suite_name", suite_file.stem)
    suite_output = Path(output_dir or suite.get("output_dir", "runs/suites"))
    suite_run_dir = suite_output / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{suite_name}"
    suite_run_dir.mkdir(parents=True, exist_ok=False)

    results: list[dict[str, Any]] = []
    for item in suite.get("experiments", []):
        if not item.get("enabled", True) and not include_disabled:
            results.append({"name": item.get("name", "unnamed"), "status": "skipped"})
            continue

        experiment_name = item.get("name") or Path(item["config"]).stem
        try:
            config = _build_experiment_config(suite_file, item)
            config.experiment_name = experiment_name
            config.output_dir = str(suite_run_dir)
            config = apply_cli_overrides(
                config,
                subjects=subjects,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )
            if dry_run:
                results.append(
                    {
                        "name": experiment_name,
                        "status": "dry_run",
                        "config": to_plain_data(config),
                    }
                )
                continue
            summary = run_experiment(config)
            results.append(
                {
                    "name": experiment_name,
                    "status": "ok",
                    "run_dir": summary["run_dir"],
                    "metrics": _compact_metrics(summary.get("test_metrics", {})),
                }
            )
        except Exception as exc:
            results.append({"name": experiment_name, "status": "failed", "error": str(exc)})
            if not continue_on_error:
                break

    payload = {
        "suite_name": suite_name,
        "suite_run_dir": str(suite_run_dir),
        "results": results,
    }
    _write_json(suite_run_dir / "suite_summary.json", payload)
    _write_suite_csv(suite_run_dir / "suite_summary.csv", results)
    return payload


def _build_experiment_config(suite_file: Path, item: dict[str, Any]) -> Any:
    config_path = _resolve_path(suite_file, item["config"])
    config = load_config(config_path)
    raw = to_plain_data(config)
    _deep_update(raw, deepcopy(item.get("overrides", {})))
    return experiment_config_from_mapping(raw)


def _resolve_path(suite_file: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repo_root = _find_repo_root(suite_file.resolve())
    from_suite = (suite_file.parent / candidate).resolve()
    if from_suite.exists():
        return from_suite
    return (repo_root / candidate).resolve()


def _find_repo_root(path: Path) -> Path:
    for parent in [path.parent, *path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return path.parents[1]


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    wanted = [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "roc_auc",
        "average_precision",
        "positive_prevalence",
    ]
    return {key: metrics.get(key) for key in wanted if key in metrics}


def _write_suite_csv(path: Path, results: list[dict[str, Any]]) -> None:
    metric_names = sorted(
        {metric for result in results for metric in result.get("metrics", {}).keys()}
    )
    fieldnames = ["name", "status", "run_dir", "error", *metric_names]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "name": result.get("name"),
                "status": result.get("status"),
                "run_dir": result.get("run_dir"),
                "error": result.get("error"),
            }
            row.update(result.get("metrics", {}))
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Suite files require PyYAML.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
