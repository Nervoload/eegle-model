#!/usr/bin/env python
"""Export attention8-ready model bundles for the closedloop EEGle runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eegworkbench.attention8 import (  # noqa: E402
    Attention8BridgeError,
    Attention8ExportOptions,
    attention8_model_config,
    ensure_closedloop_importable,
    kind_for_profile,
    normalize_profile,
    train_native_attention8_bundle,
)
from eegworkbench.config import ModelConfig, TrainingConfig  # noqa: E402
from eegworkbench.models import build_model  # noqa: E402
from eegworkbench.training.torch_loop import fit_torch_classifier, predict_torch_probabilities  # noqa: E402


DEFAULT_PROFILES = ("eegnet", "lora", "film")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    options = Attention8ExportOptions(
        session_dirs=tuple(Path(value) for value in args.session_dir),
        epochs_npz=tuple(Path(value) for value in args.epochs_npz),
        output_dir=Path(args.output_dir),
        target=args.target,
        attention_lapse_label=args.attention_lapse_label,
        slow_rt_quantile=args.slow_rt_quantile,
        support_trials=args.support_trials,
        seed=args.seed,
        permutations=args.permutations,
        model_version=args.model_version,
    )
    profiles = _profiles(args)
    results: dict[str, Any] = {}
    for profile in profiles:
        try:
            if profile == "bendr":
                results[profile] = train_bendr_attention8_bundle(args, options)
            else:
                results[profile] = train_native_attention8_bundle(
                    profile,
                    options,
                    closedloop_root=args.closedloop_root,
                )
        except Exception as exc:
            results[profile] = {
                "status": "failed",
                "profile": profile,
                "model_kind": _safe_kind(profile),
                "error": f"{type(exc).__name__}: {exc}",
            }
            if args.stop_on_error:
                break

    complete = [item for item in results.values() if item.get("status") == "ok"]
    status = "ok" if len(complete) == len(results) else ("degraded" if complete else "failed")
    summary = {
        "status": status,
        "workflow": "eegle-model.export_attention8",
        "model_dir": str(Path(args.output_dir).expanduser().resolve()),
        "profiles": profiles,
        "models": results,
        "next_steps": _next_steps(Path(args.output_dir), profiles),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if status in {"ok", "degraded"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closedloop-root", help="Path to the closedloop repository; defaults to CLOSEDLOOP_REPO or sibling ../closedloop.")
    parser.add_argument("--session-dir", action="append", default=[], help="attention8 calibration session directory; repeat for multi-session training.")
    parser.add_argument("--epochs-npz", action="append", default=[], help="Direct realtime/epochs/epochs.npz path; repeat for multi-session training.")
    parser.add_argument("--output-dir", default="runs/attention8_exports", help="Output model root consumed by attention8 online.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=("log-reg", "riemann", "eegnet", "lora", "film", "bendr"),
        help="Profile to export. Defaults to eegnet, lora, and film.",
    )
    parser.add_argument("--all", action="store_true", help="Export all profiles, including BENDR when BENDR inputs are provided.")
    parser.add_argument("--target", default="attention_lapse_binary")
    parser.add_argument("--attention-lapse-label", default="slow_go_rt")
    parser.add_argument("--slow-rt-quantile", type=float, default=0.8)
    parser.add_argument("--support-trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--model-version", default="attention8-workbench")
    parser.add_argument("--stop-on-error", action="store_true")

    bendr = parser.add_argument_group("BENDR")
    bendr.add_argument("--bendr-repo", help="BENDR repo path or parent eeg-foundation directory.")
    bendr.add_argument("--bendr-encoder", help="BENDR encoder checkpoint path.")
    bendr.add_argument("--bendr-context", help="BENDR contextualizer checkpoint path.")
    bendr.add_argument("--bendr-random-init", action="store_true", help="Build BENDR without pretrained weights for wiring tests.")
    bendr.add_argument("--bendr-architecture", default="bendr", choices=("bendr", "linear_head"))
    bendr.add_argument("--bendr-epochs", type=int, default=10)
    bendr.add_argument("--bendr-batch-size", type=int, default=16)
    bendr.add_argument("--bendr-learning-rate", type=float, default=1e-4)
    bendr.add_argument("--bendr-weight-decay", type=float, default=1e-4)
    bendr.add_argument("--bendr-device", default="auto")
    bendr.add_argument("--bendr-no-amp", action="store_true")
    return parser


def train_bendr_attention8_bundle(
    args: argparse.Namespace,
    options: Attention8ExportOptions,
) -> dict[str, Any]:
    """Train BENDR on attention8 epochs and export a closedloop-native bundle."""

    if not args.bendr_repo:
        raise Attention8BridgeError("BENDR export requires --bendr-repo so the workbench can import BENDR/dn3_ext.")
    if not args.bendr_random_init and not (args.bendr_encoder and args.bendr_context):
        raise Attention8BridgeError(
            "BENDR pretrained export requires --bendr-encoder and --bendr-context, "
            "or --bendr-random-init for a wiring-only bundle."
        )
    _set_env("BENDR_REPO", args.bendr_repo)
    _set_env("BENDR_ENCODER_WEIGHTS", args.bendr_encoder)
    _set_env("BENDR_CONTEXT_WEIGHTS", args.bendr_context)

    ensure_closedloop_importable(args.closedloop_root)
    cl_models = __import__("eegle.realtime.models", fromlist=["dummy"])
    cl_bundles = __import__("eegle.models.bundles", fromlist=["dummy"])
    cl_calibration = __import__("eegle.ml.calibration", fromlist=["dummy"])

    data_payload = _load_attention8_training_data(cl_models, options)
    x_fit = data_payload["x_fit"]
    y_fit = data_payload["y_fit"]
    model = build_model(
        _bendr_model_config(args),
        n_chans=int(x_fit.shape[1]),
        n_outputs=2,
        n_times=int(x_fit.shape[2]),
    )
    training = TrainingConfig(
        epochs=int(args.bendr_epochs),
        batch_size=int(args.bendr_batch_size),
        learning_rate=float(args.bendr_learning_rate),
        weight_decay=float(args.bendr_weight_decay),
        class_weight="balanced",
        valid_size=0.2,
        test_size=0.0,
        seed=int(options.seed),
        patience=max(2, min(8, int(args.bendr_epochs))),
        num_workers=0,
        device=str(args.bendr_device),
        amp=not bool(args.bendr_no_amp),
    )
    train_split, valid_split = _train_valid_split(x_fit, y_fit, seed=int(options.seed))
    output_root = options.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bendr-train-", dir=str(output_root)) as tmp:
        run_dir = Path(tmp)
        history = fit_torch_classifier(model, train_split, valid_split, training=training, run_dir=run_dir)
        probabilities = predict_torch_probabilities(model, x_fit, training=training)[:, 1]
        artifact_source = run_dir / "foundation_bendr.ts"
        _save_4d_torchscript(model, artifact_source, n_chans=int(x_fit.shape[1]), n_times=int(x_fit.shape[2]))
        threshold_calibration = cl_calibration.select_binary_threshold(
            y_fit,
            probabilities,
            metric="balanced_accuracy",
            max_candidates=512,
            positive_label="attention_lapse",
        )
        threshold_calibration = _with_calibration_id({**threshold_calibration, "source": "training_fit"})
        selected_threshold = float(threshold_calibration.get("selected_threshold", 0.5))
        metrics = cl_models.binary_classification_metrics(
            y_fit,
            probabilities,
            threshold=selected_threshold,
            positive_label="attention_lapse",
        )
        metrics.update(
            {
                "target": options.target,
                "target_spec": data_payload["target_spec"],
                "evaluation_level": "training_fit",
                "threshold_source": threshold_calibration.get("source"),
                "threshold_calibration": threshold_calibration,
                "selected_threshold_metrics": threshold_calibration.get("selected_metrics"),
                "training_history": history,
                "support_query": data_payload["support_query"],
                "query_metrics": _query_metrics(cl_models, data_payload, model, training, selected_threshold),
            }
        )
        manifest = cl_bundles.write_model_bundle(
            output_root / "foundation_bendr",
            kind="foundation_bendr",
            artifact_path=artifact_source,
            artifact_format="torchscript",
            contract=data_payload["contract"],
            metrics=metrics,
            training_source={
                **cl_models.training_source_provenance(data_payload["data"], data_payload["source_paths"]),
                "training_epochs": int(x_fit.shape[0]),
                "query_epochs": int(data_payload["query_epoch_count"]),
                "producer": "eegle-model/scripts/export_attention8.py",
            },
            extra={
                "model_version": f"{options.model_version}-bendr",
                "model_family": "eeg_foundation",
                "target": options.target,
                "target_spec": data_payload["target_spec"],
                "label_mapping": data_payload["label_mapping"],
                "calibration": threshold_calibration,
                "support_query": data_payload["support_query"],
                "comparison_profile": attention8_model_config("bendr", options)["comparison_profile"],
            },
        )
    return {
        "status": "ok",
        "profile": "bendr",
        "model_kind": "foundation_bendr",
        "model_family": "eeg_foundation",
        "target": options.target,
        "bundle_path": str(output_root / "foundation_bendr"),
        "bundle_hash": manifest["bundle_hash"],
        "training_epochs": int(x_fit.shape[0]),
        "query_epochs": int(data_payload["query_epoch_count"]),
        "classes": sorted(int(value) for value in set(y_fit.tolist())),
        "channel_names": list(data_payload["contract"]["channel_names"]),
        "sample_rate_hz": float(data_payload["contract"]["sample_rate_hz"]),
        "epoch_window_seconds": data_payload["contract"]["epoch_window_seconds"],
        "calibration": threshold_calibration,
        "metrics": metrics,
    }


def _load_attention8_training_data(cl_models: Any, options: Attention8ExportOptions) -> dict[str, Any]:
    cfg = attention8_model_config("bendr", options)
    data, source_paths = cl_models.load_epoch_dataset(options.epoch_paths())
    x = np.asarray(data["X"], dtype=float)
    target = __import__("eegle.ml.targets", fromlist=["dummy"]).build_training_target(data, cfg)
    y = np.asarray(target.y, dtype=int)
    trials = np.asarray(cl_models.npz_value(data, "trials", np.arange(y.size) + 1), dtype=int)
    valid = np.asarray(target.eligible, dtype=bool) & (y >= 0) & (trials >= 1)
    x = x[valid]
    y = y[valid]
    trials = trials[valid]
    if x.shape[0] == 0 or len(set(y.tolist())) < 2:
        raise Attention8BridgeError(f"BENDR training requires both classes for target {target.name}")

    quality_config = dict(cfg.get("quality_gate", {}))
    quality_valid = np.asarray([cl_models.assess_epoch_quality(epoch.T, quality_config).valid for epoch in x], dtype=bool)
    x = x[quality_valid]
    y = y[quality_valid]
    trials = trials[quality_valid]
    if x.shape[0] == 0 or len(set(y.tolist())) < 2:
        raise Attention8BridgeError("BENDR training data lost one or both classes after quality gating")

    contract = cl_models.training_contract(data, cfg)
    times = np.asarray(cl_models.npz_value(data, "times", []), dtype=float)
    source_channel_names = [str(value) for value in contract.get("source_channel_names", contract["channel_names"])]
    corrected = np.stack(
        [
            cl_models.prepare_classifier_epoch(
                epoch,
                float(contract["sample_rate_hz"]),
                source_channel_names,
                {"relative_times": times.tolist(), "epoch_window_seconds": contract["epoch_window_seconds"]},
                contract,
            )[0]
            for epoch in x
        ],
        axis=0,
    )
    support_info = cl_models.temporal_support_query_split(trials, y, cfg)
    fit_mask = np.asarray(support_info["support_mask"], dtype=bool)
    query_mask = np.asarray(support_info["query_mask"], dtype=bool)
    x_fit = corrected[fit_mask]
    y_fit = y[fit_mask]
    if x_fit.shape[0] == 0 or len(set(y_fit.tolist())) < 2:
        raise Attention8BridgeError(
            f"BENDR support set requires both classes; support_trials={support_info['support_trials']}"
        )
    return {
        "data": data,
        "source_paths": source_paths,
        "x_fit": x_fit.astype("float32"),
        "y_fit": y_fit.astype("int64"),
        "x_query": corrected[query_mask].astype("float32"),
        "y_query": y[query_mask].astype("int64"),
        "query_epoch_count": int(np.sum(query_mask)),
        "contract": contract,
        "target_spec": target.metadata,
        "label_mapping": target.label_mapping,
        "support_query": support_info["payload"],
    }


def _bendr_model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        name="bendr",
        framework="torch",
        external={
            "repo_path": args.bendr_repo or os.environ.get("BENDR_REPO", ""),
            "module": "eegworkbench.adapters.bendr",
            "factory": "build_classifier",
            "checkpoint": None,
            "strict": False,
            "kwargs": {
                "architecture": args.bendr_architecture,
                "encoder_weights": args.bendr_encoder or os.environ.get("BENDR_ENCODER_WEIGHTS"),
                "context_weights": args.bendr_context or os.environ.get("BENDR_CONTEXT_WEIGHTS"),
                "random_init": bool(args.bendr_random_init),
                "freeze_encoder": True,
                "freeze_contextualizer": False,
                "freeze_position_conv": False,
                "strict": False,
                "model_kwargs": {
                    "encoder_h": 512,
                    "contextualizer_hidden": 3076,
                    "dropout": 0.1,
                },
            },
        },
    )


def _train_valid_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    labels, counts = np.unique(y, return_counts=True)
    if y.size < 8 or labels.size < 2 or counts.min() < 2:
        return (x, y), (x, y)
    from sklearn.model_selection import train_test_split

    train_idx, valid_idx = train_test_split(
        np.arange(y.size),
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    return (x[train_idx], y[train_idx]), (x[valid_idx], y[valid_idx])


def _save_4d_torchscript(model: Any, path: Path, *, n_chans: int, n_times: int) -> None:
    import torch

    class FourDimensionalEpochWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, inputs: Any) -> Any:
            values = inputs[:, 0, :, :] if inputs.dim() == 4 else inputs
            output = self.inner(values)
            if isinstance(output, (list, tuple)):
                output = output[0]
            while output.dim() > 2 and output.shape[-1] == 1:
                output = output.squeeze(-1)
            if output.dim() > 2:
                output = output.flatten(start_dim=1)
            return output

    wrapper = FourDimensionalEpochWrapper(model.cpu().eval())
    example = torch.zeros((1, 1, int(n_chans), int(n_times)), dtype=torch.float32)
    traced = torch.jit.trace(wrapper, example)
    traced.save(str(path))


def _query_metrics(
    cl_models: Any,
    data_payload: dict[str, Any],
    model: Any,
    training: TrainingConfig,
    selected_threshold: float,
) -> dict[str, Any] | None:
    x_query = data_payload["x_query"]
    y_query = data_payload["y_query"]
    if x_query.shape[0] == 0 or len(set(y_query.tolist())) < 2:
        return None
    probabilities = predict_torch_probabilities(model, x_query, training=training)[:, 1]
    metrics = cl_models.binary_classification_metrics(
        y_query,
        probabilities,
        threshold=selected_threshold,
        positive_label="attention_lapse",
    )
    metrics["evaluation_level"] = "temporal_query"
    metrics["query_epoch_count"] = int(x_query.shape[0])
    return metrics


def _with_calibration_id(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("calibration_id"):
        return payload
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return {**payload, "calibration_id": "cal-" + hashlib.sha256(encoded).hexdigest()[:12]}


def _profiles(args: argparse.Namespace) -> list[str]:
    if args.all:
        values = ["log-reg", "riemann", "eegnet", "lora", "film", "bendr"]
    else:
        values = args.profile or list(DEFAULT_PROFILES)
    return list(dict.fromkeys(normalize_profile(value) for value in values))


def _safe_kind(profile: str) -> str | None:
    try:
        return kind_for_profile(profile)
    except Exception:
        return None


def _set_env(name: str, value: str | None) -> None:
    if value:
        os.environ[name] = value


def _next_steps(output_dir: Path, profiles: list[str]) -> dict[str, str]:
    model_dir = str(output_dir.expanduser().resolve())
    shadows = " ".join(
        f"--shadow {kind_for_profile(profile)}"
        for profile in profiles
        if kind_for_profile(profile) != "torch_eegnet"
    )
    return {
        "check_aliases": "eegle model-list",
        "online": f"attention8 online --participant sub-001 --model-dir {model_dir} --primary torch_eegnet {shadows}".strip(),
        "compare": f"attention8 compare --session-dir <calibration-session> --method eegnet --method lora --method film --output-dir {model_dir}/comparison",
    }


if __name__ == "__main__":
    raise SystemExit(main())
