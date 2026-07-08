"""Bridge helpers for exporting workbench models into EEGle attention8."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ATTENTION8_CHANNELS = ("Fz", "Cz", "Pz", "C3", "C4", "P3", "P4", "Oz")
ATTENTION8_CHANNEL_GROUPS = {
    "frontal": ["Fz"],
    "central": ["Cz", "C3", "C4"],
    "posterior": ["Pz", "P3", "P4", "Oz"],
}
ATTENTION8_EPOCH_WINDOW_SECONDS = (-2.0, 0.0)

PROFILE_TO_KIND = {
    "log-reg": "causal_bandpower_logreg",
    "riemann": "riemann_tangent_logreg",
    "eegnet": "torch_eegnet",
    "lora": "foundation_head_logreg",
    "film": "foundation_prototype",
    "bendr": "foundation_bendr",
}
PROFILE_ALIASES = {
    "logreg": "log-reg",
    "log_reg": "log-reg",
    "causal_bandpower_logreg": "log-reg",
    "riemannian": "riemann",
    "riemann_tangent": "riemann",
    "riemann_tangent_logreg": "riemann",
    "torch_eegnet": "eegnet",
    "cnn_eegnet": "eegnet",
    "foundation_head": "lora",
    "foundation_head_logreg": "lora",
    "foundation_lora": "lora",
    "foundation_prototype": "film",
    "foundation_film": "film",
    "foundation_bendr": "bendr",
    "bendr_external": "bendr",
}
KIND_TO_PROFILE = {kind: profile for profile, kind in PROFILE_TO_KIND.items()}


class Attention8BridgeError(RuntimeError):
    """Raised when the attention8 export bridge cannot prepare a model."""


@dataclass(frozen=True)
class Attention8ExportOptions:
    """Training/export options shared by native and BENDR exports."""

    session_dirs: tuple[Path, ...] = ()
    epochs_npz: tuple[Path, ...] = ()
    output_dir: Path = Path("runs/attention8_exports")
    target: str = "attention_lapse_binary"
    attention_lapse_label: str = "slow_go_rt"
    slow_rt_quantile: float = 0.8
    support_trials: int = 50
    seed: int = 42
    permutations: int = 100
    model_version: str = "attention8-workbench"
    extra_config: dict[str, Any] = field(default_factory=dict)

    def epoch_paths(self) -> list[Path]:
        paths = [path.expanduser().resolve() for path in self.epochs_npz]
        paths.extend(
            path.expanduser().resolve() / "realtime" / "epochs" / "epochs.npz"
            for path in self.session_dirs
        )
        if not paths:
            raise Attention8BridgeError("--session-dir or --epochs-npz is required")
        return paths


def normalize_profile(value: str) -> str:
    """Normalize user-facing model names to bridge profile names."""

    key = str(value).strip().lower()
    key = PROFILE_ALIASES.get(key, key)
    if key not in PROFILE_TO_KIND:
        choices = ", ".join(sorted(PROFILE_TO_KIND))
        raise Attention8BridgeError(f"unknown attention8 model profile {value!r}; choose one of {choices}")
    return key


def kind_for_profile(value: str) -> str:
    """Return the EEGle model kind backing a workbench profile."""

    return PROFILE_TO_KIND[normalize_profile(value)]


def profile_for_kind(value: str) -> str:
    """Return the workbench profile for a resolved EEGle model kind."""

    return KIND_TO_PROFILE.get(str(value).strip().lower(), normalize_profile(value))


def attention8_input_contract(
    *,
    sample_rate_hz: float | None = None,
    sample_count: int | None = None,
    epoch_window_seconds: tuple[float, float] = ATTENTION8_EPOCH_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Return the expected 8-dry-electrode attention8 input contract."""

    contract: dict[str, Any] = {
        "input_layout": "channels_x_samples",
        "input_units": "microvolts",
        "channel_names": list(ATTENTION8_CHANNELS),
        "channel_order": list(ATTENTION8_CHANNELS),
        "required_channels": list(ATTENTION8_CHANNELS),
        "optional_channels": [],
        "channel_groups": dict(ATTENTION8_CHANNEL_GROUPS),
        "missing_channel_policy": "error",
        "resampling": "none",
        "tensor_layout": "batch_1_channels_samples",
        "epoch_window_seconds": [float(epoch_window_seconds[0]), float(epoch_window_seconds[1])],
        "prediction_window_seconds": [float(epoch_window_seconds[0]), float(epoch_window_seconds[1])],
        "prediction_horizon": "same_trial_response",
        "baseline_seconds": [float(epoch_window_seconds[0]), float(epoch_window_seconds[1])],
    }
    if sample_rate_hz is not None:
        contract["sample_rate_hz"] = float(sample_rate_hz)
    if sample_count is not None:
        contract["sample_count"] = int(sample_count)
    return contract


def attention8_model_config(
    profile_or_kind: str,
    options: Attention8ExportOptions,
) -> dict[str, Any]:
    """Build an EEGle realtime-model config for a workbench profile."""

    profile = normalize_profile(profile_or_kind)
    kind = PROFILE_TO_KIND[profile]
    config: dict[str, Any] = {
        "kind": kind,
        "family": _family_for_kind(kind),
        "target": options.target,
        "attention_lapse_label": options.attention_lapse_label,
        "slow_rt_quantile": float(options.slow_rt_quantile),
        "attention_lapse_threshold": 0.5,
        "prediction_horizon": "same_trial_response",
        "support_trials": max(0, int(options.support_trials)),
        "session_dirs": [str(path.expanduser().resolve()) for path in options.session_dirs],
        "input_layout": "samples_x_channels",
        "input_units": "microvolts",
        "required_channels": list(ATTENTION8_CHANNELS),
        "input_contract": attention8_input_contract(),
        "baseline_seconds": list(ATTENTION8_EPOCH_WINDOW_SECONDS),
        "decision_probability": 0.5,
        "seed": int(options.seed),
        "permutations": int(options.permutations),
        "model_version": f"{options.model_version}-{profile}",
        "calibration": {
            "threshold_metric": "balanced_accuracy",
            "support_trials": max(0, int(options.support_trials)),
            "prototype_distance_metric": "cosine",
        },
        "quality_gate": {
            "minimum_finite_fraction": 1.0,
            "minimum_channel_std_uv": 0.01,
            "max_abs_uv": 250.0,
            "max_peak_to_peak_uv": 400.0,
        },
        "preprocessing": {
            "notch_hz": 60.0,
            "bandpass_low_hz": 1.0,
            "bandpass_high_hz": 45.0,
            "downsample_factor": 1,
            "reference": "average",
        },
        "epoch_data_source": "causal_preprocessed",
        "comparison_profile": _comparison_profile(profile, kind),
    }
    if profile in {"lora", "film"}:
        config["embedding"] = {
            "profile": profile,
            "encoder_update_policy": "frozen",
            "source": "eegle-model-attention8-bridge",
            "note": (
                "Uses EEGle frozen-embedding features so attention8 can compare "
                f"{profile.upper()}-style calibration behavior in realtime."
            ),
        }
    if profile == "eegnet":
        config.update(
            {
                "F1": 8,
                "D": 2,
                "F2": 16,
                "temporal_kernel": 64,
                "dropout": 0.5,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 32,
                "max_epochs": 200,
                "patience": 20,
            }
        )
    config.update(options.extra_config)
    return config


def train_native_attention8_bundle(
    profile_or_kind: str,
    options: Attention8ExportOptions,
    *,
    closedloop_root: str | Path | None = None,
) -> dict[str, Any]:
    """Train a closedloop-native attention8 bundle using EEGle's trainers."""

    profile = normalize_profile(profile_or_kind)
    kind = kind_for_profile(profile)
    if kind == "foundation_bendr":
        raise Attention8BridgeError("BENDR export uses the workbench BENDR path, not EEGle native training")

    ensure_closedloop_importable(closedloop_root)
    train_epoch_model = importlib.import_module("eegle.realtime.models").train_epoch_model
    output_root = options.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = attention8_model_config(profile, options)
    return train_epoch_model(kind, options.epoch_paths(), output_root / kind, config)


def resolve_closedloop_root(value: str | Path | None = None) -> Path:
    """Resolve the closedloop repository path used by the bridge."""

    candidates: list[Path] = []
    if value:
        candidates.append(Path(value))
    if os.environ.get("CLOSEDLOOP_REPO"):
        candidates.append(Path(os.environ["CLOSEDLOOP_REPO"]))
    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[3] / "closedloop",
            Path.cwd().parent / "closedloop",
            Path.cwd(),
        ]
    )
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "eegle" / "realtime" / "models.py").exists():
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise Attention8BridgeError(
        "Could not find the closedloop EEGle repo. Pass --closedloop-root or set CLOSEDLOOP_REPO. "
        f"Searched: {searched}"
    )


def ensure_closedloop_importable(value: str | Path | None = None) -> Path:
    """Resolve closedloop and add it to ``sys.path`` for bridge imports."""

    root = resolve_closedloop_root(value)
    _ensure_on_path(root)
    return root


def _ensure_on_path(root: Path) -> None:
    text = str(root)
    if text not in sys.path:
        sys.path.insert(0, text)


def _family_for_kind(kind: str) -> str:
    if kind in {"torch_eegnet", "torch_shallowconvnet"}:
        return "cnn"
    if kind.startswith("foundation_"):
        return "eeg_foundation"
    return "classical"


def _comparison_profile(profile: str, kind: str) -> dict[str, str]:
    labels = {
        "log-reg": "Causal bandpower logistic regression",
        "riemann": "Riemannian tangent-space logistic regression",
        "eegnet": "Torch EEGNet",
        "lora": "LoRA-style frozen-embedding head",
        "film": "FiLM-style frozen-embedding prototype",
        "bendr": "BENDR TorchScript foundation encoder",
    }
    notes = {
        "lora": (
            "Runtime-compatible calibration profile: frozen embeddings with a logistic head. "
            "It is named for LoRA comparison bookkeeping, not live gradient LoRA updates."
        ),
        "film": (
            "Runtime-compatible calibration profile: frozen embeddings with calibrated prototypes. "
            "It is named for FiLM comparison bookkeeping, not live FiLM conditioning updates."
        ),
        "bendr": "BENDR is exported as a TorchScript shadow model for attention8 realtime inference.",
    }
    return {
        "method": profile,
        "model_kind": kind,
        "label": labels.get(profile, kind),
        "implementation_note": notes.get(profile, f"Implemented as {kind}."),
    }
