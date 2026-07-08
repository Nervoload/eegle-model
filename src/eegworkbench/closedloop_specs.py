"""Optional EEGle model-registry aliases for attention8 workbench exports.

This module is loaded by EEGle through the ``eegle.models`` entry point when
``eegle-model`` is installed in the same Python environment as ``closedloop``.
It intentionally imports EEGle only inside the entry-point function so the
workbench can still be used without the realtime repository installed.
"""

from __future__ import annotations


def attention8_model_specs() -> tuple[object, ...]:
    """Return EEGle ``ModelSpec`` entries with workbench-friendly aliases."""

    try:
        from eegle.ml.registry_types import ModelSpec
    except ImportError as exc:  # pragma: no cover - only hit outside EEGle envs.
        raise RuntimeError(
            "The attention8 alias entry point requires the closedloop EEGle "
            "package to be installed in the active environment."
        ) from exc

    return (
        ModelSpec(
            kind="foundation_head_logreg",
            family="eeg_foundation",
            description="LoRA-style frozen EEG embedding plus logistic-regression calibration head.",
            aliases=("lora", "foundation_lora", "foundation_head"),
            adapter_kind="foundation_head_logreg",
            train_kind="foundation_head_logreg",
            trainable=True,
            realtime_supported=True,
            primary_realtime_allowed=False,
            dependencies=("sklearn", "joblib"),
            artifact_format="joblib",
            checkpoint_format="eegle_native_bundle_or_runtime_zip",
            supported_targets=("attention_lapse_binary", "attention_lapse_score"),
            latency_budget_ms=100.0,
            external_checkpoint=True,
        ),
        ModelSpec(
            kind="foundation_prototype",
            family="eeg_foundation",
            description="FiLM-style frozen EEG embeddings with calibrated lapse/non-lapse prototypes.",
            aliases=("film", "foundation_film"),
            adapter_kind="foundation_prototype",
            train_kind="foundation_prototype",
            trainable=True,
            realtime_supported=True,
            primary_realtime_allowed=False,
            dependencies=("sklearn", "joblib"),
            artifact_format="joblib",
            checkpoint_format="eegle_native_bundle_or_runtime_zip",
            supported_targets=("attention_lapse_binary", "attention_lapse_score"),
            latency_budget_ms=100.0,
            external_checkpoint=True,
        ),
        ModelSpec(
            kind="torch_eegnet",
            family="cnn",
            description="Trainable TorchScript EEGNet-style compact convolutional epoch model.",
            aliases=("eegnet", "cnn_eegnet"),
            adapter_kind="torch_eegnet",
            train_kind="torch_eegnet",
            trainable=True,
            realtime_supported=True,
            primary_realtime_allowed=True,
            dependencies=("torch",),
            artifact_format="torchscript",
            checkpoint_format="torchscript",
            supported_targets=("condition", "attention_lapse_binary", "attention_lapse_score"),
            latency_budget_ms=50.0,
        ),
        ModelSpec(
            kind="foundation_bendr",
            family="eeg_foundation",
            description="BENDR-style pretrained EEG encoder exported from eegle-model as TorchScript.",
            aliases=("bendr", "bendr_external"),
            adapter_kind="foundation_bendr",
            train_kind=None,
            trainable=False,
            realtime_supported=True,
            primary_realtime_allowed=False,
            dependencies=("torch",),
            artifact_format="torchscript",
            checkpoint_format="torchscript_or_repo_checkpoint",
            supported_targets=("condition", "attention_lapse_binary", "attention_lapse_score"),
            latency_budget_ms=250.0,
            external_checkpoint=True,
        ),
    )
