"""Load EEG datasets from MOABB or synthetic smoke data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eegworkbench.config import DatasetConfig


@dataclass
class EEGDataBundle:
    X: np.ndarray
    y: np.ndarray
    label_names: list[str]
    metadata: Any | None = None
    source: str = "unknown"

    @property
    def n_trials(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_chans(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.X.shape[2])

    @property
    def n_outputs(self) -> int:
        return len(self.label_names)


def load_eeg_data(config: DatasetConfig) -> EEGDataBundle:
    """Load an EEG dataset according to the configured source."""

    if config.source.lower() == "synthetic":
        return _make_synthetic_data(config)
    if config.source.lower() == "moabb":
        return _load_moabb_data(config)
    raise ValueError(f"Unsupported dataset source: {config.source!r}")


def _make_synthetic_data(config: DatasetConfig) -> EEGDataBundle:
    rng = np.random.default_rng(7)
    n_trials = config.synthetic_trials
    n_chans = config.synthetic_channels
    n_times = config.synthetic_times
    n_classes = config.synthetic_classes

    X = rng.normal(0.0, 0.15, size=(n_trials, n_chans, n_times)).astype("float32")
    y = np.arange(n_trials, dtype="int64") % n_classes

    if config.paradigm.lower() == "p300" and n_classes == 2:
        return _make_synthetic_p300_data(config, rng)

    time = np.linspace(0, 1, n_times, dtype="float32")
    for class_id in range(n_classes):
        class_mask = y == class_id
        freq = 8.0 + class_id * 4.0
        signal = np.sin(2 * np.pi * freq * time).astype("float32")
        X[class_mask, class_id % n_chans, :] += signal

    return EEGDataBundle(
        X=X,
        y=y,
        label_names=[f"class_{idx}" for idx in range(n_classes)],
        source="synthetic",
    )


def _make_synthetic_p300_data(config: DatasetConfig, rng: np.random.Generator) -> EEGDataBundle:
    n_trials = config.synthetic_trials
    n_chans = config.synthetic_channels
    n_times = config.synthetic_times
    target_ratio = 0.18

    X = rng.normal(0.0, 0.25, size=(n_trials, n_chans, n_times)).astype("float32")
    y = (rng.random(n_trials) < target_ratio).astype("int64")
    if y.sum() == 0:
        y[0] = 1

    time = np.linspace(0, 1, n_times, dtype="float32")
    p300 = np.exp(-0.5 * ((time - 0.32) / 0.075) ** 2).astype("float32")
    slow_wave = 0.45 * np.exp(-0.5 * ((time - 0.48) / 0.11) ** 2).astype("float32")
    posterior_channels = np.arange(max(1, n_chans // 2), n_chans)
    for trial_idx in np.flatnonzero(y == 1):
        channel_scale = rng.normal(1.0, 0.1, size=len(posterior_channels)).astype("float32")
        X[trial_idx, posterior_channels, :] += channel_scale[:, None] * (p300 + slow_wave)

    return EEGDataBundle(
        X=X,
        y=y,
        label_names=["NonTarget", "Target"],
        source="synthetic:p300",
    )


def _load_moabb_data(config: DatasetConfig) -> EEGDataBundle:
    try:
        import mne
        import moabb
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        raise RuntimeError(
            "MOABB loading requires mne, moabb, and scikit-learn. "
            "Install the EEG environment before running a MOABB experiment."
        ) from exc

    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    mne.set_config("MNE_DATA", str(data_dir.resolve()), set_env=True)
    moabb.set_log_level("info")

    dataset = _build_moabb_dataset(config.name)
    paradigm = _build_moabb_paradigm(config)
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=config.subjects)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype("int64")
    X = np.asarray(X, dtype="float32")
    if X.ndim != 3:
        raise ValueError(f"Expected MOABB data shaped (trials, channels, time), got {X.shape}")

    return EEGDataBundle(
        X=X,
        y=y,
        label_names=[str(label) for label in encoder.classes_],
        metadata=metadata,
        source=f"moabb:{config.name}",
    )


def _build_moabb_dataset(name: str) -> Any:
    import moabb.datasets as datasets

    try:
        dataset_cls = getattr(datasets, name)
    except AttributeError as exc:
        available = sorted(item for item in dir(datasets) if not item.startswith("_"))
        raise ValueError(
            f"Unknown MOABB dataset {name!r}. Examples: {', '.join(available[:12])}"
        ) from exc
    return dataset_cls()


def _build_moabb_paradigm(config: DatasetConfig) -> Any:
    from moabb import paradigms

    name = config.paradigm.lower()
    mapping = {
        "left_right_imagery": "LeftRightImagery",
        "motor_imagery": "MotorImagery",
        "p300": "P300",
        "ssvep": "SSVEP",
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported MOABB paradigm {config.paradigm!r}. "
            f"Choose one of: {', '.join(mapping)}"
        )

    paradigm_cls = getattr(paradigms, mapping[name])
    kwargs = {
        "fmin": config.fmin,
        "fmax": config.fmax,
        "tmin": config.tmin,
        "tmax": config.tmax,
        "resample": config.resample,
        "channels": config.channels,
    }
    return _construct_with_supported_kwargs(paradigm_cls, kwargs)


def _construct_with_supported_kwargs(cls: type[Any], kwargs: dict[str, Any]) -> Any:
    import inspect

    signature = inspect.signature(cls)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in signature.parameters
    }
    return cls(**accepted)
