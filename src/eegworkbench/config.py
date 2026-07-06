"""Configuration objects for EEG workbench experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    source: str = "moabb"
    name: str = "BNCI2014_001"
    paradigm: str = "left_right_imagery"
    subjects: list[int] = field(default_factory=lambda: [1])
    data_dir: str = "data/moabb"
    fmin: float | None = 4.0
    fmax: float | None = 38.0
    tmin: float | None = 0.0
    tmax: float | None = None
    resample: float | None = 128.0
    channels: list[str] | None = None
    standardize: str = "trial"
    synthetic_trials: int = 160
    synthetic_channels: int = 22
    synthetic_times: int = 512
    synthetic_classes: int = 2


@dataclass
class ModelConfig:
    name: str = "eegnet"
    framework: str = "torch"
    kwargs: dict[str, Any] = field(default_factory=dict)
    external: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    class_weight: str | list[float] | None = None
    valid_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42
    patience: int = 8
    num_workers: int = 0
    device: str = "auto"
    amp: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "eeg_experiment"
    output_dir: str = "runs"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from YAML."""

    yaml = _yaml_module()
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return experiment_config_from_mapping(raw)


def experiment_config_from_mapping(raw: dict[str, Any]) -> ExperimentConfig:
    """Build a typed experiment config while tolerating missing fields."""

    dataset = _dataclass_from_mapping(DatasetConfig, raw.get("dataset", {}))
    model = _dataclass_from_mapping(ModelConfig, raw.get("model", {}))
    training = _dataclass_from_mapping(TrainingConfig, raw.get("training", {}))
    root_values = {
        key: value
        for key, value in raw.items()
        if key not in {"dataset", "model", "training"}
    }
    return ExperimentConfig(
        dataset=dataset,
        model=model,
        training=training,
        **_known_kwargs(ExperimentConfig, root_values),
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Write a resolved config as YAML."""

    yaml = _yaml_module()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_plain_data(config), handle, sort_keys=False)


def to_plain_data(value: Any) -> Any:
    """Convert dataclasses and paths into JSON/YAML-friendly containers."""

    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    return value


def apply_cli_overrides(
    config: ExperimentConfig,
    *,
    dataset: str | None = None,
    model: str | None = None,
    subjects: list[int] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    output_dir: str | None = None,
    smoke: bool = False,
) -> ExperimentConfig:
    """Apply lightweight CLI overrides to an existing config object."""

    if dataset:
        config.dataset.name = dataset
    if model:
        config.model.name = model
    if subjects:
        config.dataset.subjects = subjects
    if epochs is not None:
        config.training.epochs = epochs
    if batch_size is not None:
        config.training.batch_size = batch_size
    if device:
        config.training.device = device
    if output_dir:
        config.output_dir = output_dir
    if smoke:
        config.dataset.source = "synthetic"
        config.dataset.synthetic_trials = min(config.dataset.synthetic_trials, 96)
        config.training.epochs = min(config.training.epochs, 2)
        config.training.batch_size = min(config.training.batch_size, 32)
        config.experiment_name = f"{config.experiment_name}_smoke"
    return config


def _dataclass_from_mapping(cls: type[Any], raw: dict[str, Any]) -> Any:
    return cls(**_known_kwargs(cls, raw or {}))


def _known_kwargs(cls: type[Any], raw: dict[str, Any]) -> dict[str, Any]:
    names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {key: value for key, value in raw.items() if key in names}


def _yaml_module() -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading/writing YAML configs requires PyYAML.") from exc
    return yaml
