"""External foundation-model adapter support."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any


class ExternalModelError(RuntimeError):
    """Raised when an external model adapter is not configured correctly."""


def build_external_torch_model(
    external_config: dict[str, Any],
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
) -> Any:
    """Load a torch model from a user-supplied repo/module/factory.

    The factory must return an ``nn.Module`` whose forward pass accepts tensors
    shaped ``(batch, channels, time)`` and returns class logits.
    """

    if not external_config:
        raise ExternalModelError(_external_help())

    repo_path = external_config.get("repo_path")
    module_name = external_config.get("module")
    factory_name = external_config.get("factory")
    if not module_name or not factory_name:
        raise ExternalModelError(_external_help())

    if repo_path:
        resolved_repo = str(_expand_path(repo_path).resolve())
        if resolved_repo not in sys.path:
            sys.path.insert(0, resolved_repo)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ExternalModelError(
            f"Could not import external module {module_name!r}. "
            "Check repo_path/module in the config."
        ) from exc

    factory = _resolve_attr(module, factory_name)
    kwargs = dict(external_config.get("kwargs", {}))
    kwargs = _add_shape_kwargs(factory, kwargs, n_chans=n_chans, n_outputs=n_outputs, n_times=n_times)
    model = factory(**kwargs)

    checkpoint = external_config.get("checkpoint")
    if checkpoint:
        _load_checkpoint(model, checkpoint, strict=bool(external_config.get("strict", False)))
    return model


def _expand_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.fspath(value))).expanduser()


def _resolve_attr(module: Any, dotted_name: str) -> Any:
    value = module
    for part in dotted_name.split("."):
        value = getattr(value, part)
    return value


def _add_shape_kwargs(
    factory: Any,
    kwargs: dict[str, Any],
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
) -> dict[str, Any]:
    signature = inspect.signature(factory)
    synonyms = {
        "n_chans": n_chans,
        "in_chans": n_chans,
        "channels": n_chans,
        "n_outputs": n_outputs,
        "n_classes": n_outputs,
        "num_classes": n_outputs,
        "n_times": n_times,
        "input_window_samples": n_times,
        "samples": n_times,
    }
    for key, value in synonyms.items():
        if key in signature.parameters and key not in kwargs:
            kwargs[key] = value
    return kwargs


def _load_checkpoint(model: Any, checkpoint: str, *, strict: bool) -> None:
    try:
        import torch
    except ImportError as exc:
        raise ExternalModelError("Loading checkpoints requires PyTorch.") from exc

    checkpoint_path = _expand_path(checkpoint)
    if not checkpoint_path.exists():
        raise ExternalModelError(f"Checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload.get("model", payload))
    cleaned = {
        key.removeprefix("module.").removeprefix("model."): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=strict)


def _external_help() -> str:
    return (
        "External foundation models need a config with external.repo_path, "
        "external.module, and external.factory. The factory should return a "
        "torch nn.Module classifier with logits shaped (batch, classes)."
    )
