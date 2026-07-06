"""BENDR adapter for the EEG workbench external torch model contract.

This module expects the BENDR repository to be importable. The workbench's
external loader adds ``model.external.repo_path`` to ``sys.path`` before
importing this adapter, so configs can point directly at a future local clone.
"""

from __future__ import annotations

import collections
import collections.abc
import importlib
import importlib.util
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def build_classifier(
    *,
    n_chans: int | None = None,
    in_chans: int | None = None,
    n_outputs: int | None = None,
    n_classes: int | None = None,
    num_classes: int | None = None,
    n_times: int | None = None,
    input_window_samples: int | None = None,
    architecture: str = "bendr",
    encoder_weights: str | None = None,
    context_weights: str | None = None,
    random_init: bool = False,
    freeze_encoder: bool = True,
    freeze_contextualizer: bool = False,
    freeze_position_conv: bool = False,
    strict: bool = False,
    model_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Build a BENDR classifier compatible with the workbench trainer."""

    channels = _first_int(n_chans, in_chans, name="n_chans")
    targets = _first_int(n_outputs, n_classes, num_classes, name="n_outputs")
    samples = _first_int(n_times, input_window_samples, name="n_times")
    _patch_collections_abc_aliases()

    bendr_module = _import_dn3_ext()
    BENDRClassification = bendr_module.BENDRClassification
    LinearHeadBENDR = bendr_module.LinearHeadBENDR

    architecture = architecture.lower()
    if architecture in {"bendr", "contextual", "contextualizer", "full"}:
        model_cls = BENDRClassification
    elif architecture in {"linear", "linear_head", "linear-head"}:
        model_cls = LinearHeadBENDR
    else:
        raise ValueError("architecture must be 'bendr' or 'linear_head'")

    model = model_cls(targets=targets, samples=samples, channels=channels, **(model_kwargs or {}))
    if random_init:
        return model

    encoder_path = _required_existing_path("encoder_weights", encoder_weights)
    context_path = _required_existing_path("context_weights", context_weights)
    _load_pretrained_modules(
        model,
        encoder_path,
        context_path,
        strict=strict,
        freeze_encoder=freeze_encoder,
        freeze_contextualizer=freeze_contextualizer,
        freeze_position_conv=freeze_position_conv,
    )
    return model


def _import_dn3_ext() -> Any:
    spec = importlib.util.find_spec("dn3_ext")
    if spec is None:
        raise RuntimeError(
            "Could not find BENDR's dn3_ext.py on sys.path. Set BENDR_REPO or "
            "model.external.repo_path to the BENDR clone or its parent "
            f"eeg-foundation folder. First sys.path entries: {sys.path[:8]}"
        )

    try:
        return importlib.import_module("dn3_ext")
    except Exception as exc:
        short_trace = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise RuntimeError(
            f"Found dn3_ext at {spec.origin}, but importing it failed inside "
            f"BENDR/DN3: {short_trace}. This usually means a nested dependency "
            "such as dn3, yamlinclude, parse, or objgraph is missing from the "
            f"active environment. First sys.path entries: {sys.path[:8]}"
        ) from exc


def _patch_collections_abc_aliases() -> None:
    """Make old DN3/BENDR imports work on Python 3.10+."""

    for name in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Callable"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def _load_pretrained_modules(
    model: Any,
    encoder_path: Path,
    context_path: Path,
    **kwargs: Any,
) -> None:
    loader = model.load_pretrained_modules
    signature = inspect.signature(loader)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    loader(str(encoder_path), str(context_path), **supported)


def _first_int(*values: int | None, name: str) -> int:
    for value in values:
        if value is not None:
            return int(value)
    raise ValueError(f"BENDR adapter requires {name}.")


def _required_existing_path(name: str, value: str | None) -> Path:
    if value is None or value == "":
        raise RuntimeError(
            f"BENDR pretrained mode requires {name}. Set BENDR_ENCODER_WEIGHTS "
            "and BENDR_CONTEXT_WEIGHTS, or set random_init: true for a wiring smoke test."
        )
    path = Path(os.path.expandvars(value)).expanduser()
    if not path.exists():
        raise RuntimeError(f"BENDR {name} path does not exist: {path}")
    return path.resolve()
