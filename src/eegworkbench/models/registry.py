"""Model registry for Braindecode, sklearn, and external model adapters."""

from __future__ import annotations

import inspect
from typing import Any

from eegworkbench.config import ModelConfig
from eegworkbench.models.external import build_external_torch_model


BRAIND_DECODE_ALIASES = {
    "eegnet": ("EEGNet", "EEGNetv4"),
    "eegnetv4": ("EEGNetv4", "EEGNet"),
    "shallow": ("ShallowFBCSPNet",),
    "shallowfbcspnet": ("ShallowFBCSPNet",),
    "deep4": ("Deep4Net",),
    "deep4net": ("Deep4Net",),
    "eegconformer": ("EEGConformer",),
}

SKLEARN_MODELS = {
    "riemann_tangent",
    "csp_lda",
}

EXTERNAL_MODELS = {
    "bendr",
    "bendr_external",
    "labram",
    "labram_external",
    "external_torch",
}


def list_model_names() -> list[str]:
    return sorted([*BRAIND_DECODE_ALIASES, *SKLEARN_MODELS, *EXTERNAL_MODELS])


def model_framework(name: str) -> str:
    normalized = name.lower()
    if normalized in SKLEARN_MODELS:
        return "sklearn"
    return "torch"


def build_model(
    config: ModelConfig,
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
) -> Any:
    """Build a model from the configured registry."""

    name = config.name.lower()
    if name in EXTERNAL_MODELS:
        return build_external_torch_model(
            config.external,
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
    if name in BRAIND_DECODE_ALIASES:
        return _build_braindecode_model(
            name,
            config.kwargs,
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
    raise ValueError(
        f"Unknown model {config.name!r}. Available models: {', '.join(list_model_names())}"
    )


def _build_braindecode_model(
    name: str,
    model_kwargs: dict[str, Any],
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
) -> Any:
    try:
        import braindecode.models as models
    except ImportError as exc:
        raise RuntimeError(
            "Braindecode models require the braindecode package. "
            "Install/activate the EEG environment before training neural models."
        ) from exc

    for class_name in BRAIND_DECODE_ALIASES[name]:
        model_cls = getattr(models, class_name, None)
        if model_cls is None:
            continue
        return _instantiate_with_shape_synonyms(
            model_cls,
            model_kwargs,
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
        )
    raise RuntimeError(f"Braindecode is installed but does not expose a {name!r} model.")


def _instantiate_with_shape_synonyms(
    model_cls: type[Any],
    model_kwargs: dict[str, Any],
    *,
    n_chans: int,
    n_outputs: int,
    n_times: int,
) -> Any:
    signature = inspect.signature(model_cls)
    kwargs = dict(model_kwargs)
    shape_synonyms = {
        "n_chans": n_chans,
        "in_chans": n_chans,
        "n_outputs": n_outputs,
        "n_classes": n_outputs,
        "n_times": n_times,
        "input_window_samples": n_times,
    }
    for key, value in shape_synonyms.items():
        if key in signature.parameters and key not in kwargs:
            kwargs[key] = value
    if "final_conv_length" in signature.parameters and "final_conv_length" not in kwargs:
        kwargs["final_conv_length"] = "auto"
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return model_cls(**accepted)

