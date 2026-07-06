"""Small preprocessing helpers for trial tensors."""

from __future__ import annotations

import numpy as np


def standardize_trials(X: np.ndarray, mode: str = "trial", eps: float = 1e-6) -> np.ndarray:
    """Standardize EEG tensors shaped (trials, channels, time)."""

    mode = (mode or "none").lower()
    X = np.asarray(X, dtype="float32")
    if mode in {"none", "off", "false"}:
        return X
    if mode == "trial":
        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True)
        return (X - mean) / np.maximum(std, eps)
    if mode == "global":
        mean = X.mean(axis=(0, 2), keepdims=True)
        std = X.std(axis=(0, 2), keepdims=True)
        return (X - mean) / np.maximum(std, eps)
    raise ValueError(f"Unknown standardization mode: {mode!r}")

