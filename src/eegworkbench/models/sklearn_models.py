"""Classic EEG baselines built with sklearn-compatible estimators."""

from __future__ import annotations

from typing import Any


def build_sklearn_model(name: str) -> Any:
    normalized = name.lower()
    if normalized == "riemann_tangent":
        return _build_riemann_tangent()
    if normalized == "csp_lda":
        return _build_csp_lda()
    raise ValueError(f"Unknown sklearn EEG model: {name!r}")


def _build_riemann_tangent() -> Any:
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The riemann_tangent baseline requires pyriemann and scikit-learn."
        ) from exc

    return make_pipeline(
        Covariances(estimator="lwf"),
        TangentSpace(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def _build_csp_lda() -> Any:
    try:
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise RuntimeError("The csp_lda baseline requires mne and scikit-learn.") from exc

    return make_pipeline(
        CSP(n_components=8, reg="ledoit_wolf", log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )

