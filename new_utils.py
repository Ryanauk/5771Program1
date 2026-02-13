"""new_utils.py

Only place you should add helper functions for the assignment.

IMPORTANT (per assignment instructions):
- Only use numpy (and optionally sklearn, utils) inside your solutions.
- Do NOT add any non-allowed imports (no pandas, matplotlib, etc.).
"""

import numpy as np
from numpy.typing import NDArray


def scale_data(X: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return a *new* array that is float and scaled to [0, 1]."""
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X
    maxv = float(np.max(X))
    if maxv > 1.0:
        X = X / maxv
    X = np.clip(X, 0.0, 1.0)
    return X


def scores_from_cross_validate(scores: dict) -> dict[str, float]:
    """Convert sklearn.model_selection.cross_validate output to required mean/std dict."""
    test = np.asarray(scores.get("test_score", []), dtype=float)
    fit = np.asarray(scores.get("fit_time", []), dtype=float)
    return {
        "mean_fit_time": float(np.mean(fit)) if fit.size else 0.0,
        "std_fit_time": float(np.std(fit)) if fit.size else 0.0,
        "mean_accuracy": float(np.mean(test)) if test.size else 0.0,
        "std_accuracy": float(np.std(test)) if test.size else 0.0,
    }
