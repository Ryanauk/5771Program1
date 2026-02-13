# new_utils.py
# Reusable helper functions for Assignment 1.
# Keep imports minimal (numpy only).

import numpy as np


def scale_data(X):
    """Convert X to float and scale pixel values into [0, 1].

    MNIST pixels are typically 0..255. If max(X) > 1, divide by 255.
    Always clip to [0, 1] to enforce bounds.
    """
    X = np.asarray(X).astype(float)
    if X.size > 0 and float(np.max(X)) > 1.0:
        X = X / 255.0
    return np.clip(X, 0.0, 1.0)


def scale(X):
    """Alias required by the assignment text."""
    return scale_data(X)


def summarize_cv(scores):
    """Summarize sklearn.model_selection.cross_validate output into plain floats."""
    test_score = np.asarray(scores.get("test_score", []), dtype=float)
    fit_time = np.asarray(scores.get("fit_time", []), dtype=float)
    out = {
        "mean_test_score": float(test_score.mean()) if test_score.size else 0.0,
        "std_test_score": float(test_score.std()) if test_score.size else 0.0,
        "mean_fit_time": float(fit_time.mean()) if fit_time.size else 0.0,
        "std_fit_time": float(fit_time.std()) if fit_time.size else 0.0,
    }
    return out


def class_counts(y, n_classes=10):
    """Return numpy array of class counts (length n_classes)."""
    y = np.asarray(y).astype(int)
    counts = np.zeros(int(n_classes), dtype=int)
    uniq, cts = np.unique(y, return_counts=True)
    for u, c in zip(uniq, cts):
        u = int(u)
        if 0 <= u < int(n_classes):
            counts[u] = int(c)
    return counts
