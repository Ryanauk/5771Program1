"""new_utils.py

This file is meant for *your* helper functions.
The assignment instructions explicitly allow you to add reusable code here.

We implement:
- scale(): force data into float and scale into [0,1]
- print_cv_scores(): convenience printer for cross-validation results
- accuracy_from_confusion(): helper for sanity checks

All functions are heavily commented so you understand what is happening.
"""

import numpy as np


def scale(X):
    """Scale a data matrix into the range [0,1].

    The assignment requires:
    - Every element must be floating point
    - Values must be between 0 and 1

    MNIST pixels are originally integers in [0,255].
    Scaling improves learning stability for most classifiers.

    Parameters
    ----------
    X : np.ndarray
        Raw input matrix.

    Returns
    -------
    X_scaled : np.ndarray
        Float matrix with values in [0,1].
    """

    # Convert to float (required!)
    X = X.astype(float)

    # If max value is > 1, assume raw pixels and divide by 255
    if np.max(X) > 1.0:
        X = X / 255.0

    # Clip just in case numerical issues appear
    X = np.clip(X, 0.0, 1.0)

    return X


def print_cv_scores(scores, label="Model"):
    """Print mean/std of cross validation accuracy scores."""
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"{label} CV Accuracy: mean={mean_score:.4f}, std={std_score:.4f}")
    return mean_score, std_score


def accuracy_from_confusion(cm):
    """Compute accuracy given a confusion matrix."""
    return np.trace(cm) / np.sum(cm)
import numpy as np

def summarize_cv(scores_dict: dict) -> dict:
    """
    Convert sklearn.cross_validate output into mean/std values.
    """
    out = {}
    for k, arr in scores_dict.items():
        arr = np.asarray(arr)
        out[f"mean_{k}"] = float(arr.mean())
        out[f"std_{k}"] = float(arr.std())
    return out
