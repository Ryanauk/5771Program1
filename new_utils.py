"""
Use to create your own functions for reuse across the assignment.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray


def scale(x: NDArray[np.floating]) -> NDArray[np.float32]:
    """Return x as floating point values scaled to the interval [0, 1]."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val <= 1.0 and min_val >= 0.0:
        return arr.astype(np.float32, copy=False)
    if max_val == min_val:
        return np.zeros_like(arr, dtype=np.float32)
    if min_val >= 0.0:
        return (arr / max_val).astype(np.float32)
    return ((arr - min_val) / (max_val - min_val)).astype(np.float32)


def score_summary(cv_results: dict[str, NDArray[np.floating]]) -> dict[str, np.float64]:
    """Convert sklearn cross_validate output to the assignment score dictionary."""
    return {
        "mean_accuracy": np.float64(np.mean(cv_results["test_score"])),
        "std_accuracy": np.float64(np.std(cv_results["test_score"])),
        "mean_fit_time": np.float64(np.mean(cv_results["fit_time"])),
        "std_fit_time": np.float64(np.std(cv_results["fit_time"])),
    }


def accuracy_from_confusion(cm: NDArray[np.integer]) -> float:
    """Compute accuracy from a confusion matrix."""
    total = float(np.sum(cm))
    return 0.0 if total == 0.0 else float(np.trace(cm) / total)


def macro_precision_from_confusion(cm: NDArray[np.integer]) -> float:
    """Compute macro precision from a confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    denom = np.sum(cm, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.divide(np.diag(cm), denom, out=np.zeros_like(denom), where=denom != 0)
    return float(np.mean(values)) if values.size else 0.0


def class_count_array(y: NDArray[np.integer], minlength: int | None = None) -> NDArray[np.int32]:
    """Return class counts as an array indexed by class label."""
    y = np.asarray(y, dtype=np.int32)
    if y.size == 0:
        return np.zeros(0 if minlength is None else minlength, dtype=np.int32)
    if minlength is None:
        minlength = int(np.max(y)) + 1
    return np.bincount(y, minlength=minlength).astype(np.int32)


def hard_pairs_from_confusion(cm: NDArray[np.integer], n_pairs: int = 5) -> set[tuple[int, int]]:
    """Find digit pairs with the largest off-diagonal confusion counts."""
    cm = np.asarray(cm)
    pairs: list[tuple[int, int, int]] = []
    for i in range(cm.shape[0]):
        for j in range(i + 1, cm.shape[1]):
            mistakes = int(cm[i, j] + cm[j, i])
            pairs.append((mistakes, i, j))
    pairs.sort(reverse=True)
    return {(i, j) for mistakes, i, j in pairs[:n_pairs]}
