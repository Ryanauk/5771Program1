"""Utility functions for the homework."""
from __future__ import annotations

import pickle
from enum import Enum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate
from sklearn.tree import DecisionTreeClassifier


class Normalization(Enum):
    APPLY_NORMALIZATION = True
    SKIP_NORMALIZATION = False


class PrintResults(Enum):
    PRINT_RESULTS = True
    SKIP_PRINT_RESULTS = False


def load_mnist_dataset(nb_samples: int | None = None) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    try:
        print("... Is MNIST dataset local?")
        x: NDArray[np.floating] = np.load("mnist_X.npy")
        y: NDArray[np.int32] = np.load("mnist_y.npy", allow_pickle=True)
    except FileNotFoundError:
        print("... download MNIST dataset")
        bunch = datasets.fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        x = np.array(bunch[0], dtype=np.float32)
        y = np.array(bunch[1], dtype=np.int32)

    if nb_samples is not None and nb_samples < x.shape[0]:
        x = x[0:nb_samples, :]
        y = y[0:nb_samples]

    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    np.save("mnist_X.npy", x)
    np.save("mnist_y.npy", y)
    return x, y.astype(np.int32)


def prepare_data(
    num_train: int = 60000,
    num_test: int = 10000,
    normalize: bool = True,
    frac_train: float = 0.8,
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    x, y = load_mnist_dataset()
    x = np.asarray(x, dtype=np.float32)
    if normalize and x.size and float(np.max(x)) > 1.0:
        x = x / float(np.max(x))
    y = y.astype(np.int32)
    x_train, x_test = x[:num_train], x[num_train : num_train + num_test]
    y_train, y_test = y[:num_train], y[num_train : num_train + num_test]
    print("prepare_data before return")
    return x_train, y_train, x_test, y_test


def create_data(n_rows: int, n_features: int, frac_train: float = 0.8):
    rng = np.random.default_rng(42)
    x_full = rng.random((n_rows, n_features))
    y_full = (x_full[:, :5].sum(axis=1) > 2.5).astype(int)
    n_train = int(frac_train * n_rows)
    return x_full[:n_train, :], y_full[:n_train], x_full[n_train:, :], y_full[n_train:]


def filter_out_7_9s(x: NDArray[np.floating], y: NDArray[np.int32]) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Filter x/y so only labels 7 and 9 remain."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.int32)
    mask = np.logical_or(y == 7, y == 9)
    return x[mask], y[mask]


def remove_nines_convert_to_01(x: NDArray[np.floating], y: NDArray[np.int32], frac: float) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Remove frac of the 9s, convert 7 -> 0 and remaining 9 -> 1."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=np.int32)
    frac = min(max(float(frac), 0.0), 1.0)
    idx_7 = np.where(y == 7)[0]
    idx_9 = np.where(y == 9)[0]
    keep_9_count = int(round((1.0 - frac) * len(idx_9)))
    idx_keep = np.sort(np.concatenate((idx_7, idx_9[:keep_9_count])))
    y_new = y[idx_keep].copy()
    y_new[y_new == 7] = 0
    y_new[y_new == 9] = 1
    return x[idx_keep], y_new.astype(np.int32)


def prepare_and_filter_data(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.int32],
    x_test: NDArray[np.floating],
    y_test: NDArray[np.int32],
    frac_to_remove: float = 0.90,
):
    x_train, y_train = filter_out_7_9s(x_train, y_train)
    x_test, y_test = filter_out_7_9s(x_test, y_test)
    x_train, y_train = remove_nines_convert_to_01(x_train, y_train, frac_to_remove)
    x_test, y_test = remove_nines_convert_to_01(x_test, y_test, frac_to_remove)
    return x_train.astype(np.float32), y_train, x_test.astype(np.float32), y_test


def train_simple_classifier_with_cv(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.int32],
    clf: BaseEstimator,
    n_splits: int = 5,
    cv_class: type[KFold] = KFold,
) -> dict[str, NDArray[np.float32]]:
    cv = cv_class(n_splits=n_splits)
    return cross_validate(clf, x_train, y_train, cv=cv)


def print_cv_result_dict(cv_dict: dict[str, NDArray[np.float32]]) -> None:
    for key, array in cv_dict.items():
        print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")


def starter_code() -> int:
    x_train, y_train, x_test, y_test = prepare_data()
    x_train, y_train = filter_out_7_9s(x_train, y_train)
    out_dict = train_simple_classifier_with_cv(x_train, y_train, DecisionTreeClassifier(random_state=42))
    print("running cross validation...")
    print_cv_result_dict(out_dict)
    return 100


def save_dict(filenm: str, dct: dict) -> None:
    with Path(filenm).open("wb") as file:
        pickle.dump(dct, file)


def load_dict(filenm: str) -> dict:
    with Path(filenm).open("rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    starter_code()
