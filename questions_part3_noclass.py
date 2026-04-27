"""Part 3 of the assignment."""
from __future__ import annotations

from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import new_utils as nu
import utils as u
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None

normalize = u.Normalization.APPLY_NORMALIZATION
seed = 42
frac_train = 0.8
n_splits = 5


def _ensure_data() -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    global x_train, y_train, x_test, y_test
    if x_train is None or y_train is None or x_test is None or y_test is None:
        x_train, y_train, x_test, y_test = u.prepare_data()
    x_train = nu.scale(x_train)
    x_test = nu.scale(x_test)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    return x_train, y_train, x_test, y_test


def analyze_class_distribution(y: NDArray[np.int32]) -> dict[str, Any]:
    uniq, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(uniq, counts, strict=True))
    num_classes = len(class_counts)
    print(f"{uniq=}")
    print(f"{counts=}")
    print(f"{class_counts=}")
    print(f"{num_classes=}")
    print(f"{np.sum(counts)=}")
    return {"class_counts": class_counts, "num_classes": num_classes}


def part_3a(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    global x_train, y_train, x_test, y_test
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    x_train, y_train, x_test, y_test = _ensure_data()
    clf = LogisticRegression(max_iter=500, random_state=seed, solver="saga", tol=0.001, C=0.5, penalty="l1")
    clf.fit(x_train, y_train)
    train_scores = clf.predict_proba(x_train)
    test_scores = clf.predict_proba(x_test)
    labels = np.arange(10)
    top_k: dict[int, list[float]] = {}
    for k in [1, 2, 3, 4, 5]:
        top_k[k] = [
            float(top_k_accuracy_score(y_train, train_scores, k=k, labels=labels)),
            float(top_k_accuracy_score(y_test, test_scores, k=k, labels=labels)),
        ]

    ks = list(top_k.keys())
    train_vals = [top_k[k][0] for k in ks]
    test_vals = [top_k[k][1] for k in ks]
    plt.figure()
    plt.plot(ks, train_vals, marker="o", label="train")
    plt.plot(ks, test_vals, marker="o", label="test")
    plt.xlabel("k")
    plt.ylabel("top-k accuracy")
    plt.title("Top-k accuracy on MNIST")
    plt.legend()
    plt.tight_layout()
    plt.savefig("part_3a_top_k_accuracy.png")
    plt.close()

    return {
        "top_k_accuracy": top_k,
        "top_k_accuracy_explain": (
            "Accuracy increases quickly as k grows because the correct digit only needs to appear among the top k classes. "
            "For MNIST, top-k is less useful than top-1 for the main task because the desired output is one exact digit."
        ),
    }


def part_3b(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    global x_train, y_train, x_test, y_test
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    x_train, y_train, x_test, y_test = _ensure_data()
    x_train, y_train, x_test, y_test = u.prepare_and_filter_data(x_train, y_train, x_test, y_test, frac_to_remove=0.90)
    x_train = nu.scale(x_train)
    x_test = nu.scale(x_test)

    answers: dict[str, Any] = {}
    answers["number_of_samples"] = {
        "length_x_train": int(len(x_train)),
        "length_x_test": int(len(x_test)),
        "length_y_train": int(len(y_train)),
        "length_y_test": int(len(y_test)),
    }
    answers["data_bounds"] = {
        "max_x_train": float(np.max(x_train)) if len(x_train) else 0.0,
        "max_x_test": float(np.max(x_test)) if len(x_test) else 0.0,
    }
    answers["class_counts"] = {
        "num_0s_train": int(np.sum(y_train == 0)),
        "num_1s_train": int(np.sum(y_train == 1)),
        "num_0s_test": int(np.sum(y_test == 0)),
        "num_1s_test": int(np.sum(y_test == 1)),
    }
    answers["x_train"] = x_train
    answers["y_train"] = y_train
    answers["x_test"] = x_test
    answers["y_test"] = y_test
    print(answers["class_counts"])
    return answers


def _svc_metrics(clf: SVC, x: NDArray[np.floating], y: NDArray[np.int32], splits: int = n_splits) -> tuple[StratifiedKFold, dict[str, Any], dict[str, float], dict[str, float]]:
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    scores = cross_validate(
        clf,
        x,
        y,
        cv=cv,
        scoring=["accuracy", "recall", "precision", "f1"],
        return_train_score=True,
    )
    mean_metrics = {
        "mean_F1": float(np.mean(scores["test_f1"])),
        "mean_recall": float(np.mean(scores["test_recall"])),
        "mean_accuracy": float(np.mean(scores["test_accuracy"])),
        "mean_precision": float(np.mean(scores["test_precision"])),
    }
    std_metrics = {
        "std_F1": float(np.std(scores["test_f1"])),
        "std_recall": float(np.std(scores["test_recall"])),
        "std_accuracy": float(np.std(scores["test_accuracy"])),
        "std_precision": float(np.std(scores["test_precision"])),
    }
    return cv, scores, mean_metrics, std_metrics


def part_3c(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    global x_train, y_train, x_test, y_test
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    x_train, y_train, x_test, y_test = _ensure_data()
    if not set(np.unique(y_train)).issubset({0, 1}):
        filtered = part_3b(x_train, y_train, x_test, y_test)
        x_train = filtered["x_train"]
        y_train = filtered["y_train"]
        x_test = filtered["x_test"]
        y_test = filtered["y_test"]

    clf = SVC(random_state=seed)
    cv, _, mean_metrics, std_metrics = _svc_metrics(clf, x_train, y_train, splits=2)
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    clf.fit(x_all, y_all)
    conf_mat = confusion_matrix(y_all, clf.predict(x_all), labels=[0, 1])
    higher = mean_metrics["mean_precision"] > mean_metrics["mean_recall"]
    return {
        "cv": cv,
        "clf": clf,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "is_precision_higher_than_recall": bool(higher),
        "is_precision_higher_than_recall_explain": (
            "Precision is higher when the classifier is conservative about predicting the minority class 1, causing fewer false positives. "
            "Recall is higher when it finds more of the minority class but may include more false positives."
        ),
        "confusion_matrix": conf_mat,
    }


def part_3d(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    global x_train, y_train, x_test, y_test
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    x_train, y_train, x_test, y_test = _ensure_data()
    if not set(np.unique(y_train)).issubset({0, 1}):
        filtered = part_3b(x_train, y_train, x_test, y_test)
        x_train = filtered["x_train"]
        y_train = filtered["y_train"]
        x_test = filtered["x_test"]
        y_test = filtered["y_test"]

    classes = np.array([0, 1], dtype=np.int32)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights, strict=True)}
    print(f"class weights: {weight_dict}")
    clf = SVC(random_state=seed, class_weight=weight_dict)
    cv, _, mean_metrics, std_metrics = _svc_metrics(clf, x_train, y_train, splits=5)
    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    clf.fit(x_all, y_all)
    conf_mat = confusion_matrix(y_all, clf.predict(x_all), labels=[0, 1])
    higher = mean_metrics["mean_precision"] > mean_metrics["mean_recall"]
    return {
        "cv": cv,
        "clf": clf,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "is_precision_higher_than_recall": bool(higher),
        "is_precision_higher_than_recall_explain": (
            "With class weights, the minority class receives a larger penalty for mistakes. This often raises recall for class 1, "
            "although precision can decrease because the classifier predicts the minority class more often."
        ),
        "performance_difference_explain": (
            "Class weighting compensates for the removed 9s by penalizing errors on the minority class more heavily."
        ),
        "confusion_matrix": conf_mat,
        "weight_dict": weight_dict,
    }


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = u.prepare_data()
    all_answers = {}
    all_answers["part_3a"] = part_3a()
    all_answers["part_3b"] = part_3b()
    all_answers["part_3c"] = part_3c()
    all_answers["part_3d"] = part_3d()
    u.save_dict("section3.pkl", dct=all_answers)
