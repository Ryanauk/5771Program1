"""Questions for part1 of Assignment 1."""
from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

import new_utils as nu
import utils as u
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeClassifier

x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None

seed = 42
frac_train = 0.2
max_iter = 500


def _summary(results: dict[str, NDArray[np.floating]]) -> dict[str, float]:
    return nu.score_summary(results)


def _ensure_binary_data() -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    global x_train, y_train, x_test, y_test
    if x_train is None or y_train is None or x_test is None or y_test is None:
        x_train, y_train, x_test, y_test = u.prepare_data()
    x_train, y_train = u.filter_out_7_9s(x_train, y_train)
    x_test, y_test = u.filter_out_7_9s(x_test, y_test)
    x_train = nu.scale(x_train)
    x_test = nu.scale(x_test)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    return x_train, y_train, x_test, y_test


def part_1a() -> dict[str, Any]:
    answers = {}
    answers["starter_code"] = u.starter_code()
    return answers


def part_1b(
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

    x_train, y_train, x_test, y_test = _ensure_binary_data()

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
    answers["x_train"] = x_train
    answers["y_train"] = y_train
    answers["x_test"] = x_test
    answers["y_test"] = y_test
    print(answers["number_of_samples"])
    print(answers["data_bounds"])
    return answers


def part_1c(x_train_=None, y_train_=None, x_test_=None, y_test_=None) -> dict[str, Any]:
    part_1b(x_train_, y_train_, x_test_, y_test_)
    x, y, _, _ = _ensure_binary_data()
    clf = DecisionTreeClassifier(random_state=seed)
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = _summary(cross_validate(clf, x, y, cv=cv, scoring="accuracy"))
    print(scores)
    return {"clf": clf, "cv": cv, "scores": scores}


def part_1d(x_train_=None, y_train_=None, x_test_=None, y_test_=None) -> dict[str, Any]:
    part_1b(x_train_, y_train_, x_test_, y_test_)
    x, y, _, _ = _ensure_binary_data()
    clf = DecisionTreeClassifier(random_state=seed)
    cv = ShuffleSplit(n_splits=5, random_state=seed)
    scores = _summary(cross_validate(clf, x, y, cv=cv, scoring="accuracy"))
    return {
        "clf": clf,
        "cv": cv,
        "scores": scores,
        "explain_kfold_vs_shuffle_split": (
            "KFold partitions the data into k non-overlapping validation folds, so each sample is validated once. "
            "ShuffleSplit creates random train/validation splits; validation sets can overlap, which is flexible and useful "
            "for repeated random sampling, but the scores can have more randomness and not every point is guaranteed to be "
            "used in validation the same number of times."
        ),
    }


def part_1e(x_train_=None, y_train_=None) -> dict[str, Any]:
    part_1b(x_train_, y_train_, None, None)
    x, y, _, _ = _ensure_binary_data()
    out: dict[int, dict[str, Any]] = {}
    for k in [2, 5, 8, 16]:
        clf = DecisionTreeClassifier(random_state=seed)
        cv = ShuffleSplit(n_splits=k, random_state=seed)
        scores = _summary(cross_validate(clf, x, y, cv=cv, scoring="accuracy"))
        out[k] = {"clf": clf, "cv": cv, **scores}
        print(f"k={k}: mean_accuracy={scores['mean_accuracy']}, std_accuracy={scores['std_accuracy']}")
    return {"scores": out}


def part_1f(x_train_=None, y_train_=None) -> dict[str, Any]:
    part_1b(x_train_, y_train_, None, None)
    x, y, _, _ = _ensure_binary_data()
    cv = ShuffleSplit(n_splits=5, random_state=seed)
    clf_rf = RandomForestClassifier(random_state=seed)
    clf_dt = DecisionTreeClassifier(random_state=seed)
    scores_rf = _summary(cross_validate(clf_rf, x, y, cv=cv, scoring="accuracy"))
    scores_dt = _summary(cross_validate(clf_dt, x, y, cv=cv, scoring="accuracy"))
    return {
        "clf_RF": clf_rf,
        "cv_RF": cv,
        "scores_RF": scores_rf,
        "clf_DT": clf_dt,
        "cv_DT": cv,
        "scores_DT": scores_dt,
        "model_highest_accuracy": "random-forest" if scores_rf["mean_accuracy"] >= scores_dt["mean_accuracy"] else "decision-tree",
        "model_lowest_variance": "random-forest" if scores_rf["std_accuracy"] <= scores_dt["std_accuracy"] else "decision-tree",
        "model_fastest": "random-forest" if scores_rf["mean_fit_time"] <= scores_dt["mean_fit_time"] else "decision-tree",
    }


def part_1g(x_train_=None, y_train_=None, x_test_=None, y_test_=None) -> dict[str, Any]:
    part_1b(x_train_, y_train_, x_test_, y_test_)
    x, y, xt, yt = _ensure_binary_data()
    clf = RandomForestClassifier(random_state=seed)
    param_grid = {
        "criterion": ["entropy"],
        "max_features": [50],
        "n_estimators": [10],
        "max_depth": [10],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=seed),
        param_grid=param_grid,
        cv=3,
        refit=True,
    )
    grid_search.fit(x, y)
    best = grid_search.best_estimator_
    clf.fit(x, y)

    cm = {
        "confusion_matrix_train_orig": confusion_matrix(y, clf.predict(x)),
        "confusion_matrix_train_best": confusion_matrix(y, best.predict(x)),
        "confusion_matrix_test_orig": confusion_matrix(yt, clf.predict(xt)),
        "confusion_matrix_test_best": confusion_matrix(yt, best.predict(xt)),
    }
    acc = {
        "accuracy_orig_full_training": nu.accuracy_from_confusion(cm["confusion_matrix_train_orig"]),
        "accuracy_best_full_training": nu.accuracy_from_confusion(cm["confusion_matrix_train_best"]),
        "accuracy_orig_full_testing": nu.accuracy_from_confusion(cm["confusion_matrix_test_orig"]),
        "accuracy_best_full_testing": nu.accuracy_from_confusion(cm["confusion_matrix_test_best"]),
    }
    prec = {
        "precision_orig_full_training": nu.macro_precision_from_confusion(cm["confusion_matrix_train_orig"]),
        "precision_best_full_training": nu.macro_precision_from_confusion(cm["confusion_matrix_train_best"]),
        "precision_orig_full_testing": nu.macro_precision_from_confusion(cm["confusion_matrix_test_orig"]),
        "precision_best_full_testing": nu.macro_precision_from_confusion(cm["confusion_matrix_test_best"]),
    }
    print(acc)
    return {
        "clf": clf,
        "best_estimator": best,
        "grid_search": grid_search,
        "default_parameters": {
            "criterion": "gini",
            "max_features": 100,
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        },
        "confusion_matrix": cm,
        "accuracy_full_training": acc,
        "precision_full_training": prec,
    }


if __name__ == "__main__":
    answer1_a = part_1a()
    answer1_b = part_1b()
    answer1_c = part_1c()
    answer1_d = part_1d()
    answer1_e = part_1e()
    answer1_f = part_1f()
    answer1_g = part_1g()
    u.save_dict("section1.pkl", {
        "part_1a": answer1_a,
        "part_1b": answer1_b,
        "part_1c": answer1_c,
        "part_1d": answer1_d,
        "part_1e": answer1_e,
        "part_1f": answer1_f,
        "part_1g": answer1_g,
    })
