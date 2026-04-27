"""Part 2 of Assignment 1: Multi-class classification."""
from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

import new_utils as nu
import utils as u
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit, cross_validate
from sklearn.tree import DecisionTreeClassifier

x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None
ntrain_list = None

normalize = u.Normalization.APPLY_NORMALIZATION
seed = 42
frac_train = 0.8


def _summary(results: dict[str, NDArray[np.floating]]) -> dict[str, float]:
    return nu.score_summary(results)


def _ensure_multiclass_data() -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    global x_train, y_train, x_test, y_test
    if x_train is None or y_train is None or x_test is None or y_test is None:
        x_train, y_train, x_test, y_test = u.prepare_data()
    x_train = nu.scale(x_train)
    x_test = nu.scale(x_test)
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test = np.asarray(y_test, dtype=np.int32)
    return x_train, y_train, x_test, y_test


def part_2a() -> dict[str, Any]:
    global x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = u.prepare_data()
    x_train, y_train, x_test, y_test = _ensure_multiclass_data()

    answers: dict[str, Any] = {}
    answers["nb_classes"] = {
        "nb_classes_train": int(len(np.unique(y_train))),
        "nb_classes_test": int(len(np.unique(y_test))),
    }
    answers["class_count"] = {
        "class_count_train": nu.class_count_array(y_train, minlength=10),
        "class_count_test": nu.class_count_array(y_test, minlength=10),
    }
    answers["nb_samples_data"] = {
        "nb_samples_x_train": int(len(x_train)),
        "nb_samples_x_test": int(len(x_test)),
        "nb_samples_y_train": int(len(y_train)),
        "nb_samples_y_test": int(len(y_test)),
    }
    answers["max_data"] = {
        "max_x_train": float(np.max(x_train)) if len(x_train) else 0.0,
        "max_x_test": float(np.max(x_test)) if len(x_test) else 0.0,
    }
    print(answers["nb_classes"])
    print(answers["class_count"])
    return answers


def part_2b(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
    ntrain_list_: list[int] | None = None,
) -> dict[Any, Any]:
    global ntrain_list, x_train, y_train, x_test, y_test
    if ntrain_list_ is not None:
        ntrain_list = ntrain_list_
    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    x_train, y_train, x_test, y_test = _ensure_multiclass_data()
    if ntrain_list is None:
        ntrain_list = [1000, 5000, 10000]

    ntrain_use = min(max(ntrain_list), len(x_train))
    ntest_use = min(2000, len(x_test))
    xtr = x_train[:ntrain_use]
    ytr = y_train[:ntrain_use]
    xte = x_test[:ntest_use]
    yte = y_test[:ntest_use]

    clf_1c = DecisionTreeClassifier(random_state=seed)
    cv_1c = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores_1c = _summary(cross_validate(clf_1c, xtr, ytr, cv=cv_1c, scoring="accuracy"))

    clf_1d = DecisionTreeClassifier(random_state=seed)
    cv_1d = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    scores_1d = _summary(cross_validate(clf_1d, xtr, ytr, cv=cv_1d, scoring="accuracy"))

    clf_1f = LogisticRegression(max_iter=300, random_state=seed, n_jobs=-1)
    cv_1f = ShuffleSplit(n_splits=5, test_size=0.2, random_state=seed)
    scores_lr = cross_validate(clf_1f, xtr, ytr, cv=cv_1f, scoring="accuracy")
    mean_cv_accuracy_1f = float(np.mean(scores_lr["test_score"]))
    clf_1f.fit(xtr, ytr)
    pred_train = clf_1f.predict(xtr)
    pred_test = clf_1f.predict(xte)

    # Small grid search for the requested 1g-like outputs on multiclass data.
    rf_orig = RandomForestClassifier(random_state=seed, n_estimators=50)
    rf_orig.fit(xtr, ytr)
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=seed, n_estimators=50),
        param_grid={"max_depth": [None, 20], "min_samples_leaf": [1, 2]},
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid_search.fit(xtr, ytr)
    best_estimator = grid_search.best_estimator_

    cm_train_orig = confusion_matrix(ytr, rf_orig.predict(xtr), labels=np.arange(10))
    cm_train_best = confusion_matrix(ytr, best_estimator.predict(xtr), labels=np.arange(10))
    cm_test_orig = confusion_matrix(yte, rf_orig.predict(xte), labels=np.arange(10))
    cm_test_best = confusion_matrix(yte, best_estimator.predict(xte), labels=np.arange(10))
    cm_1g = {
        "confusion_matrix_train_orig": cm_train_orig,
        "confusion_matrix_train_best": cm_train_best,
        "confusion_matrix_test_orig": cm_test_orig,
        "confusion_matrix_test_best": cm_test_best,
    }
    accuracy_1g = {
        "accuracy_orig_full_training": nu.accuracy_from_confusion(cm_train_orig),
        "accuracy_best_full_training": nu.accuracy_from_confusion(cm_train_best),
        "accuracy_orig_full_testing": nu.accuracy_from_confusion(cm_test_orig),
        "accuracy_best_full_testing": nu.accuracy_from_confusion(cm_test_best),
    }
    precision_1g = {
        "precision_orig_full_training": nu.macro_precision_from_confusion(cm_train_orig),
        "precision_best_full_training": nu.macro_precision_from_confusion(cm_train_best),
        "precision_orig_full_testing": nu.macro_precision_from_confusion(cm_test_orig),
        "precision_best_full_testing": nu.macro_precision_from_confusion(cm_test_best),
    }

    experiment_by_size: dict[int, dict[str, float]] = {}
    for n in ntrain_list:
        n = min(int(n), len(x_train) - 1)
        t = min(2000, len(x_train) - n, len(x_test))
        xtr_n = x_train[:n]
        ytr_n = y_train[:n]
        xte_n = x_train[n : n + t] if t > 0 else x_test[:ntest_use]
        yte_n = y_train[n : n + t] if t > 0 else y_test[:ntest_use]
        clf_n = LogisticRegression(max_iter=300, random_state=seed, n_jobs=-1)
        clf_n.fit(xtr_n, ytr_n)
        experiment_by_size[int(n)] = {
            "train_accuracy": float(clf_n.score(xtr_n, ytr_n)),
            "test_accuracy": float(clf_n.score(xte_n, yte_n)),
        }

    answers: dict[str, Any] = {}
    answers["scores_1c"] = scores_1c
    answers["clf_1c"] = clf_1c
    answers["cv_1c"] = cv_1c
    answers["scores_1d"] = scores_1d
    answers["clf_1d"] = clf_1d
    answers["cv_1d"] = cv_1d
    answers["accuracy_train_1f"] = float(accuracy_score(ytr, pred_train))
    answers["accuracy_test_1f"] = float(accuracy_score(yte, pred_test))
    answers["mean_cv_accuracy_1f"] = mean_cv_accuracy_1f
    answers["clf_1f"] = clf_1f
    answers["cv_1f"] = cv_1f
    answers["confusion_matrix_1f"] = {
        "conf_mat_train_1f": confusion_matrix(ytr, pred_train, labels=np.arange(10)),
        "conf_mat_test_1f": confusion_matrix(yte, pred_test, labels=np.arange(10)),
    }
    answers["confusion_matrix_1g"] = cm_1g
    answers["accuracy_1g"] = accuracy_1g
    answers["precision_1g"] = precision_1g
    answers["clf_1g"] = rf_orig
    answers["default_parameters_1g"] = rf_orig.get_params()
    answers["best_estimator_1g"] = best_estimator
    answers["grid_search_1g"] = grid_search
    answers["class_count_1g"] = {
        "class_count_train": nu.class_count_array(ytr, minlength=10),
        "class_count_test": nu.class_count_array(yte, minlength=10),
    }
    answers["hard_to_distinguish_pairs"] = nu.hard_pairs_from_confusion(answers["confusion_matrix_1f"]["conf_mat_test_1f"], 5)
    answers["explain_multiclass_logistic_regression"] = (
        "Scikit-learn LogisticRegression handles multiclass MNIST directly with a multinomial/softmax objective when possible. "
        "Instead of training one binary model per digit pair, the model learns weights for all digit classes and predicts the "
        "class with the highest probability."
    )
    answers["comment_results"] = (
        "Training accuracy is usually at least as high as testing accuracy because the model is optimized on the training set. "
        "As ntrain increases, the test score generally improves or stabilizes because the classifier has more examples of each digit."
    )
    answers["scores_as_function_of_ntrain"] = experiment_by_size
    print({"train": answers["accuracy_train_1f"], "test": answers["accuracy_test_1f"]})
    return answers


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = u.prepare_data()
    all_answers = {"part_2a": part_2a(), "part_2b": part_2b(ntrain_list_=[1000, 5000, 10000])}
    u.save_dict("section2.pkl", dct=all_answers)
