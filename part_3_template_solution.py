# part_3_template_solution.py
# Section 3: Different evaluation metrics, class imbalance, and weighted loss.

import numpy as np
from typing import Any, Dict, Optional

import utils as u
import new_utils as nu

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


class Section3:
    def __init__(self, normalize: bool = True, frac_train: float = 0.2, seed: int = 42):
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    # -------------------------
    # 3A
    # -------------------------
    def partA(self, X, y, Xtest, ytest):
        # Use LogisticRegression(300 iters) like end of Section 2B
        X = nu.scale_data(X)
        Xtest = nu.scale_data(Xtest)
        y = np.asarray(y).astype(int)
        ytest = np.asarray(ytest).astype(int)

        clf = LogisticRegression(max_iter=300, random_state=self.seed, solver='saga', tol=1e-2)
        clf.fit(X, y)

        proba_tr = clf.predict_proba(X)
        proba_te = clf.predict_proba(Xtest)

        labels = np.arange(10)
        top_k_accuracy: Dict[int, list] = {}
        for k in [1, 2, 3, 4, 5]:
            s_tr = float(top_k_accuracy_score(y, proba_tr, k=k, labels=labels))
            s_te = float(top_k_accuracy_score(ytest, proba_te, k=k, labels=labels))
            top_k_accuracy[k] = [s_tr, s_te]

        answer = {
            "top_k_accuracy": top_k_accuracy,
            "comment_rate_accuracy_change": (
                "Top-k accuracy increases with k because the correct label has more chances to appear in the top-k list; "
                "gains typically diminish as k grows."
            ),
            "is_metric_useful_explain": (
                "It can be useful if you care about whether the model is close to correct (e.g., top-3), but MNIST is usually evaluated with top-1 accuracy."
            ),
        }
        return answer, X, y, Xtest, ytest

    # -------------------------
    # 3B
    # -------------------------
    def partB(self, X, y, Xtest, ytest):

        # Filter to 7/9 using provided data
        X, y = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        X = nu.scale_data(X)
        Xtest = nu.scale_data(Xtest)

        y = np.asarray(y).astype(int)
        ytest = np.asarray(ytest).astype(int)

        # Convert 7->0 and 9->1
        y = np.where(y == 7, 0, 1).astype(int)
        ytest = np.where(ytest == 7, 0, 1).astype(int)

        # Remove 90% of the 9s (class 1) from TRAINING set
        idx1 = np.where(y == 1)[0]
        idx0 = np.where(y == 0)[0]

        rng = np.random.RandomState(self.seed)
        keep1 = rng.choice(idx1, size=max(1, int(0.1 * len(idx1))), replace=False)
        keep = np.concatenate([idx0, keep1])
        keep.sort()

        Ximb = X[keep]
        yimb = y[keep]

        answer: Dict[str, Any] = {}
        answer["number_of_samples"] = {"Xtrain": int(len(Ximb)), "ytrain": int(len(yimb)), "Xtest": int(len(Xtest)), "ytest": int(len(ytest))}
        answer["data_bounds"] = {"min_Xtrain": float(np.min(Ximb)), "max_Xtrain": float(np.max(Ximb)), "min_Xtest": float(np.min(Xtest)), "max_Xtest": float(np.max(Xtest))}
        answer["class_counts"] = {"train_0": int(np.sum(yimb == 0)), "train_1": int(np.sum(yimb == 1)), "test_0": int(np.sum(ytest == 0)), "test_1": int(np.sum(ytest == 1))}

        return answer, Ximb, yimb, Xtest, ytest

    def _metric_mean_std(self, cv_out, key):
        arr = np.asarray(cv_out[key], dtype=float)
        return float(arr.mean()), float(arr.std())

    # -------------------------
    # 3C
    # -------------------------
    def partC(self, X, y, Xtest, ytest):
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
        }

        cv_out = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        mean_metrics = {
            "accuracy": self._metric_mean_std(cv_out, "test_accuracy")[0],
            "precision": self._metric_mean_std(cv_out, "test_precision")[0],
            "recall": self._metric_mean_std(cv_out, "test_recall")[0],
            "f1": self._metric_mean_std(cv_out, "test_f1")[0],
        }
        std_metrics = {
            "accuracy": self._metric_mean_std(cv_out, "test_accuracy")[1],
            "precision": self._metric_mean_std(cv_out, "test_precision")[1],
            "recall": self._metric_mean_std(cv_out, "test_recall")[1],
            "f1": self._metric_mean_std(cv_out, "test_f1")[1],
        }

        print("test_accuracy mean=", mean_metrics["accuracy"], "std=", std_metrics["accuracy"])
        print("test_precision mean=", mean_metrics["precision"], "std=", std_metrics["precision"])
        print("test_recall mean=", mean_metrics["recall"], "std=", std_metrics["recall"])
        print("test_f1 mean=", mean_metrics["f1"], "std=", std_metrics["f1"])

        # Fit on full training set and compute confusion matrix on test
        clf.fit(X, y)
        yhat_test = clf.predict(Xtest)
        cm = confusion_matrix(ytest, yhat_test)

        is_prec_higher = bool(mean_metrics["precision"] > mean_metrics["recall"])
        explain = (
            "With strong class imbalance, a model can achieve high precision by predicting the minority class only "
            "when it is very confident (few false positives), but miss many minority examples (more false negatives), "
            "leading to lower recall. That often makes precision > recall."
        )

        return {
            "clf": clf,
            "cv": cv,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "is_precision_higher_than_recall": is_prec_higher,
            "is_precision_higher_than_recall_explain": explain,
            "confusion_matrix": cm,
        }

    # -------------------------
    # 3D
    # -------------------------
    def partD(self, X, y, Xtest, ytest):
        classes = np.unique(y)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        weight_dict = {int(c): float(wi) for c, wi in zip(classes, w)}
        print("Computed class weights:", weight_dict)

        clf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=weight_dict)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
        }

        cv_out = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        mean_metrics = {
            "accuracy": self._metric_mean_std(cv_out, "test_accuracy")[0],
            "precision": self._metric_mean_std(cv_out, "test_precision")[0],
            "recall": self._metric_mean_std(cv_out, "test_recall")[0],
            "f1": self._metric_mean_std(cv_out, "test_f1")[0],
        }
        std_metrics = {
            "accuracy": self._metric_mean_std(cv_out, "test_accuracy")[1],
            "precision": self._metric_mean_std(cv_out, "test_precision")[1],
            "recall": self._metric_mean_std(cv_out, "test_recall")[1],
            "f1": self._metric_mean_std(cv_out, "test_f1")[1],
        }

        print("WEIGHTED test_accuracy mean=", mean_metrics["accuracy"], "std=", std_metrics["accuracy"])
        print("WEIGHTED test_precision mean=", mean_metrics["precision"], "std=", std_metrics["precision"])
        print("WEIGHTED test_recall mean=", mean_metrics["recall"], "std=", std_metrics["recall"])
        print("WEIGHTED test_f1 mean=", mean_metrics["f1"], "std=", std_metrics["f1"])

        clf.fit(X, y)
        yhat_test = clf.predict(Xtest)
        cm = confusion_matrix(ytest, yhat_test)

        return {
            "clf": clf,
            "cv": cv,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "confusion_matrix": cm,
            "weight_dict": weight_dict,
        }
