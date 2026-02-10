"""part_3_template_solution.py

run_part_3.py imports: `from part_3_template_solution import Section3`
So this file MUST define class `Section3` with methods partA, partB, partC, partD.

Part 3 tasks:
A) Top-k accuracy for the same classifier as end of Part2.B (LogReg 300 iters).
B) Build imbalanced 7/9 dataset, relabel 7->0, 9->1, remove 90% of 9s.
C) SVC with Stratified CV; report accuracy/precision/recall/F1; confusion matrix.
D) Same as C but with class-weighted loss.

This code is heavily commented and prints what the prompt asks for.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    top_k_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

import utils as u
from new_utils import scale


class Section3:
    def __init__(self, seed: int = 42, frac_train: float = 0.2):
        self.seed = seed
        self.frac_train = frac_train

    # -------------------------------
    # Part A
    # -------------------------------
    def partA(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ):
        """Top-k accuracy (k=1..5) for multiclass logistic regression."""

        # Ensure scaling requirements
        X = scale(X)
        Xtest = scale(Xtest)
        y = np.asarray(y).astype(np.int32)
        ytest = np.asarray(ytest).astype(np.int32)

        # Same model as end of Part 2.B: LogisticRegression with 300 iterations
        clf = LogisticRegression(max_iter=300, solver="lbfgs", multi_class="auto", random_state=self.seed)

        # Fit on training data
        clf.fit(X, y)

        # Predict probabilities needed for top-k
        prob_train = clf.predict_proba(X)
        prob_test = clf.predict_proba(Xtest)

        ks = [1, 2, 3, 4, 5]
        train_scores = []
        test_scores = []

        for k in ks:
            train_scores.append(float(top_k_accuracy_score(y, prob_train, k=k)))
            test_scores.append(float(top_k_accuracy_score(ytest, prob_test, k=k)))

        # Plot k vs accuracy
        plt.figure()
        plt.plot(ks, train_scores, label="Train")
        plt.plot(ks, test_scores, label="Test")
        plt.xlabel("k")
        plt.ylabel("Top-k accuracy")
        plt.title("Top-k Accuracy vs k")
        plt.legend()
        plt.show()

        answer = {
            "ks": ks,
            "train_topk": train_scores,
            "test_topk": test_scores,
            "comment": (
                "Top-k accuracy increases with k because the model gets 'k chances'. "
                "For MNIST, it can be useful if you plan to show multiple candidate digits "
                "to a user or do a second-stage re-ranker."
            ),
        }
        return answer, X, y, Xtest, ytest

    # -------------------------------
    # Part B
    # -------------------------------
    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ):
        """Create imbalanced binary dataset from 7/9 and relabel to 0/1."""

        # Filter 7/9 from the provided train and test sets
        X_bin, y_bin = u.filter_out_7_9s(X, y)
        Xtest_bin, ytest_bin = u.filter_out_7_9s(Xtest, ytest)

        # Scale and relabel
        X_bin = scale(X_bin)
        Xtest_bin = scale(Xtest_bin)

        y_bin = (np.asarray(y_bin).astype(np.int32) == 9).astype(np.int32)  # 7->0, 9->1
        ytest_bin = (np.asarray(ytest_bin).astype(np.int32) == 9).astype(np.int32)

        # Remove 90% of the 1-class (which corresponds to digit 9)
        ones_idx = np.where(y_bin == 1)[0]
        n_remove = int(0.90 * len(ones_idx))

        # Deterministic removal: use a fixed seed shuffle
        rng = np.random.default_rng(self.seed)
        rng.shuffle(ones_idx)
        remove_idx = ones_idx[:n_remove]

        keep = np.ones(len(y_bin), dtype=bool)
        keep[remove_idx] = False

        X_imb = X_bin[keep]
        y_imb = y_bin[keep]

        print("Imbalanced train size:", len(y_imb))
        print("Class 0 count:", int(np.sum(y_imb == 0)))
        print("Class 1 count:", int(np.sum(y_imb == 1)))

        answer = {
            "train_size": int(len(y_imb)),
            "class0": int(np.sum(y_imb == 0)),
            "class1": int(np.sum(y_imb == 1)),
            "note": "Removed 90% of the digit-9 class and relabeled 7->0, 9->1.",
        }
        return answer, X_imb, y_imb, Xtest_bin, ytest_bin

    # -------------------------------
    # Part C
    # -------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """SVC + Stratified CV, report accuracy/precision/recall/F1 and plot confusion matrix."""

        # StratifiedKFold keeps class proportions similar in each fold (important for imbalance)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        clf = SVC()  # required: use SVC, not LinearSVC

        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score),
            "f1": make_scorer(f1_score),
        }

        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        # Compute mean/std for each metric
        out = {}
        for k in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
            out[f"mean_{k}"] = float(np.mean(scores[k]))
            out[f"std_{k}"] = float(np.std(scores[k]))
            print(k, "mean=", out[f"mean_{k}"], "std=", out[f"std_{k}"])

        # Precision vs recall comment
        prec = out["mean_test_precision"]
        rec = out["mean_test_recall"]
        pr_comment = (
            "Precision is higher" if prec > rec else "Recall is higher" if rec > prec else "Precision and recall are equal"
        )

        # Fit on all training data and plot confusion matrix on test data
        clf.fit(X, y)
        preds = clf.predict(Xtest)
        cm = confusion_matrix(ytest, preds)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix: Unweighted SVC")
        plt.show()

        out["precision_vs_recall"] = pr_comment
        out["explanation"] = (
            "With heavy imbalance (few 1s), a model may predict mostly 0s. "
            "That can keep accuracy high but often hurts recall for the minority class."
        )
        return out

    # -------------------------------
    # Part D
    # -------------------------------
    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """Weighted SVC: compute class weights and repeat Part C metrics."""

        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

        print("Computed class weights:", class_weight)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        clf = SVC(class_weight=class_weight)

        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score),
            "f1": make_scorer(f1_score),
        }

        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        out = {"class_weights": class_weight}
        for k in ["test_accuracy", "test_precision", "test_recall", "test_f1"]:
            out[f"mean_{k}"] = float(np.mean(scores[k]))
            out[f"std_{k}"] = float(np.std(scores[k]))
            print("WEIGHTED", k, "mean=", out[f"mean_{k}"], "std=", out[f"std_{k}"])

        out["comment"] = (
            "Class weighting usually increases recall (and often F1) for the minority class, "
            "sometimes at a small cost to precision or overall accuracy."
        )
        return out
