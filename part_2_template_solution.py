"""part_2_template_solution.py

run_part_2.py imports: `from part_2_template_solution import Section2`
So this file MUST define class `Section2` with methods partA and partB.

Part 2 tasks:
A) Load full MNIST (0-9), ensure scaling, print class counts.
B) Repeat Part1-style CV for multiclass; use LogisticRegression with 300 iters.
   Run experiments for multiple train/test sizes.

We keep everything deterministic using self.seed where applicable.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate
from sklearn.metrics import accuracy_score

import utils as u
from new_utils import scale, summarize_cv


class Section2:
    def __init__(self, seed: int = 42, frac_train: float = 0.2):
        self.seed = seed
        self.frac_train = frac_train

    def partA(self):
        """Load full MNIST, scale, and print class counts for train/test."""
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()

        # Enforce float + [0,1]
        Xtrain = scale(Xtrain)
        Xtest = scale(Xtest)

        ytrain = np.asarray(ytrain).astype(np.int32)
        ytest = np.asarray(ytest).astype(np.int32)

        # Count classes
        classes_train, counts_train = np.unique(ytrain, return_counts=True)
        classes_test, counts_test = np.unique(ytest, return_counts=True)

        train_counts = {int(c): int(n) for c, n in zip(classes_train, counts_train)}
        test_counts = {int(c): int(n) for c, n in zip(classes_test, counts_test)}

        print("Train class counts:", train_counts)
        print("Test class counts:", test_counts)
        print("Num classes train:", len(classes_train))
        print("Num classes test:", len(classes_test))

        answer = {
            "train_class_counts": train_counts,
            "test_class_counts": test_counts,
            "num_classes_train": int(len(classes_train)),
            "num_classes_test": int(len(classes_test)),
        }

        return answer, Xtrain, ytrain, Xtest, ytest

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list,
        ntest_list: list,
    ) -> dict[str, Any]:
        """Run CV + size experiments for multiclass classification."""

        # Models requested by run_part_2 prompt:
        dt = DecisionTreeClassifier(random_state=self.seed)
        lr = LogisticRegression(max_iter=300, solver="lbfgs", random_state=self.seed, multi_class="auto")

        # CV strategies
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)

        # --- Baseline CV on the FULL provided X/y
        dt_kf = summarize_cv(cross_validate(dt, X, y, cv=kf))
        dt_ss = summarize_cv(cross_validate(dt, X, y, cv=ss))
        lr_ss = summarize_cv(cross_validate(lr, X, y, cv=ss))

        print("DecisionTree (KFold) mean acc:", dt_kf["mean_test_score"])
        print("DecisionTree (ShuffleSplit) mean acc:", dt_ss["mean_test_score"])
        print("LogReg 300 iters (ShuffleSplit) mean acc:", lr_ss["mean_test_score"])

        # --- Size experiments
        size_results = []
        for ntrain, ntest in zip(ntrain_list, ntest_list):
            # Use first ntrain samples from provided training pool
            Xtr = X[:ntrain]
            ytr = y[:ntrain]

            # Use first ntest samples from provided test pool
            Xte = Xtest[:ntest]
            yte = ytest[:ntest]

            # Fit logistic regression
            lr.fit(Xtr, ytr)

            train_acc = accuracy_score(ytr, lr.predict(Xtr))
            test_acc = accuracy_score(yte, lr.predict(Xte))

            print(f"ntrain={ntrain}, ntest={ntest}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

            size_results.append({
                "ntrain": int(ntrain),
                "ntest": int(ntest),
                "train_accuracy": float(train_acc),
                "test_accuracy": float(test_acc),
            })

        return {
            "decision_tree_kfold": dt_kf,
            "decision_tree_shufflesplit": dt_ss,
            "logistic_regression_shufflesplit": {
                **lr_ss,
                "explanation": (
                    "sklearn LogisticRegression supports multiclass using either "
                    "one-vs-rest (OvR) or multinomial (softmax) depending on solver/settings. "
                    "With solver='lbfgs' and multi_class='auto', it will typically use multinomial "
                    "when possible."
                ),
            },
            "size_experiments": size_results,
            "comment": (
                "Training accuracy is usually higher than test accuracy. As ntrain increases, "
                "test accuracy typically improves (up to diminishing returns)."
            ),
        }
