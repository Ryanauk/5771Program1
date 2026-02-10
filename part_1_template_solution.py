"""part_1_template_solution.py

IMPORTANT:
- run_part_1.py imports: `from part_1_template_solution import Section1`
- Therefore, this file MUST define a class named `Section1`
  with methods partA ... partG.

This implementation follows the prompts inside run_part_1.py and assignment1.md:
- Part A: environment check via utils.starter_code()
- Part B: load MNIST, filter to digits 7 and 9, ensure float + [0,1] scaling
- Part C: DecisionTree + 5-fold KFold CV (print mean/std of accuracy + fit_time)
- Part D: DecisionTree + ShuffleSplit CV (print mean/std + explain pros/cons)
- Part E: ShuffleSplit for k=2,5,8,16 (report trend)
- Part F: Compare LogisticRegression vs SVC on SAME splits (accuracy/variance/fit_time)
- Part G: Tune SVC (grid search), then train on full train and evaluate on test

Everything is commented so you can follow it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import utils as u
from new_utils import scale, summarize_cv


class Section1:
    """Solutions for Assignment 1, Part 1 (binary 7 vs 9)."""

    def __init__(self, seed: int = 42, frac_train: float = 0.2):
        # The runner passes seed=42 and frac_train=0.2. Keep the seed fixed.
        self.seed = seed
        self.frac_train = frac_train

    # -------------------------------
    # Part A
    # -------------------------------
    def partA(self) -> dict[str, Any]:
        """Run starter code to sanity-check the Python environment."""
        status = u.starter_code()
        answer = {
            "status": int(status),
            "comment": "starter_code returns 0 if everything runs; -1 otherwise.",
        }
        print("Part 1A starter_code status:", status)
        return answer

    # -------------------------------
    # Part B
    # -------------------------------
    def partB(self):
        """Load MNIST, filter to 7/9, enforce float + [0,1] scaling, and print checks."""
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()

        # Filter 7s and 9s separately for train/test
        Xtrain, ytrain = u.filter_out_7_9s(Xtrain, ytrain)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        # Enforce float + [0,1]
        Xtrain = scale(Xtrain)
        Xtest = scale(Xtest)

        # Enforce integer labels
        ytrain = np.asarray(ytrain).astype(np.int32)
        ytest = np.asarray(ytest).astype(np.int32)

        # Required prints
        print("(Train) len(X), len(y):", len(Xtrain), len(ytrain))
        print("(Test)  len(X), len(y):", len(Xtest), len(ytest))
        print("Max Xtrain:", float(np.max(Xtrain)))
        print("Max Xtest:", float(np.max(Xtest)))

        answer = {
            "train_size": int(len(Xtrain)),
            "test_size": int(len(Xtest)),
            "max_train": float(np.max(Xtrain)),
            "max_test": float(np.max(Xtest)),
            "dtype_train": str(Xtrain.dtype),
            "dtype_y": str(ytrain.dtype),
        }
        return answer, Xtrain, ytrain, Xtest, ytest

    # -------------------------------
    # Helper used by C/D/E/F
    # -------------------------------
    def _cv_eval(self, X: NDArray[np.floating], y: NDArray[np.int32],
                 clf: BaseEstimator, cv) -> dict[str, Any]:
        """Run cross_validate and return mean/std summary + raw arrays."""
        scores = cross_validate(clf, X, y, cv=cv)  # returns fit_time, score_time, test_score
        summary = summarize_cv(scores)

        # Print the key quantities the assignment asks for
        print(f"{clf.__class__.__name__} mean_test_score={summary['mean_test_score']:.4f} std_test_score={summary['std_test_score']:.4f}")
        print(f"{clf.__class__.__name__} mean_fit_time={summary['mean_fit_time']:.4f} std_fit_time={summary['std_fit_time']:.4f}")

        return {
            "summary": summary,
            "raw": {k: np.asarray(v).tolist() for k, v in scores.items()},
        }

    # -------------------------------
    # Part C
    # -------------------------------
    def partC(self, X: NDArray[np.floating], y: NDArray[np.int32]) -> dict[str, Any]:
        """Decision Tree with 5-fold KFold CV."""
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        return self._cv_eval(X, y, clf, cv)

    # -------------------------------
    # Part D
    # -------------------------------
    def partD(self, X: NDArray[np.floating], y: NDArray[np.int32]) -> dict[str, Any]:
        """Decision Tree with ShuffleSplit CV, plus pros/cons explanation."""
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)

        out = self._cv_eval(X, y, clf, cv)
        out["explanation"] = (
            "KFold uses non-overlapping folds so every sample is used exactly once as validation. "
            "ShuffleSplit randomly resamples train/validation splits; it can be faster to configure "
            "and works well with large datasets, but validation sets can overlap and some points "
            "may appear in validation multiple times or not at all."
        )
        return out

    # -------------------------------
    # Part E
    # -------------------------------
    def partE(self, X: NDArray[np.floating], y: NDArray[np.int32]) -> dict[str, Any]:
        """Repeat ShuffleSplit for k=2,5,8,16 (do NOT print training time per prompt)."""
        clf = DecisionTreeClassifier(random_state=self.seed)
        ks = [2, 5, 8, 16]

        results = {}
        for k in ks:
            cv = ShuffleSplit(n_splits=k, test_size=0.2, random_state=self.seed)
            scores = cross_validate(clf, X, y, cv=cv)

            # Only print accuracy stats (test_score), not training time
            mean_acc = float(np.mean(scores["test_score"]))
            std_acc = float(np.std(scores["test_score"]))
            print(f"ShuffleSplit k={k}: mean_acc={mean_acc:.4f}, std_acc={std_acc:.4f}")

            results[str(k)] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "raw_test_scores": np.asarray(scores["test_score"]).tolist(),
            }

        # Comment on trends: as k increases, mean tends to stabilize, std often decreases a bit
        results["comment"] = (
            "As k (number of random splits) increases, the estimate of mean accuracy typically "
            "stabilizes, and the standard deviation of the estimate often decreases because you're "
            "averaging over more splits."
        )
        return results

    # -------------------------------
    # Part F
    # -------------------------------
    def partF(self, X: NDArray[np.floating], y: NDArray[np.int32]) -> dict[str, Any]:
        """Compare Logistic Regression vs SVC on the SAME ShuffleSplit splits."""

        # SAME cross-validator instance ensures the same splits for both models.
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)

        lr = LogisticRegression(max_iter=300, random_state=self.seed)
        svm = SVC()  # default kernel='rbf'

        lr_scores = cross_validate(lr, X, y, cv=cv)
        svm_scores = cross_validate(svm, X, y, cv=cv)

        lr_sum = summarize_cv(lr_scores)
        svm_sum = summarize_cv(svm_scores)

        print("LogReg mean_acc/std:", lr_sum["mean_test_score"], lr_sum["std_test_score"])
        print("SVC    mean_acc/std:", svm_sum["mean_test_score"], svm_sum["std_test_score"])

        # Decide winners by calculation (not hard-coded)
        higher_acc = "LogisticRegression" if lr_sum["mean_test_score"] >= svm_sum["mean_test_score"] else "SVC"
        lower_var = "LogisticRegression" if lr_sum["std_test_score"] <= svm_sum["std_test_score"] else "SVC"
        faster = "LogisticRegression" if lr_sum["mean_fit_time"] <= svm_sum["mean_fit_time"] else "SVC"

        return {
            "logistic_regression": lr_sum,
            "svc": svm_sum,
            "highest_mean_accuracy": higher_acc,
            "lowest_variance": lower_var,
            "fastest_to_train": faster,
            "explanation": (
                "We used the SAME ShuffleSplit instance for both models so their splits match exactly. "
                "This makes the comparison fair."
            ),
        }

    # -------------------------------
    # Part G
    # -------------------------------
    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """Tune SVC hyperparameters, then train on all train data and test on held-out test data."""

        # Use ShuffleSplit CV for tuning (consistent with earlier parts)
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=self.seed)

        # Small grid to keep runtime reasonable
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", 0.01, 0.1],
            "kernel": ["rbf"],
        }

        base = SVC()

        gs = GridSearchCV(base, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
        gs.fit(X, y)

        best = gs.best_estimator_

        # Fit on full training data
        best.fit(X, y)

        train_acc = accuracy_score(y, best.predict(X))
        test_acc = accuracy_score(ytest, best.predict(Xtest))

        print("Best SVC params:", gs.best_params_)
        print("Train acc:", train_acc)
        print("Test acc:", test_acc)

        return {
            "best_params": gs.best_params_,
            "cv_best_mean_accuracy": float(gs.best_score_),
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "comment": (
                "If train accuracy >> test accuracy, that suggests overfitting. "
                "If test accuracy is close to CV mean accuracy, that's what we want."
            ),
        }
