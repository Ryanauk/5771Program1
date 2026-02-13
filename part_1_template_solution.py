# Inspired by GPT4

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, cross_validate, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from typing import Any
from numpy.typing import NDArray

import numpy as np
import utils as u
import new_utils as nu


class Section1:
    def __init__(self, normalize: bool = True, seed: int | None = None, frac_train: float = 0.2):
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed

    def partA(self):
        return u.starter_code()

    def partB(self):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)
        ytrain = np.asarray(ytrain, dtype=np.int32)
        ytest = np.asarray(ytest, dtype=np.int32)

        print("(Train) len(X), len(y):", len(Xtrain), len(ytrain))
        print("(Test)  len(X), len(y):", len(Xtest), len(ytest))
        print("Max Xtrain:", float(np.max(Xtrain)))
        print("Max Xtest:", float(np.max(Xtest)))

        answer = {
            "length_Xtrain": int(len(Xtrain)),
            "length_Xtest": int(len(Xtest)),
            "length_ytrain": int(len(ytrain)),
            "length_ytest": int(len(ytest)),
            "max_Xtrain": float(np.max(Xtrain)),
            "max_Xtest": float(np.max(Xtest)),
        }
        return answer, Xtrain, ytrain, Xtest, ytest

    def partC(self, X: NDArray[np.floating], y: NDArray[np.int32]):
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        raw = cross_validate(clf, X, y, cv=cv)
        scores = nu.scores_from_cross_validate(raw)

        answer = {"clf": clf, "cv": cv, "scores": scores}
        return answer

    def partD(self, X: NDArray[np.floating], y: NDArray[np.int32]):
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)
        raw = cross_validate(clf, X, y, cv=cv)
        scores = nu.scores_from_cross_validate(raw)

        explain = (
            "KFold uses non-overlapping folds so each sample is validated once; it is stable. "
            "ShuffleSplit uses random splits (validation sets may overlap), offering flexibility but often higher variance."
        )
        answer = {"clf": clf, "cv": cv, "scores": scores, "explain_kfold_vs_shuffle_split": explain}
        return answer

    def partE(self, X: NDArray[np.floating], y: NDArray[np.int32]):
        answer: dict[int, dict[str, Any]] = {}
        for k in [2, 5, 8, 16]:
            clf = DecisionTreeClassifier(random_state=self.seed)
            cv = ShuffleSplit(n_splits=k, test_size=self.frac_train, random_state=self.seed)
            raw = cross_validate(clf, X, y, cv=cv)
            test = np.asarray(raw["test_score"], dtype=float)
            scores = {"mean_accuracy": float(np.mean(test)), "std_accuracy": float(np.std(test))}
            answer[k] = {"clf": clf, "cv": cv, "scores": scores}
        return answer

    def partF(self, X: NDArray[np.floating], y: NDArray[np.int32]) -> dict[str, Any]:
        cv = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)
        clf_RF = RandomForestClassifier(random_state=self.seed)
        clf_DT = DecisionTreeClassifier(random_state=self.seed)

        scores_RF = nu.scores_from_cross_validate(cross_validate(clf_RF, X, y, cv=cv))
        scores_DT = nu.scores_from_cross_validate(cross_validate(clf_DT, X, y, cv=cv))

        model_highest_accuracy = "RandomForestClassifier" if scores_RF["mean_accuracy"] >= scores_DT["mean_accuracy"] else "DecisionTreeClassifier"
        model_lowest_variance = "RandomForestClassifier" if scores_RF["std_accuracy"] <= scores_DT["std_accuracy"] else "DecisionTreeClassifier"
        model_fastest = "RandomForestClassifier" if scores_RF["mean_fit_time"] <= scores_DT["mean_fit_time"] else "DecisionTreeClassifier"

        return {
            "clf_RF": clf_RF,
            "clf_DT": clf_DT,
            "cv": cv,
            "scores_RF": scores_RF,
            "scores_DT": scores_DT,
            "model_highest_accuracy": model_highest_accuracy,
            "model_lowest_variance": model_lowest_variance,
            "model_fastest": model_fastest,
        }

    def partG(self, X: NDArray[np.floating], y: NDArray[np.int32], Xtest: NDArray[np.floating], ytest: NDArray[np.int32]) -> dict[str, Any]:
        clf = RandomForestClassifier(random_state=self.seed)
        default_parameters = clf.get_params()

        cv = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)

        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2", None],
            "n_estimators": [50, 100],
        }

        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv, scoring="accuracy", refit=True, n_jobs=-1)
        grid_search.fit(X, y)
        best_estimator = grid_search.best_estimator_

        labels = np.array([7, 9], dtype=np.int32)

        # Original
        clf.fit(X, y)
        cm_train_orig = confusion_matrix(y, clf.predict(X), labels=labels)
        cm_test_orig = confusion_matrix(ytest, clf.predict(Xtest), labels=labels)

        # Best
        cm_train_best = confusion_matrix(y, best_estimator.predict(X), labels=labels)
        cm_test_best = confusion_matrix(ytest, best_estimator.predict(Xtest), labels=labels)

        def acc(cm: np.ndarray) -> float:
            return float(np.trace(cm) / np.sum(cm)) if np.sum(cm) else 0.0

        def prec9(cm: np.ndarray) -> float:
            tp = float(cm[1, 1])
            fp = float(cm[0, 1])
            return tp / (tp + fp) if (tp + fp) else 0.0

        answer = {
            "clf": clf,
            "default_parameters": default_parameters,
            "best_estimator": best_estimator,
            "grid_search": grid_search,
            "mean_accuracy_cv": float(grid_search.best_score_),

            "confusion_matrix_train_orig": cm_train_orig,
            "confusion_matrix_train_best": cm_train_best,
            "confusion_matrix_test_orig": cm_test_orig,
            "confusion_matrix_test_best": cm_test_best,

            "accuracy_orig_full_training": acc(cm_train_orig),
            "accuracy_best_full_training": acc(cm_train_best),
            "accuracy_orig_full_testing": acc(cm_test_orig),
            "accuracy_best_full_testing": acc(cm_test_best),

            # aggregated variants
            "confusion_matrix": {
                "train_orig": cm_train_orig,
                "train_best": cm_train_best,
                "test_orig": cm_test_orig,
                "test_best": cm_test_best,
            },
            "accuracy_full_training": {"orig": acc(cm_train_orig), "best": acc(cm_train_best)},
            "precision_full_training": {"orig": prec9(cm_train_orig), "best": prec9(cm_train_best)},
        }
        return answer
