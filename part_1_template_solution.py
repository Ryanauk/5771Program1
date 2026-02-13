# part_1_template_solution.py
# Section 1: Binary Classification (7 vs 9)
# Follow the instructions in assignment handout and run_part_1.py.

import numpy as np
from typing import Any, Dict, Optional

import utils as u
import new_utils as nu

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


class Section1:
    def __init__(self, normalize: bool = True, seed: Optional[int] = None, frac_train: float = 0.2):
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # -------------------------
    # 1A
    # -------------------------
    def partA(self) -> int:
        return int(u.starter_code())

    # -------------------------
    # 1B
    # -------------------------
    def partB(self):
        # Load and filter to 7/9
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)

        # Scale to [0,1] and enforce float
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        # Ensure labels are ints
        ytrain = np.asarray(ytrain).astype(int)
        ytest = np.asarray(ytest).astype(int)

        # Required prints
        print("(Train) len(X), len(y):", len(Xtrain), len(ytrain))
        print("(Test)  len(X), len(y):", len(Xtest), len(ytest))
        print("Max Xtrain:", float(np.max(Xtrain)))
        print("Max Xtest:", float(np.max(Xtest)))

        answer: Dict[str, Any] = {}
        # Autograder expects these dicts
        answer["number_of_samples"] = {
            "Xtrain": int(len(Xtrain)),
            "ytrain": int(len(ytrain)),
            "Xtest": int(len(Xtest)),
            "ytest": int(len(ytest)),
        }
        answer["data_bounds"] = {
            "min_Xtrain": float(np.min(Xtrain)),
            "max_Xtrain": float(np.max(Xtrain)),
            "min_Xtest": float(np.min(Xtest)),
            "max_Xtest": float(np.max(Xtest)),
        }
        return answer, Xtrain, ytrain, Xtest, ytest

    def _scores_to_dict(self, cv_scores: Dict[str, Any]) -> Dict[str, float]:
        # cross_validate returns arrays; convert to floats
        acc = np.asarray(cv_scores["test_score"], dtype=float)
        ft = np.asarray(cv_scores["fit_time"], dtype=float)
        return {
            "mean_accuracy": float(acc.mean()),
            "std_accuracy": float(acc.std()),
            "mean_fit_time": float(ft.mean()),
            "std_fit_time": float(ft.std()),
        }

    # -------------------------
    # 1C
    # -------------------------
    def partC(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        scores = cross_validate(clf, X, y, cv=cv)

        out_scores = self._scores_to_dict(scores)

        # Required prints (mean/std of accuracy + training time)
        print("DecisionTreeClassifier mean_test_score={:.4f} std_test_score={:.4f}".format(
            out_scores["mean_accuracy"], out_scores["std_accuracy"]
        ))
        print("DecisionTreeClassifier mean_fit_time={:.4f} std_fit_time={:.4f}".format(
            out_scores["mean_fit_time"], out_scores["std_fit_time"]
        ))

        return {"clf": clf, "cv": cv, "scores": out_scores}

    # -------------------------
    # 1D
    # -------------------------
    def partD(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)
        scores = cross_validate(clf, X, y, cv=cv)

        out_scores = self._scores_to_dict(scores)

        print("DecisionTreeClassifier mean_test_score={:.4f} std_test_score={:.4f}".format(
            out_scores["mean_accuracy"], out_scores["std_accuracy"]
        ))
        print("DecisionTreeClassifier mean_fit_time={:.4f} std_fit_time={:.4f}".format(
            out_scores["mean_fit_time"], out_scores["std_fit_time"]
        ))

        explain = (
            "Pros of ShuffleSplit: flexible train/test fraction; repeated random splits can estimate "
            "performance under multiple random partitions. Cons: validation splits may overlap, so some "
            "examples can appear multiple times (or not at all), increasing variance. "
            "Pros of KFold: each sample is used exactly once for validation; deterministic coverage. "
            "Cons: less flexible split sizes and can be slower if k is large."
        )

        return {"clf": clf, "cv": cv, "scores": out_scores, "explain_kfold_vs_shuffle_split": explain}

    # -------------------------
    # 1E
    # -------------------------
    def partE(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for k in [2, 5, 8, 16]:
            clf = DecisionTreeClassifier(random_state=self.seed)
            cv = ShuffleSplit(n_splits=k, test_size=self.frac_train, random_state=self.seed)
            scores = cross_validate(clf, X, y, cv=cv)
            acc = np.asarray(scores["test_score"], dtype=float)
            score_dict = {"mean_accuracy": float(acc.mean()), "std_accuracy": float(acc.std())}

            # Required prints: mean/std only (no training time)
            print("ShuffleSplit k={}: mean_acc={:.4f}, std_acc={:.4f}".format(k, score_dict["mean_accuracy"], score_dict["std_accuracy"]))

            out[k] = {"scores": score_dict}
        return out

    # -------------------------
    # 1F
    # -------------------------
    def partF(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        # Use SAME cv instance so splits are identical for DT and RF
        cv = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)

        clf_RF = RandomForestClassifier(random_state=self.seed)
        clf_DT = DecisionTreeClassifier(random_state=self.seed)

        rf_scores = self._scores_to_dict(cross_validate(clf_RF, X, y, cv=cv))
        dt_scores = self._scores_to_dict(cross_validate(clf_DT, X, y, cv=cv))

        model_highest_accuracy = "RandomForest" if rf_scores["mean_accuracy"] >= dt_scores["mean_accuracy"] else "DecisionTree"
        model_lowest_variance = "RandomForest" if rf_scores["std_accuracy"] <= dt_scores["std_accuracy"] else "DecisionTree"
        model_fastest = "RandomForest" if rf_scores["mean_fit_time"] <= dt_scores["mean_fit_time"] else "DecisionTree"

        return {
            "clf_RF": clf_RF,
            "cv_RF": cv,
            "scores_RF": rf_scores,
            "clf_DT": clf_DT,
            "cv_DT": cv,
            "scores_DT": dt_scores,
            "model_highest_accuracy": model_highest_accuracy,
            "model_lowest_variance": model_lowest_variance,
            "model_fastest": model_fastest,
        }

    # -------------------------
    # 1G
    # -------------------------
    def partG(self, X: np.ndarray, y: np.ndarray, Xtest: np.ndarray, ytest: np.ndarray) -> Dict[str, Any]:
        clf = RandomForestClassifier(random_state=self.seed)
        default_parameters = clf.get_params()

        cv = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)

        # Grid search on allowed hyperparameters
        param_grid = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }

        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
        grid_search.fit(X, y)

        best = grid_search.best_estimator_
        best.fit(X, y)

        yhat_train = best.predict(X)
        yhat_test = best.predict(Xtest)

        acc_train = float(accuracy_score(y, yhat_train))
        acc_test = float(accuracy_score(ytest, yhat_test))
        prec_train = float(precision_score(y, yhat_train, zero_division=0))
        prec_test = float(precision_score(ytest, yhat_test, zero_division=0))

        cm_train = confusion_matrix(y, yhat_train)
        cm_test = confusion_matrix(ytest, yhat_test)

        print("Train acc:", acc_train)
        print("Test acc:", acc_test)

        comment = (
            "Training accuracy on the full training set is usually higher than mean CV accuracy "
            "because the model is evaluated on the same data it was trained on. Test accuracy is "
            "typically closer to CV mean accuracy (sometimes slightly higher/lower depending on split)."
        )

        return {
            "clf": clf,
            "grid_search": grid_search,
            "default_parameters": default_parameters,
            "confusion_matrix": {"train": cm_train, "test": cm_test},
            "accuracy_full_training": {"train": acc_train, "test": acc_test},
            "precision_full_training": {"train": prec_train, "test": prec_test},
            "comment_train_test_vs_cv": comment,
        }
