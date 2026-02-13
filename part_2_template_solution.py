# part_2_template_solution.py
# Section 2: Multi-class classification (0-9)

import numpy as np
from typing import Any, Dict, Optional

import utils as u
import new_utils as nu

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


class Section2:
    def __init__(self, normalize: bool = True, seed: Optional[int] = None, frac_train: float = 0.2):
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train

    # -------------------------
    # 2A
    # -------------------------
    def partA(self):
        X, y, Xtest, ytest = u.prepare_data()

        Xtrain = nu.scale_data(X)
        Xtest = nu.scale_data(Xtest)
        ytrain = np.asarray(y).astype(int)
        ytest = np.asarray(ytest).astype(int)

        # Print counts per class + number of classes
        uniq_tr, cnt_tr = np.unique(ytrain, return_counts=True)
        uniq_te, cnt_te = np.unique(ytest, return_counts=True)
        print("Train class counts:", {int(k): int(v) for k, v in zip(uniq_tr, cnt_tr)})
        print("Test class counts:", {int(k): int(v) for k, v in zip(uniq_te, cnt_te)})
        print("Num classes train:", int(len(uniq_tr)))
        print("Num classes test:", int(len(uniq_te)))

        answer: Dict[str, Any] = {}
        answer["nb_classes"] = {"train": int(len(uniq_tr)), "test": int(len(uniq_te))}
        answer["class_count"] = {"train": nu.class_counts(ytrain, 10), "test": nu.class_counts(ytest, 10)}
        answer["nb_samples_data"] = {"Xtrain": int(len(Xtrain)), "ytrain": int(len(ytrain)), "Xtest": int(len(Xtest)), "ytest": int(len(ytest))}
        answer["max_data"] = {"max_Xtrain": float(np.max(Xtrain)), "max_Xtest": float(np.max(Xtest))}

        return answer, Xtrain, ytrain, Xtest, ytest

    def _scores_to_dict(self, cv_scores: Dict[str, Any]) -> Dict[str, float]:
        acc = np.asarray(cv_scores["test_score"], dtype=float)
        ft = np.asarray(cv_scores["fit_time"], dtype=float)
        return {
            "mean_accuracy": float(acc.mean()),
            "std_accuracy": float(acc.std()),
            "mean_fit_time": float(ft.mean()),
            "std_fit_time": float(ft.std()),
        }

    # -------------------------
    # 2B
    # -------------------------
    def partB(self, X, y, Xtest, ytest, ntrain_list, ntest_list):
        # Use ntrain_list/ntest_list provided by run_part_2.py

        answer: Dict[str, Any] = {}

        # Explain multi-class logistic regression
        explain_multiclass = (
            "Multi-class logistic regression can be done as One-vs-Rest (OvR) or as a single multinomial "
            "(softmax) model. OvR trains one binary classifier per class; multinomial trains all classes jointly "
            "with a softmax probability distribution. sklearn's LogisticRegression can do either depending on solver."
        )

        for ntrain, ntest in zip(ntrain_list, ntest_list):
            Xtr = X[0:ntrain, :]
            ytr = y[0:ntrain]
            Xte = X[ntrain:ntrain + ntest, :]
            yte = y[ntrain:ntrain + ntest]

            # 1C-style: DecisionTree + KFold
            clf_1c = DecisionTreeClassifier(random_state=self.seed)
            cv_1c = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            scores_1c = self._scores_to_dict(cross_validate(clf_1c, Xtr, ytr, cv=cv_1c))

            # 1D-style: DecisionTree + ShuffleSplit
            clf_1d = DecisionTreeClassifier(random_state=self.seed)
            cv_1d = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)
            scores_1d = self._scores_to_dict(cross_validate(clf_1d, Xtr, ytr, cv=cv_1d))

            # 1F-style: LogisticRegression (300 iters) + ShuffleSplit
            cv_1f = ShuffleSplit(n_splits=5, test_size=self.frac_train, random_state=self.seed)
            clf_1f = LogisticRegression(max_iter=300, random_state=self.seed, solver='saga', tol=1e-2)
            cv_out_1f = cross_validate(clf_1f, Xtr, ytr, cv=cv_1f)
            mean_cv_accuracy_1f = float(np.mean(cv_out_1f["test_score"]))

            clf_1f.fit(Xtr, ytr)
            pred_tr = clf_1f.predict(Xtr)
            pred_te = clf_1f.predict(Xte)
            accuracy_train_1f = float(accuracy_score(ytr, pred_tr))
            accuracy_test_1f = float(accuracy_score(yte, pred_te))
            confusion_matrix_1f = {"train": confusion_matrix(ytr, pred_tr), "test": confusion_matrix(yte, pred_te)}

            # 1G-style: grid search over C
            clf_1g = LogisticRegression(max_iter=300, random_state=self.seed, solver='saga', tol=1e-2)
            default_parameters_1g = clf_1g.get_params()
            param_grid = {"C": [0.1, 1.0, 10.0]}
            grid_search_1g = GridSearchCV(clf_1g, param_grid=param_grid, cv=cv_1d, scoring="accuracy", n_jobs=-1)
            grid_search_1g.fit(Xtr, ytr)
            best_estimator_1g = grid_search_1g.best_estimator_

            best_estimator_1g.fit(Xtr, ytr)
            pred_tr_g = best_estimator_1g.predict(Xtr)
            pred_te_g = best_estimator_1g.predict(Xte)

            cm_train_g = confusion_matrix(ytr, pred_tr_g)
            cm_test_g = confusion_matrix(yte, pred_te_g)

            acc_train_g = float(accuracy_score(ytr, pred_tr_g))
            acc_test_g = float(accuracy_score(yte, pred_te_g))

            # Find a few hard pairs (largest off-diagonal confusions in test CM)
            cm_tmp = cm_test_g.copy()
            np.fill_diagonal(cm_tmp, 0)
            flat = np.argsort(cm_tmp.ravel())[::-1]
            hard_pairs = set()
            for idx in flat[:5]:
                if cm_tmp.ravel()[idx] == 0:
                    break
                i = int(idx // cm_tmp.shape[1])
                j = int(idx % cm_tmp.shape[1])
                hard_pairs.add((i, j))

            # Print requested summary
            print("ntrain={}, ntest={}, train_acc={:.4f}, test_acc={:.4f}".format(ntrain, ntest, accuracy_train_1f, accuracy_test_1f))

            answer[str(ntrain)] = {
                "clf_1c": clf_1c,
                "cv_1c": cv_1c,
                "scores_1c": scores_1c,

                "clf_1d": clf_1d,
                "cv_1d": cv_1d,
                "scores_1d": scores_1d,

                "accuracy_train_1f": accuracy_train_1f,
                "accuracy_test_1f": accuracy_test_1f,
                "mean_cv_accuracy_1f": mean_cv_accuracy_1f,
                "clf_1f": clf_1f,
                "cv_1f": cv_1f,
                "confusion_matrix_1f": confusion_matrix_1f,

                "clf_1g": clf_1g,
                "best_estimator_1g": best_estimator_1g,
                "grid_search_1g": grid_search_1g,
                "default_parameters_1g": default_parameters_1g,
                "confusion_matrix_1g": {"train": cm_train_g, "test": cm_test_g},
                "accuracy_1g": {"train": acc_train_g, "test": acc_test_g},
                "precision_1g": {"note": "macro precision computed in report if needed"},
                "class_count_1g": {"train": nu.class_counts(ytr, 10), "test": nu.class_counts(yte, 10)},
                "hard_to_distinguish_pairs": hard_pairs,

                "explain_multiclass_logistic_regression": explain_multiclass,
            }

        return answer
