import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

import utils as u
import new_utils as nu


class Section3:
    def __init__(self, normalize: bool = True, frac_train=0.2, seed=42):
        self.seed = seed
        self.normalize = normalize
        self.frac_train = frac_train

    def partA(self, Xtrain, ytrain, Xtest, ytest):
        from sklearn.linear_model import LogisticRegression

        Xtrain = nu.scale_data(Xtrain); Xtest = nu.scale_data(Xtest)
        ytrain = np.asarray(ytrain, dtype=np.int32); ytest = np.asarray(ytest, dtype=np.int32)

        clf = LogisticRegression(max_iter=300, random_state=self.seed)
        clf.fit(Xtrain, ytrain)
        prob_train = clf.predict_proba(Xtrain)
        prob_test = clf.predict_proba(Xtest)

        cls = clf.classes_
        idx_map = {int(lbl): i for i, lbl in enumerate(cls)}
        ytr_idx = np.array([idx_map[int(v)] for v in ytrain], dtype=int)
        yte_idx = np.array([idx_map[int(v)] for v in ytest], dtype=int)

        def topk_acc(y_idx, prob, k):
            topk = np.argpartition(prob, -k, axis=1)[:, -k:]
            return float(np.mean(np.any(topk == y_idx[:, None], axis=1)))

        answer: dict[Any, Any] = {"clf": clf}
        plot_train=[]; plot_test=[]
        for k in [1,2,3,4,5]:
            s_tr = topk_acc(ytr_idx, prob_train, k)
            s_te = topk_acc(yte_idx, prob_test, k)
            answer[k] = {"score_train": float(s_tr), "score_test": float(s_te)}
            plot_train.append((k, float(s_tr)))
            plot_test.append((k, float(s_te)))
        answer["plot_k_vs_score_train"]=plot_train
        answer["plot_k_vs_score_test"]=plot_test
        answer["text_rate_accuracy_change"]="Top-k accuracy increases with k, but gains typically diminish as k grows."
        answer["text_is_topk_useful_and_why"]="Useful if you can consider multiple candidates; for strict classification, k=1 is standard."
        return answer, Xtrain, ytrain, Xtest, ytest

    def partB(self, X, y, Xtest, ytest):
        Xtr, ytr = u.filter_out_7_9s(X, y)
        Xte, yte = u.filter_out_7_9s(Xtest, ytest)

        Xtr = nu.scale_data(Xtr); Xte = nu.scale_data(Xte)
        ytr = np.asarray(ytr, dtype=np.int32); yte = np.asarray(yte, dtype=np.int32)

        ytr = (ytr == 9).astype(np.int32)
        yte = (yte == 9).astype(np.int32)

        idx1 = np.where(ytr == 1)[0]
        rng = np.random.default_rng(self.seed)
        rng.shuffle(idx1)
        n_remove = int(0.9 * len(idx1))
        remove = idx1[:n_remove]
        keep = np.ones(len(ytr), dtype=bool); keep[remove]=False

        Xtr = Xtr[keep]; ytr = ytr[keep]

        answer = {
            "length_Xtrain": int(len(Xtr)),
            "length_Xtest": int(len(Xte)),
            "length_ytrain": int(len(ytr)),
            "length_ytest": int(len(yte)),
            "max_Xtrain": float(np.max(Xtr)) if len(Xtr) else 0.0,
            "max_Xtest": float(np.max(Xte)) if len(Xte) else 0.0,
            "class_counts": {"0": int(np.sum(ytr==0)), "1": int(np.sum(ytr==1))},
        }
        return answer, Xtr, ytr, Xte, yte

    def partC(self, X, y, Xtest, ytest):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        clf = SVC()

        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score),
            "f1": make_scorer(f1_score),
        }
        raw = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        def ms(key):
            arr=np.asarray(raw[key], dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        mean_acc,std_acc=ms("test_accuracy")
        mean_prec,std_prec=ms("test_precision")
        mean_rec,std_rec=ms("test_recall")
        mean_f1,std_f1=ms("test_f1")

        scores = {
            "mean_accuracy": mean_acc, "std_accuracy": std_acc,
            "mean_precision": mean_prec, "std_precision": std_prec,
            "mean_recall": mean_rec, "std_recall": std_rec,
            "mean_f1": mean_f1, "std_f1": std_f1,
        }

        is_prec_higher = bool(mean_prec > mean_rec)
        explain = "Precision is higher because the model avoids false positives on the minority class." if is_prec_higher else "Recall is higher because the model predicts the minority class more often."

        clf.fit(X, y)
        cm_train = confusion_matrix(y, clf.predict(X), labels=np.array([0,1], dtype=np.int32))
        cm_test = confusion_matrix(ytest, clf.predict(Xtest), labels=np.array([0,1], dtype=np.int32))

        return {
            "scores": scores,
            "cv": cv,
            "clf": clf,
            "is_precision_higher_than_recall": is_prec_higher,
            "explain_is_precision_higher_than_recall": explain,
            "confusion_matrix_train": cm_train,
            "confusion_matrix_test": cm_test,
        }

    def partD(self, X, y, Xtest, ytest):
        classes = np.unique(y)
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weights = {int(c): float(wi) for c, wi in zip(classes, w)}

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        clf = SVC(class_weight=class_weights)

        scoring = {
            "accuracy": "accuracy",
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score),
            "f1": make_scorer(f1_score),
        }
        raw = cross_validate(clf, X, y, cv=cv, scoring=scoring)

        def ms(key):
            arr=np.asarray(raw[key], dtype=float)
            return float(np.mean(arr)), float(np.std(arr))

        mean_acc,std_acc=ms("test_accuracy")
        mean_prec,std_prec=ms("test_precision")
        mean_rec,std_rec=ms("test_recall")
        mean_f1,std_f1=ms("test_f1")

        scores = {
            "mean_accuracy": mean_acc, "std_accuracy": std_acc,
            "mean_precision": mean_prec, "std_precision": std_prec,
            "mean_recall": mean_rec, "std_recall": std_rec,
            "mean_f1": mean_f1, "std_f1": std_f1,
        }

        clf.fit(X, y)
        cm_train = confusion_matrix(y, clf.predict(X), labels=np.array([0,1], dtype=np.int32))
        cm_test = confusion_matrix(ytest, clf.predict(Xtest), labels=np.array([0,1], dtype=np.int32))

        return {
            "scores": scores,
            "cv": cv,
            "clf": clf,
            "class_weights": class_weights,
            "confusion_matrix_train": cm_train,
            "confusion_matrix_test": cm_test,
            "explain_purpose_of_class_weights": "Weights penalize mistakes on minority class more to improve minority recall.",
            "explain_performance_difference": "Weighted SVM often increases recall for minority class, sometimes lowering precision.",
        }
