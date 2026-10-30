"""Classical baselines trained under the same conditions as the quantum model:
fit on ecg_train only, hyper-parameters chosen on ecg_val only (small grids), same scaler."""
from __future__ import annotations

import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _f1(y, p):
    return f1_score(y, p, average="macro")


def tune(make, grid, Xtr, ytr, Xva, yva):
    best = None
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid.keys(), vals))
        m = make(**kw).fit(Xtr, ytr)
        s = _f1(yva, m.predict(Xva))
        if best is None or s > best[0] + 1e-12:
            best = (s, kw, m)
    return best


def run_baselines(Xtr, ytr, Xva, yva, seed=42, reduced=None):
    """Returns {name: dict(val_macro_f1, params, model)}.
    reduced = (Rtr, Rva): the same low-dimensional inputs that feed the quantum circuit."""
    out = {}
    def lr(C, class_weight): return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=seed))
    def svm(C, gamma): return make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, class_weight="balanced", random_state=seed))
    def rf(n_estimators, max_depth): return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced", random_state=seed, n_jobs=1)
    specs = [("Logistic regression (30 features)", lr, dict(C=[0.01, 0.1, 1, 10], class_weight=[None, "balanced"]), Xtr, Xva),
             ("RBF-SVM (30 features)", svm, dict(C=[0.1, 1, 10, 100], gamma=["scale", 0.01, 0.1]), Xtr, Xva),
             ("Random forest (30 features)", rf, dict(n_estimators=[100, 300], max_depth=[None, 8]), Xtr, Xva)]
    if reduced is not None:
        specs.append((f"Logistic regression ({reduced[0].shape[1]} reduced features, same as circuit)", lr,
                      dict(C=[0.01, 0.1, 1, 10], class_weight=[None, "balanced"]), reduced[0], reduced[1]))
    for name, make, grid, A, B in specs:
        s, kw, m = tune(make, grid, A, ytr, B, yva)
        out[name] = dict(val_macro_f1=s, params=kw, model=m, inputs=(A, B))
    return out
