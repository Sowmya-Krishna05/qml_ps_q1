"""Small evaluation helpers (macro-F1 with bootstrap confidence intervals)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def macro_f1(y, p):
    return float(f1_score(y, p, average="macro"))


def bootstrap_ci(y, p, n=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI of macro-F1 over resampled samples."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    s = np.array([f1_score(y[i], p[i], average="macro") for i in idx])
    return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))


def paired_bootstrap_diff(y, p_a, p_b, n=2000, seed=42, alpha=0.05):
    """CI of macro-F1(a) - macro-F1(b) on the same resampled rows."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = np.array([f1_score(y[i], p_a[i], average="macro") - f1_score(y[i], p_b[i], average="macro")
                  for i in idx])
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def summary(y, p):
    lo, hi = bootstrap_ci(y, p)
    return {"macro_f1": macro_f1(y, p), "ci95": (lo, hi), "accuracy": float(accuracy_score(y, p)),
            "confusion": confusion_matrix(y, p).tolist()}
