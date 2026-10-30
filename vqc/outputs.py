"""Requested deliverable outputs: full classification metrics and a per-sample copy of the test set.

Test labels are held by the organisers, so accuracy / precision / recall / F1 can only be computed
on the labelled splits (train, validation). The test copy therefore carries the model's
predictions, probabilities and logits for every test sample, not scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CLASS_NAMES = {0: "Normal", 1: "Arrhythmia"}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def metrics_dict(y, p) -> dict:
    """Accuracy plus precision / recall / F1 for the abnormal class (label 1) and macro-averaged."""
    y, p = np.asarray(y), np.asarray(p)
    prec, rec, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
            "precision_abnormal": float(prec[1]), "recall_abnormal": float(rec[1]), "f1_abnormal": float(f1[1]),
            "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def metrics_table(rows: dict) -> pd.DataFrame:
    """rows = {"model / split": (y_true, y_pred)} -> one row per entry."""
    out = []
    for name, (y, p) in rows.items():
        out.append({"model / split": name, **metrics_dict(y, p)})
    return pd.DataFrame(out)


def write_test_copy(te: pd.DataFrame, pred, logit, path: str, extra: dict | None = None) -> pd.DataFrame:
    """Copy of ecg_test_features.csv (sample_id + 30 features, unchanged) with the model output appended."""
    pred = np.asarray(pred).astype(int)
    out = te.copy()
    out["predicted_label"] = pred
    out["predicted_class"] = [CLASS_NAMES[int(v)] for v in pred]
    out["p_arrhythmia"] = sigmoid(logit)
    out["logit"] = np.asarray(logit, dtype=float)
    for k, v in (extra or {}).items():
        out[k] = v
    assert len(out) == len(te) and (out["sample_id"].to_numpy() == te["sample_id"].to_numpy()).all()
    out.to_csv(path, index=False)
    return out
