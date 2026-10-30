"""Re-trains the selected model (same seed/hyper-parameters as the notebook) with per-epoch loss logging and writes
curves, confusion matrices, ROC and the extended metric table. Needs NumPy/scikit-learn/matplotlib only.

    python scripts/make_diagnostics.py
"""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
                             log_loss, brier_score_loss, ConfusionMatrixDisplay, average_precision_score)
from vqc.model import CircuitSpec, FeaturePipeline, VQC, build_ops, circuit_stats
from vqc import experiments as ex, outputs

tr = pd.read_csv("data/q1/ecg_train.csv"); va = pd.read_csv("data/q1/ecg_val.csv"); te = pd.read_csv("data/q1/ecg_test_features.csv")
F = [c for c in tr.columns if c != "label"]
Xtr, ytr, Xva, yva, Xte = tr[F].to_numpy(), tr.label.to_numpy(), va[F].to_numpy(), va.label.to_numpy(), te[F].to_numpy()
best, _ = ex.pick_final(pd.read_csv("results/model_selection.csv"))
spec = CircuitSpec(n_qubits=int(best["qubits_cfg"]), n_layers=int(best["layers"]), entangler="ring",
                   readout=tuple(range(int(best["qubits_cfg"]))) if best["readout"] == "Z-all" else (0,))
SEED = 0                                           # the seed the notebook keeps (best validation, lowest seed on ties)
fp = FeaturePipeline(spec.n_features, str(best["reducer"]), SEED).fit(Xtr, ytr)
A, Av, At = fp.transform(Xtr), fp.transform(Xva), fp.transform(Xte)
m = VQC(spec, SEED); r = m.fit(A, ytr, Av, yva, epochs=ex.EPOCHS, batch=ex.BATCH, lr=ex.LR)
h = pd.DataFrame(r.history); os.makedirs("results/figures", exist_ok=True); h.to_csv("results/training_history.csv", index=False)

# --- curves
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="validation"); ax[0].set_title("Class-weighted BCE loss (the training objective)")
ax[1].plot(h.epoch, h.train_loss_unweighted, label="train"); ax[1].plot(h.epoch, h.val_loss_unweighted, label="validation"); ax[1].set_title("Unweighted log-loss")
ax[2].plot(h.epoch, h.train_macro_f1, label="train"); ax[2].plot(h.epoch, h.val_macro_f1, label="validation"); ax[2].set_title("Macro-F1")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig("results/figures/training_validation_curves.png", dpi=150)

# --- predictions / metrics
th, w, b = r.theta, r.w, r.b
def logit(Z): return w * m.expectation(Z, th) + b
L = {"train": logit(A), "validation": logit(Av), "test": logit(At)}
P = {k: (v > 0).astype(int) for k, v in L.items()}
rows = []
for name, y in (("train", ytr), ("validation", yva)):
    z, p = L[name], P[name]; pr = outputs.sigmoid(z); d = outputs.metrics_dict(y, p)
    d.update(split=name, specificity=d["TN"] / (d["TN"] + d["FP"]), balanced_accuracy=balanced_accuracy_score(y, p), mcc=matthews_corrcoef(y, p),
             roc_auc=roc_auc_score(y, pr), pr_auc=average_precision_score(y, pr), log_loss=log_loss(y, pr), brier=brier_score_loss(y, pr),
             weighted_bce=float(h[f"{'train' if name=='train' else 'val'}_loss"].iloc[-1]))
    rows.append(d)
M = pd.DataFrame(rows).set_index("split"); M.to_csv("results/extended_metrics.csv"); print(M.T.round(4).to_string())

# --- confusion matrices and ROC
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (name, y) in zip(ax[:2], (("train", ytr), ("validation", yva))):
    ConfusionMatrixDisplay(confusion_matrix(y, P[name]), display_labels=["Normal", "Arrhythmia"]).plot(ax=a, colorbar=False, cmap="Blues"); a.set_title(f"Confusion matrix - {name} (n={len(y)})")
fpr, tpr, _ = roc_curve(yva, outputs.sigmoid(L["validation"])); ax[2].plot(fpr, tpr); ax[2].plot([0, 1], [0, 1], "k:"); ax[2].set_title("ROC - validation (AUC %.4f)" % M.loc["validation", "roc_auc"]); ax[2].set_xlabel("false-positive rate"); ax[2].set_ylabel("true-positive rate")
fig.tight_layout(); fig.savefig("results/figures/confusion_matrices_roc.png", dpi=150)
fig, ax = plt.subplots(figsize=(6, 3.4))
for c, col in ((0, "tab:blue"), (1, "tab:red")): ax.hist(L["validation"][yva == c], bins=50, alpha=.6, color=col, label=["Normal", "Arrhythmia"][c])
ax.axvline(0, color="k", ls="--"); ax.set_xlabel("logit w*<ZZZZ>+b (validation)"); ax.set_title("Decision margin"); ax.legend(); fig.tight_layout(); fig.savefig("results/figures/logit_margins.png", dpi=150)

# --- test set facts (labels hidden)
pt = outputs.sigmoid(L["test"])
tc = pd.read_csv("results/test_samples_with_predictions.csv")
same = bool((tc.predicted_label.to_numpy() == P["test"]).all())
info = dict(seed=SEED, epochs=ex.EPOCHS, batch=ex.BATCH, lr=ex.LR, optimizer="Adam", n_circuit_params=int(len(th)), theta=[round(float(t), 4) for t in th], w=round(float(w), 4), b=round(float(b), 4),
            train_seconds=round(r.seconds, 1), circuit_evaluations=int(r.circuit_evals), logical_circuit=circuit_stats(m.ops),
            test_n=len(pt), test_pred_normal=int((P["test"] == 0).sum()), test_pred_arrhythmia=int((P["test"] == 1).sum()),
            test_share_p_above_0_95_or_below_0_05=float(((pt > .95) | (pt < .05)).mean()), test_min_abs_logit=round(float(np.abs(L["test"]).min()), 3),
            test_predictions_identical_to_saved_file=same)
json.dump(info, open("results/training_summary.json", "w"), indent=1); print(json.dumps(info, indent=1))
