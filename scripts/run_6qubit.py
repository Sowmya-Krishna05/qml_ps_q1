"""Full 6-qubit PS-Q1 run (NumPy exact simulator): train, metrics, curves, test copy, predictions_q1.csv.
    python scripts/run_6qubit.py      (about 5 min on 2 cores)"""
import json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
                             log_loss, brier_score_loss, ConfusionMatrixDisplay, average_precision_score)
from vqc.model import CircuitSpec, FeaturePipeline, VQC, build_ops, circuit_stats, macro_f1
from vqc.sim import DensityMatrixSim
from vqc import outputs, diagram
N, LAYERS, SEED, EPOCHS, BATCH, LR = 6, 1, 1, 40, 250, 0.05
tr = pd.read_csv("data/q1/ecg_train.csv"); va = pd.read_csv("data/q1/ecg_val.csv"); te = pd.read_csv("data/q1/ecg_test_features.csv")
F = [c for c in tr.columns if c != "label"]
Xtr, ytr, Xva, yva, Xte = tr[F].to_numpy(), tr.label.to_numpy(), va[F].to_numpy(), va.label.to_numpy(), te[F].to_numpy()
spec = CircuitSpec(n_qubits=N, n_layers=LAYERS, entangler="ring", readout=tuple(range(N)), reupload=False)
fp = FeaturePipeline(spec.n_features, "pca", SEED).fit(Xtr, ytr)
A, Av, At = fp.transform(Xtr), fp.transform(Xva), fp.transform(Xte)
m = VQC(spec, SEED); r = m.fit(A, ytr, Av, yva, epochs=EPOCHS, batch=BATCH, lr=LR)
h = pd.DataFrame(r.history); h.to_csv("results/training_history.csv", index=False)
th, w, b = r.theta, r.w, r.b
L = {k: w * m.expectation(Z, th) + b for k, Z in (("train", A), ("validation", Av), ("test", At))}
P = {k: (v > 0).astype(int) for k, v in L.items()}
# curves
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="validation"); ax[0].set_title("Class-weighted BCE loss")
ax[1].plot(h.epoch, h.train_loss_unweighted, label="train"); ax[1].plot(h.epoch, h.val_loss_unweighted, label="validation"); ax[1].set_title("Unweighted log-loss")
ax[2].plot(h.epoch, h.train_macro_f1, label="train"); ax[2].plot(h.epoch, h.val_macro_f1, label="validation"); ax[2].set_title("Macro-F1")
for a in ax: a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig("results/figures/training_validation_curves.png", dpi=150); plt.close(fig)
# metrics
rows = []
for name, y in (("train", ytr), ("validation", yva)):
    z, p = L[name], P[name]; pr = outputs.sigmoid(z); d = outputs.metrics_dict(y, p)
    d.update(split=name, specificity=d["TN"] / (d["TN"] + d["FP"]), balanced_accuracy=balanced_accuracy_score(y, p), mcc=matthews_corrcoef(y, p),
             roc_auc=roc_auc_score(y, pr), pr_auc=average_precision_score(y, pr), log_loss=log_loss(y, pr), brier=brier_score_loss(y, pr))
    rows.append(d)
M = pd.DataFrame(rows).set_index("split"); M.to_csv("results/extended_metrics.csv"); print(M.T.round(4).to_string())
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (name, y) in zip(ax[:2], (("train", ytr), ("validation", yva))):
    ConfusionMatrixDisplay(confusion_matrix(y, P[name]), display_labels=["Normal", "Arrhythmia"]).plot(ax=a, colorbar=False, cmap="Blues"); a.set_title(f"Confusion matrix - {name} (n={len(y)})")
fpr, tpr, _ = roc_curve(yva, outputs.sigmoid(L["validation"])); ax[2].plot(fpr, tpr); ax[2].plot([0, 1], [0, 1], "k:"); ax[2].set_title("ROC - validation (AUC %.4f)" % M.loc["validation", "roc_auc"])
fig.tight_layout(); fig.savefig("results/figures/confusion_matrices_roc.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.4))
for c, col in ((0, "tab:blue"), (1, "tab:red")): ax.hist(L["validation"][yva == c], bins=50, alpha=.6, color=col, label=["Normal", "Arrhythmia"][c])
ax.axvline(0, color="k", ls="--"); ax.set_title("Decision margin (validation)"); ax.legend(); fig.tight_layout(); fig.savefig("results/figures/logit_margins.png", dpi=150); plt.close(fig)
# architecture
ops = m.ops
diagram.draw_vqc(ops, N, spec.readout, th, "results/figures/vqc_architecture.png", "6-qubit hybrid VQC")
# baselines
sc = StandardScaler().fit(Xtr); lr30 = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xtr), ytr)
lr6 = LogisticRegression(max_iter=3000).fit(A, ytr)
base = {"VQC 6 qubits - validation": (yva, P["validation"]),
        "Logistic regression, 30 features - validation": (yva, lr30.predict(sc.transform(Xva))),
        "Logistic regression, same 6 inputs - validation": (yva, lr6.predict(Av))}
mt = outputs.metrics_table(base); mt.to_csv("results/metrics_summary.csv", index=False); print(mt.to_string())
# dev replica of organiser noise (exact density matrix, first 500 validation beats) - NOT Aer
dm = DensityMatrixSim(N); fn = dm.expz(dm.run(m.ops, Av[:500], th), spec.readout)
f1n = macro_f1(yva[:500], (w * fn + b > 0).astype(int)); f1i = macro_f1(yva[:500], P["validation"][:500])
# outputs
outputs.write_test_copy(te, P["test"], L["test"], "results/test_samples_with_predictions.csv")
pd.DataFrame({"sample_id": te["sample_id"], "label": P["test"]}).to_csv("predictions_q1.csv", index=False)
pt = outputs.sigmoid(L["test"])
info = dict(qubits=N, layers=LAYERS, reducer="StandardScaler->PCA(6)->percentile angle map", seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, optimizer="Adam",
            pca_explained_variance=[round(float(v), 4) for v in fp.pca.explained_variance_ratio_], pca_total=round(float(fp.pca.explained_variance_ratio_.sum()), 4),
            n_circuit_params=int(len(th)), theta=[round(float(t), 4) for t in th], w=round(float(w), 4), b=round(float(b), 4),
            train_seconds=round(r.seconds, 1), logical_circuit=circuit_stats(m.ops),
            dev_replica_noisy_val_macro_f1_500=round(f1n, 4), ideal_same_500=round(f1i, 4),
            test_pred_normal=int((P["test"] == 0).sum()), test_pred_arrhythmia=int((P["test"] == 1).sum()),
            test_agreement_with_LR30=float((P["test"] == lr30.predict(sc.transform(Xte))).mean()),
            test_share_confident=float(((pt > .95) | (pt < .05)).mean()))
json.dump(info, open("results/training_summary.json", "w"), indent=1); print(json.dumps(info, indent=1))
