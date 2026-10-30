"""Extra figures from real data / sweep CSVs (NumPy side only): class balance, PCA space, model selection, ablation, dev-noise effect."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from vqc.model import CircuitSpec, FeaturePipeline, VQC
from vqc.sim import DensityMatrixSim
from vqc import experiments as ex
from sklearn.metrics import f1_score
O = "results/figures/"
tr = pd.read_csv("data/q1/ecg_train.csv"); va = pd.read_csv("data/q1/ecg_val.csv"); te = pd.read_csv("data/q1/ecg_test_features.csv")
F = [c for c in tr.columns if c != "label"]; Xtr, ytr, Xva, yva, Xte = tr[F].to_numpy(), tr.label.to_numpy(), va[F].to_numpy(), va.label.to_numpy(), te[F].to_numpy()

# 1 class balance + single-feature separability
from sklearn.metrics import roc_auc_score
auc = pd.Series({c: roc_auc_score(ytr, tr[c]) for c in F}); sep = (auc - .5).abs().sort_values(ascending=False) + .5
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].bar(["train", "validation"], [np.mean(ytr == 0), np.mean(yva == 0)], label="Normal"); ax[0].bar(["train", "validation"], [np.mean(ytr == 1), np.mean(yva == 1)], bottom=[np.mean(ytr == 0), np.mean(yva == 0)], label="Arrhythmia")
for i, (a, b) in enumerate(((np.mean(ytr == 0), np.mean(ytr == 1)), (np.mean(yva == 0), np.mean(yva == 1)))): ax[0].text(i, a / 2, f"{a:.1%}", ha="center", color="w"); ax[0].text(i, a + b / 2, f"{b:.1%}", ha="center", color="w")
ax[0].set_title("Class distribution"); ax[0].legend()
ax[1].bar(range(15), sep.values[:15], color=["tab:red" if v > .99 else "tab:gray" for v in sep.values[:15]]); ax[1].set_xticks(range(15)); ax[1].set_xticklabels(sep.index[:15], rotation=60, fontsize=8)
ax[1].axhline(.99, color="k", ls=":"); ax[1].set_ylim(.5, 1.01); ax[1].set_title("Single-feature ROC-AUC (train): red = separates classes alone")
fig.tight_layout(); fig.savefig(O + "class_balance_and_feature_separability.png", dpi=150)

# 2 PCA space of the 4 encoded features
best, table = ex.pick_final(pd.read_csv("results/model_selection.csv"))
fp = FeaturePipeline(4, str(best["reducer"]), 0).fit(Xtr, ytr); R = fp.reduce(Xva); pv = fp.pca.explained_variance_ratio_
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, (i, j) in zip(ax[:2], ((0, 1), (2, 3))):
    for c, col in ((0, "tab:blue"), (1, "tab:red")): a.scatter(R[yva == c, i], R[yva == c, j], s=4, alpha=.5, c=col, label=["Normal", "Arrhythmia"][c])
    a.set_xlabel(f"PC{i+1}"); a.set_ylabel(f"PC{j+1}"); a.legend(markerscale=3); a.set_title(f"Validation beats in PCA space (PC{i+1} vs PC{j+1})")
ax[2].bar(range(1, 5), pv); ax[2].set_title("Explained variance (4 components: %.1f%%)" % (100 * pv.sum())); ax[2].set_xlabel("component"); ax[2].set_xticks(range(1, 5))
fig.tight_layout(); fig.savefig(O + "pca_feature_space.png", dpi=150)

# 3 model-selection heatmap-style bars
t = table.copy(); t["cfg"] = t.apply(lambda r: f"{int(r.qubits_cfg)}q/{int(r.layers)}L/{r.reducer}/{r.readout}", axis=1); t = t.sort_values("val")
fig, ax = plt.subplots(figsize=(8, 6)); y = np.arange(len(t)); ax.barh(y - .2, t.val, .4, label="ideal"); ax.barh(y + .2, t.noisy, .4, label="dev-noisy")
ax.set_yticks(y); ax.set_yticklabels([f"{c}  ({int(k)} CX)" for c, k in zip(t.cfg, t.cx)], fontsize=8); ax.set_xlim(.65, 1.005); ax.set_xlabel("validation macro-F1 (mean of 3 seeds)"); ax.legend(); ax.set_title("Design selection (validation only)")
fig.tight_layout(); fig.savefig(O + "model_selection.png", dpi=150)

# 4 ablation
a = pd.read_csv("results/ablation.csv").groupby("variant", sort=False).agg(v=("val_macro_f1", "mean"), s=("val_macro_f1", "std"), n=("dev_noisy_val_macro_f1", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(8, 4.5)); y = np.arange(len(a))
ax.barh(y - .2, a.v, .4, xerr=a.s.fillna(0), label="ideal"); ax.barh(y + .2, a.n, .4, label="dev-noisy (density-matrix replica of QFF2026-v2)")
ax.set_yticks(y); ax.set_yticklabels(a.variant, fontsize=8); ax.invert_yaxis(); ax.set_xlim(.4, 1.005); ax.set_xlabel("validation macro-F1"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2); ax.set_title("Ablation study (3 seeds)")
fig.tight_layout(); fig.savefig(O + "ablation_study.png", dpi=150)

# 5 dev-noise effect on the final model (contraction)
spec = CircuitSpec(n_qubits=4, n_layers=1, entangler="ring", readout=(0, 1, 2, 3)); m = VQC(spec, 0)
Atr, Av = fp.transform(Xtr), fp.transform(Xva); m.fit(Atr, ytr, Av, yva, epochs=ex.EPOCHS, batch=ex.BATCH, lr=ex.LR)
th, w, b = m.result.theta, m.result.w, m.result.b; dm = DensityMatrixSim(4)
fi = m.expectation(Av, th); fn = dm.expz(dm.run(m.ops, Av, th), spec.readout)
pi_, pn_ = (w * fi + b > 0).astype(int), (w * fn + b > 0).astype(int)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for c, col in ((0, "tab:blue"), (1, "tab:red")):
    ax[0].hist(fi[yva == c], bins=40, alpha=.45, color=col, label=f"class {c} ideal"); ax[0].hist(fn[yva == c], bins=40, histtype="step", lw=2, color=col, label=f"class {c} noisy")
ax[0].axvline(-b / w, color="k", ls="--"); ax[0].set_xlabel("<ZZZZ>"); ax[0].legend(fontsize=7); ax[0].set_title("Read-out distribution, ideal vs noise replica")
ax[1].scatter(fi, fn, s=3, alpha=.4); sl, ic = np.polyfit(fi, fn, 1); xs = np.array([-1, 1]); ax[1].plot(xs, sl * xs + ic, "r", label=f"noisy = {sl:.3f}*ideal {ic:+.3f}"); ax[1].plot(xs, xs, "k:"); ax[1].legend(); ax[1].set_xlabel("ideal"); ax[1].set_ylabel("noisy"); ax[1].set_title("Contraction by depolarising + read-out noise")
fig.tight_layout(); fig.savefig(O + "noise_replica_effect.png", dpi=150)
print("dev-noise replica (NOT official Aer): val macro-F1 ideal %.4f -> noisy %.4f ; flipped decisions %d/1500 ; slope %.3f" % (f1_score(yva, pi_, average="macro"), f1_score(yva, pn_, average="macro"), int((pi_ != pn_).sum()), sl))
print("explained variance:", np.round(pv, 4), "sum %.4f" % pv.sum(), "| features with single-feature AUC>=0.99:", int((sep >= .99).sum()))
