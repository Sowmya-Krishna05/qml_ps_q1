"""Leakage / sanity audit of the PS-Q1 pipeline (NumPy side). Writes results/audit.json."""
import os, sys, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from vqc.model import CircuitSpec, FeaturePipeline, VQC
from vqc import experiments as ex
tr = pd.read_csv("data/q1/ecg_train.csv"); va = pd.read_csv("data/q1/ecg_val.csv"); te = pd.read_csv("data/q1/ecg_test_features.csv")
F = [c for c in tr.columns if c != "label"]; Xtr, ytr, Xva, yva, Xte = tr[F].to_numpy(), tr.label.to_numpy(), va[F].to_numpy(), va.label.to_numpy(), te[F].to_numpy()
mf = lambda y, p: float(f1_score(y, p, average="macro")); out = {}
spec = CircuitSpec(n_qubits=4, n_layers=1, entangler="ring", readout=(0, 1, 2, 3))
def vqc_run(Xa, ya, Xb, yb, seed=0, epochs=ex.EPOCHS):
    fp = FeaturePipeline(4, "pca", seed).fit(Xa, ya); m = VQC(spec, seed); m.fit(fp.transform(Xa), ya, epochs=epochs, batch=ex.BATCH, lr=ex.LR)
    return mf(ya, m.predict(fp.transform(Xa))), mf(yb, m.predict(fp.transform(Xb)))
# 1 class overlap of the best single feature (train) and a train-fitted threshold applied to validation
auc = pd.Series({c: roc_auc_score(ytr, tr[c]) for c in F}); top = (auc - .5).abs().sort_values(ascending=False)
sep = [c for c in top.index if top[c] >= .49]; out["perfectly_separating_features_train"] = sep
c0 = top.index[0]; a, b = tr[c0][ytr == 0], tr[c0][ytr == 1]
out["best_feature"] = dict(name=c0, normal_range=[float(a.min()), float(a.max())], arrhythmia_range=[float(b.min()), float(b.max())])
stump = DecisionTreeClassifier(max_depth=1).fit(tr[[c0]], ytr); out["one_feature_stump_val_macroF1"] = mf(yva, stump.predict(va[[c0]]))
# 2 distribution shift train vs val vs test (adversarial): can a model tell the splits apart?
for nm, B in (("val", Xva), ("test", Xte)):
    Z = np.vstack([Xtr[:1500], B]); y = np.r_[np.zeros(1500), np.ones(len(B))]; idx = np.random.default_rng(0).permutation(len(y)); h = len(y) // 2
    sc = StandardScaler().fit(Z[idx[:h]]); lr = LogisticRegression(max_iter=2000).fit(sc.transform(Z[idx[:h]]), y[idx[:h]])
    out[f"adversarial_auc_train_vs_{nm}"] = float(roc_auc_score(y[idx[h:]], lr.predict_proba(sc.transform(Z[idx[h:]]))[:, 1]))
# 3 shuffled-label control: a leak-free pipeline must fall to chance
rng = np.random.default_rng(0); yperm = rng.permutation(ytr)
out["shuffled_labels_VQC_train_val_macroF1"] = vqc_run(Xtr, yperm, Xva, yva)
sc = StandardScaler().fit(Xtr); out["shuffled_labels_LR_val_macroF1"] = mf(yva, LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), yperm).predict(sc.transform(Xva)))
# 4 learning curve: tiny training sets
lc = {}
for n in (10, 30, 100, 500):
    i = np.random.default_rng(1).choice(len(ytr), n, replace=False); r = []
    for s in range(3):
        try: r.append(vqc_run(Xtr[i], ytr[i], Xva, yva, seed=s, epochs=30)[1])
        except Exception as e: r.append(float("nan"))
    lc[n] = dict(vqc_val_macroF1_mean=float(np.nanmean(r)), lr=mf(yva, LogisticRegression(max_iter=2000).fit(StandardScaler().fit(Xtr[i]).transform(Xtr[i]), ytr[i]).predict(StandardScaler().fit(Xtr[i]).transform(Xva))))
out["learning_curve_train_size"] = lc
# 5 harder protocol: remove the perfectly separating features, then retrain everything
keep = [j for j, c in enumerate(F) if c not in sep]; out["n_features_after_removing_separable"] = len(keep)
sc2 = StandardScaler().fit(Xtr[:, keep]); lr2 = LogisticRegression(max_iter=3000).fit(sc2.transform(Xtr[:, keep]), ytr)
out["harder_LR_val_macroF1"] = mf(yva, lr2.predict(sc2.transform(Xva[:, keep])))
out["harder_VQC_train_val_macroF1"] = vqc_run(Xtr[:, keep], ytr, Xva[:, keep], yva)
# 6 label noise: flip 5% of training labels, test on clean validation
yn = ytr.copy(); f = np.random.default_rng(2).choice(len(yn), int(.05 * len(yn)), replace=False); yn[f] = 1 - yn[f]
out["5pct_flipped_train_labels_VQC_train_val_macroF1"] = vqc_run(Xtr, yn, Xva, yva)
json.dump(out, open("results/audit.json", "w"), indent=1); print(json.dumps(out, indent=1))
