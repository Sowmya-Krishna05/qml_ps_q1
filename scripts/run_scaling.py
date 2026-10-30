"""Qubit-scaling study: more qubits, longer training, same honest protocol. Writes results/qubit_scaling.csv."""
import os, sys, time, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from vqc.model import CircuitSpec, FeaturePipeline, VQC, circuit_stats, macro_f1
tr = pd.read_csv("data/q1/ecg_train.csv"); va = pd.read_csv("data/q1/ecg_val.csv")
F = [c for c in tr.columns if c != "label"]
Xtr, ytr, Xva, yva = tr[F].to_numpy(), tr.label.to_numpy(), va[F].to_numpy(), va.label.to_numpy()
EPOCHS, BATCH, LR = int(os.environ.get("EP", 40)), 250, 0.05
cfgs = [(4,1,1),(6,1,1),(8,1,1),(10,1,1),(4,2,2),(8,2,2) if False else (6,1,2),(8,1,2)]  # (qubits, layers, feats/qubit)
rows = []
for n, L, fpq in cfgs:
    for seed in (0, 1):
        spec = CircuitSpec(n_qubits=n, n_layers=L, feats_per_qubit=fpq, entangler="ring", readout=tuple(range(n)), reupload=(L>1))
        fp = FeaturePipeline(spec.n_features, "pca", seed).fit(Xtr, ytr)
        A, Av = fp.transform(Xtr), fp.transform(Xva)
        m = VQC(spec, seed); t = time.time()
        m.fit(A, ytr, Av, yva, epochs=EPOCHS, batch=BATCH, lr=LR)
        r = dict(qubits_cfg=n, layers_cfg=L, feats_per_qubit=fpq, pca_components=spec.n_features, seed=seed,
                 train_macro_f1=macro_f1(ytr, m.predict(A)), val_macro_f1=macro_f1(yva, m.predict(Av)),
                 seconds=round(time.time()-t,1), **circuit_stats(m.ops))
        rows.append(r); print(r, flush=True)
        pd.DataFrame(rows).to_csv("results/qubit_scaling.csv", index=False)
