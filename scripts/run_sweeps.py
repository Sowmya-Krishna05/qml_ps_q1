"""Regenerate results/model_selection.csv and results/ablation.csv (validation data only).

    python scripts/run_sweeps.py            # ~25 min on 2 CPU cores, uses 2 processes
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from vqc import experiments as ex  # noqa: E402

tr = pd.read_csv(f"{ROOT}/data/q1/ecg_train.csv")
va = pd.read_csv(f"{ROOT}/data/q1/ecg_val.csv")
data = (tr.drop(columns="label").to_numpy(), tr["label"].to_numpy(),
        va.drop(columns="label").to_numpy(), va["label"].to_numpy())
os.makedirs(f"{ROOT}/results", exist_ok=True)
jobs = min(2, os.cpu_count() or 1)

sel = ex.run_tasks(ex.selection_tasks(data), n_jobs=jobs)
sel.to_csv(f"{ROOT}/results/model_selection.csv", index=False)
best, table = ex.pick_final(sel)
print(table.round(4).to_string()); print("FINAL:", dict(best))

base = dict(n_qubits=int(best["qubits_cfg"]), n_layers=int(best["layers"]), entangler="ring",
            readout="all" if best["readout"] == "Z-all" else "Z0")
abl = ex.run_tasks(ex.ablation_tasks(base, best["reducer"], data), n_jobs=jobs)
abl.to_csv(f"{ROOT}/results/ablation.csv", index=False)
print(abl.groupby("variant")[["val_macro_f1", "dev_noisy_val_macro_f1"]].agg(["mean", "std"]).round(4).to_string())
