"""Design-selection and ablation sweeps (NumPy simulator, validation data only).

`python scripts/run_sweeps.py` regenerates results/model_selection.csv and results/ablation.csv
(about 25 min on 2 CPU cores). solution.ipynb loads those files (RUN_SWEEPS=False) or re-runs them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import CircuitSpec, FeaturePipeline, VQC, circuit_stats, macro_f1
from .sim import DensityMatrixSim

EPOCHS, BATCH, LR = 20, 500, 0.08


def run_one(task):
    """task = dict(spec=CircuitSpec kwargs, reducer, seed, data=(Xtr,ytr,Xva,yva), tag=dict)"""
    Xtr, ytr, Xva, yva = task["data"]
    spec = CircuitSpec(**task["spec"])
    fp = FeaturePipeline(spec.n_features, task["reducer"], task["seed"]).fit(Xtr, ytr)
    A, Av = fp.transform(Xtr), fp.transform(Xva)
    m = VQC(spec, task["seed"])
    m.fit(A, ytr, Av, yva, epochs=EPOCHS, batch=BATCH, lr=LR)
    f1 = macro_f1(yva, m.predict(Av))
    # development replica of the organiser noise (exact expectation, first 500 val samples)
    dm = DensityMatrixSim(spec.n_qubits)
    sub = slice(0, 500)
    fn = dm.expz(dm.run(m.ops, Av[sub], m.result.theta), spec.readout)
    f1n = macro_f1(yva[sub], (m.result.w * fn + m.result.b > 0).astype(int))
    st = circuit_stats(m.ops)
    return {**task["tag"], "seed": task["seed"], "val_macro_f1": f1, "dev_noisy_val_macro_f1": f1n,
            "train_seconds": m.result.seconds, **st}


def selection_tasks(data, seeds=(0, 1, 2)):
    out = []
    for n in (3, 4):
        for L in (1, 2):
            for red in ("anova", "pca"):
                for ro in ("Z0", "Z-all"):
                    for s in seeds:
                        readout = (0,) if ro == "Z0" else tuple(range(n))
                        out.append(dict(spec=dict(n_qubits=n, n_layers=L, entangler="ring", readout=readout),
                                        reducer=red, seed=s, data=data,
                                        tag=dict(qubits_cfg=n, layers=L, reducer=red, readout=ro)))
    return out


def _resolve(spec: dict) -> dict:
    spec = dict(spec)
    if spec.get("readout") == "all":
        spec["readout"] = tuple(range(spec["n_qubits"]))
    elif spec.get("readout") == "Z0":
        spec["readout"] = (0,)
    return spec


def ablation_tasks(base_spec: dict, base_reducer: str, data, seeds=(0, 1, 2)):
    """One-factor-at-a-time ablations around the final design (base_spec uses readout='all')."""
    variants = {
        "final design": {},
        "no entanglement (product circuit)": dict(entangler="none"),
        "chain entangler (no wrap-around CX)": dict(entangler="chain"),
        "RY-only trainable layer": dict(trainable="ry"),
        "readout Z on qubit 0 only": dict(readout="Z0"),
        "2 layers, data re-uploaded": dict(n_layers=2, reupload=True),
        "2 layers, data uploaded once": dict(n_layers=2, reupload=False),
        "3 qubits (3 features)": dict(n_qubits=3),
        "2 qubits (2 features)": dict(n_qubits=2, entangler="chain"),
        "5 qubits (5 features)": dict(n_qubits=5),
    }
    out = []
    for name, ch in variants.items():
        for s in seeds:
            out.append(dict(spec=_resolve({**base_spec, **ch}), reducer=base_reducer, seed=s, data=data,
                            tag=dict(variant=name)))
    for red in ("anova", "pca"):
        if red != base_reducer:
            for s in seeds:
                out.append(dict(spec=_resolve(base_spec), reducer=red, seed=s, data=data,
                                tag=dict(variant=f"feature reducer: {red} instead of {base_reducer}")))
    return out


def run_tasks(tasks, n_jobs=1):
    if n_jobs > 1:
        from multiprocessing import Pool
        with Pool(n_jobs) as p:
            res = p.map(run_one, tasks, chunksize=1)
    else:
        res = [run_one(t) for t in tasks]
    return pd.DataFrame(res)


def pick_final(df: pd.DataFrame, tol: float = 0.0015):
    """Selection rule (validation only): best mean val macro-F1; among configs within `tol` of the
    best choose the one with the fewest two-qubit gates, then fewest qubits."""
    keys = ["qubits_cfg", "layers", "reducer", "readout"]
    g = df.groupby(keys).agg(val=("val_macro_f1", "mean"), val_sd=("val_macro_f1", "std"),
                             noisy=("dev_noisy_val_macro_f1", "mean"), cx=("two_qubit_gates", "first"),
                             depth=("depth", "first")).reset_index()
    best = g["val"].max()
    near = g[g["val"] >= best - tol].sort_values(["cx", "qubits_cfg", "depth", "val"], ascending=[True, True, True, False])
    return near.iloc[0], g.sort_values("val", ascending=False)
