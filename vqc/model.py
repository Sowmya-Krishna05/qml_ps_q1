"""Hybrid variational classifier for ECG heartbeats: features, circuit, training.

Pipeline (everything fitted on the TRAINING split only):
    30 raw statistics -> StandardScaler -> reducer (PCA or ANOVA top-k) -> robust angle map [0, pi]
    -> data re-uploading circuit on n qubits -> <Z_readout> -> logit = w * <Z> + b -> sigmoid

The circuit is stored as an op list (see vqc/sim.py) so one description serves the fast NumPy
simulator and the Qiskit builder used for the official noisy runs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .sim import StatevectorSim


# ----------------------------------------------------------------------------- circuit
@dataclass
class CircuitSpec:
    n_qubits: int = 4
    n_layers: int = 2
    feats_per_qubit: int = 1          # 1: RY(x_q);  2: RY(x_q) then RZ(x_{q+n})
    entangler: str = "ring"           # ring | chain | diamond | none | custom
    custom_edges: tuple = ()          # [(control, target), ...] for entangler="custom"
    reupload: bool = True             # re-encode the data in every layer?
    readout: tuple = (0,)             # qubits in the measured Z-string
    trainable: str = "ryrz"           # ryrz | ry

    @property
    def n_features(self):
        return self.n_qubits * self.feats_per_qubit

    def edges(self):
        n = self.n_qubits
        if self.entangler == "ring":
            return [(q, (q + 1) % n) for q in range(n)] if n > 2 else [(0, 1)]
        if self.entangler == "chain":
            return [(q, q + 1) for q in range(n - 1)]
        if self.entangler == "diamond":          # ring + chord 0-2 (K4 minus edge 1-3), needs n == 4
            return [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        if self.entangler == "none":
            return []
        if self.entangler == "custom":
            return list(self.custom_edges)
        raise ValueError(self.entangler)


def build_ops(spec: CircuitSpec):
    """Return (ops, n_trainable). Data refs use ('x', k) (angle = x[:, k])."""
    ops, t = [], 0
    n = spec.n_qubits
    for layer in range(spec.n_layers):
        if layer == 0 or spec.reupload:
            for q in range(n):
                ops.append(("ry", (q,), ("x", q)))
                if spec.feats_per_qubit == 2:
                    ops.append(("rz", (q,), ("x", q + n)))
        for q in range(n):
            ops.append(("ry", (q,), ("t", t))); t += 1
            if spec.trainable == "ryrz":
                ops.append(("rz", (q,), ("t", t))); t += 1
        for c, tg in spec.edges():
            ops.append(("cx", (c, tg), None))
    return ops, t


def circuit_stats(ops):
    """Logical resource counts (before compilation)."""
    n_q = 1 + max(q for _, qs, _ in ops for q in qs)
    free = [0] * n_q
    for g, qs, _ in ops:
        d = max(free[q] for q in qs) + 1
        for q in qs:
            free[q] = d
    return {"qubits": n_q, "depth": max(free),
            "two_qubit_gates": sum(g == "cx" for g, _, _ in ops),
            "one_qubit_gates": sum(g != "cx" for g, _, _ in ops),
            "trainable_gates": sum(1 for _, _, r in ops if r and r[0] == "t")}


# ----------------------------------------------------------------------------- features
class FeaturePipeline:
    """Scaler -> reducer -> angle map. `fit` sees training data only."""

    def __init__(self, n_out: int, reducer: str = "pca", seed: int = 0):
        self.n_out, self.reducer, self.seed = n_out, reducer, seed

    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        if self.reducer == "pca":
            self.pca = PCA(n_components=self.n_out, random_state=self.seed).fit(Z)
            R = self.pca.transform(Z)
        elif self.reducer == "anova":
            f, _ = f_classif(Z, y)
            self.idx = np.argsort(-f)[: self.n_out]
            R = Z[:, self.idx]
        else:
            raise ValueError(self.reducer)
        self.lo, self.hi = np.percentile(R, 1, axis=0), np.percentile(R, 99, axis=0)
        return self

    def reduce(self, X):
        Z = self.scaler.transform(X)
        return self.pca.transform(Z) if self.reducer == "pca" else Z[:, self.idx]

    def transform(self, X):
        """Angles in [0, pi]; values outside the training 1-99 percentile range are clipped."""
        R = self.reduce(X)
        return np.pi * np.clip((R - self.lo) / (self.hi - self.lo), 0.0, 1.0)


# ----------------------------------------------------------------------------- training
def _bce(z, y, pos_w=1.0):
    """Mean (class-weighted) binary cross-entropy of logits z against labels y, numerically stable."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    l = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(np.where(y == 1, pos_w, 1.0) * l))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class TrainResult:
    theta: np.ndarray
    w: float
    b: float
    history: list = field(default_factory=list)
    seconds: float = 0.0
    circuit_evals: int = 0


class VQC:
    def __init__(self, spec: CircuitSpec, seed: int = 0):
        self.spec, self.seed = spec, seed
        self.ops, self.n_params = build_ops(spec)
        self.sim = StatevectorSim(spec.n_qubits)

    # f(x; theta) = <Z_readout>
    def expectation(self, A, theta):
        return self.sim.expz(self.sim.run(self.ops, A, theta), self.spec.readout)

    def fit(self, A, y, A_val=None, y_val=None, epochs=40, batch=256, lr=0.05, class_weight=True,
            verbose=False):
        rng = np.random.default_rng(self.seed)
        theta = rng.uniform(-np.pi, np.pi, self.n_params) * 0.5
        w, b = 2.0, 0.0
        P = self.n_params
        m, v, tstep = np.zeros(P + 2), np.zeros(P + 2), 0
        pos_w = (len(y) - y.sum()) / max(y.sum(), 1) if class_weight else 1.0
        t0, evals, hist = time.time(), 0, []
        for ep in range(epochs):
            order = rng.permutation(len(y))
            for i in range(0, len(y), batch):
                idx = order[i:i + batch]
                Ab, yb = A[idx], y[idx]
                f = self.expectation(Ab, theta)
                p = _sigmoid(w * f + b)
                sw = np.where(yb == 1, pos_w, 1.0)
                r = sw * (p - yb)                         # dL/dlogit (weighted BCE)
                gw, gb = np.mean(r * f), np.mean(r)
                gt = np.zeros(P)
                for k in range(P):                        # parameter-shift rule (exact)
                    tp, tm = theta.copy(), theta.copy()
                    tp[k] += np.pi / 2; tm[k] -= np.pi / 2
                    df = 0.5 * (self.expectation(Ab, tp) - self.expectation(Ab, tm))
                    gt[k] = np.mean(r * w * df)
                evals += (2 * P + 1) * len(idx)
                g = np.concatenate([gt, [gw, gb]])
                tstep += 1
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                step = lr * (m / (1 - 0.9 ** tstep)) / (np.sqrt(v / (1 - 0.999 ** tstep)) + 1e-8)
                theta -= step[:P]; w -= step[P]; b -= step[P + 1]
            rec = {"epoch": ep + 1}
            # full-split diagnostics (logging only; they do not touch the optimisation)
            ztr = w * self.expectation(A, theta) + b
            rec["train_loss"] = _bce(ztr, y, pos_w)
            rec["train_loss_unweighted"] = _bce(ztr, y, 1.0)
            rec["train_macro_f1"] = macro_f1(y, (ztr > 0).astype(int))
            if A_val is not None:
                zva = w * self.expectation(A_val, theta) + b
                rec["val_loss"] = _bce(zva, y_val, pos_w)
                rec["val_loss_unweighted"] = _bce(zva, y_val, 1.0)
                rec["val_macro_f1"] = macro_f1(y_val, (zva > 0).astype(int))
            hist.append(rec)
            if verbose and (ep % 5 == 4 or ep == epochs - 1):
                print(rec)
        self.result = TrainResult(theta, w, b, hist, time.time() - t0, evals)
        return self.result

    def decision(self, A, theta=None, w=None, b=None):
        r = getattr(self, "result", None)
        theta = r.theta if theta is None else theta
        w = r.w if w is None else w
        b = r.b if b is None else b
        return w * self.expectation(A, theta) + b

    def predict(self, A, theta=None, w=None, b=None):
        return (self.decision(A, theta, w, b) > 0).astype(int)


def macro_f1(y, yhat):
    return f1_score(y, yhat, average="macro")
