"""Problem formulation for PS-Q1: 4-qubit Variational Quantum Classifier (VQC).

Data      : sklearn make_classification (4 features, binary), 200 samples, 80/20 split.
Encoding  : angle encoding, RY(x_i) on qubit i, features scaled to [0, pi].
Ansatz    : L layers of [RY, RZ on every qubit + CX ring 0-1-2-3-0], then a final RY layer.
Readout   : parity of the 4 measured bits.  P(odd parity) = P(class 1).
Loss      : binary cross-entropy on P(class 1).
Optimiser : SPSA (noise-tolerant, 2 loss evaluations per step regardless of #params).
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

SEED = 42
N_QUBITS = 4
N_LAYERS = 2
RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
PARITY = np.array([bin(i).count("1") % 2 for i in range(2 ** N_QUBITS)])


# ----------------------------------------------------------------------------- data
def load_data(save_dir=None):
    """Official PS-Q1 dataset. NOTE: the guide's call make_classification(n_samples=200,
    n_features=4, n_informative=3, random_state=42) raises ValueError with sklearn defaults
    (n_redundant=2 -> 3+2 > 4 features), so we set n_redundant=0 explicitly."""
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                               n_redundant=0, random_state=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(Xtr)      # fit on train only (no leakage)
    Xtr_s, Xte_s = scaler.transform(Xtr), np.clip(scaler.transform(Xte), 0, np.pi)
    if save_dir:
        np.savez(f"{save_dir}/raw/dataset_raw.npz", X=X, y=y)
        np.savez(f"{save_dir}/processed/dataset_split_scaled.npz",
                 X_train=Xtr_s, X_test=Xte_s, y_train=ytr, y_test=yte)
    return Xtr_s, Xte_s, ytr, yte


# ----------------------------------------------------------------------------- circuit
def build_vqc(n_layers=N_LAYERS, measure=False):
    """Returns (circuit, x_params, w_params)."""
    x = ParameterVector("x", N_QUBITS)
    w = ParameterVector("w", 2 * N_QUBITS * n_layers + N_QUBITS)
    qc = QuantumCircuit(N_QUBITS, name="VQC")
    for i in range(N_QUBITS):
        qc.ry(x[i], i)
    k = 0
    for _ in range(n_layers):
        for i in range(N_QUBITS):
            qc.ry(w[k], i); qc.rz(w[k + 1], i); k += 2
        for a, b in RING:
            qc.cx(a, b)
    for i in range(N_QUBITS):
        qc.ry(w[k], i); k += 1
    if measure:
        qc.measure_all()
    return qc, x, w


def bind(circ, x_params, w_params, xi, w):
    d = dict(zip(x_params, xi)); d.update(zip(w_params, w))
    return circ.assign_parameters(d)


def n_weights(n_layers=N_LAYERS):
    return 2 * N_QUBITS * n_layers + N_QUBITS


# ----------------------------------------------------------------------------- ideal model
def ideal_distribution(circ, x_params, w_params, xi, w):
    return Statevector(bind(circ, x_params, w_params, xi, w)).probabilities_dict()


def ideal_p1(circ, x_params, w_params, X, w):
    out = []
    for xi in X:
        p = Statevector(bind(circ, x_params, w_params, xi, w)).probabilities()
        out.append(float(p[PARITY == 1].sum()))
    return np.array(out)


def bce(p1, y, eps=1e-6):
    p1 = np.clip(p1, eps, 1 - eps)
    return float(-np.mean(y * np.log(p1) + (1 - y) * np.log(1 - p1)))


def accuracy(p1, y):
    return float(np.mean((p1 > 0.5).astype(int) == y))


# ----------------------------------------------------------------------------- optimiser
def spsa(loss_fn, w0, iters=80, a=0.8, c=0.2, seed=SEED, batch=None, n_train=None):
    """SPSA. loss_fn(w, idx) -> float, where idx is a sample-index array (or None = all).
    Returns (w_final, history) with history = list of mean(loss+, loss-) per step."""
    rng = np.random.default_rng(seed)
    w = np.array(w0, dtype=float); hist = []
    for k in range(1, iters + 1):
        ak, ck = a / (k + 10) ** 0.602, c / k ** 0.101
        idx = rng.choice(n_train, size=batch, replace=False) if batch else None
        delta = rng.choice([-1.0, 1.0], size=w.shape)
        lp, lm = loss_fn(w + ck * delta, idx), loss_fn(w - ck * delta, idx)
        w = w - ak * (lp - lm) / (2 * ck) * delta
        hist.append(0.5 * (lp + lm))
    return w, hist
