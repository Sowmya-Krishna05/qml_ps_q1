"""Qiskit side of the project: build the circuit from the op list and run it under the
ORGANISER noise model (noise_model/benchmark_noise_model.py, unchanged).

Every noisy run goes through `prepare_for_noisy_run()` (transpile to the fully-noised basis
rx, ry, rz, cx at optimisation_level=1, seed 42; raises if any un-noised gate remains).
The circuit is transpiled once with symbolic parameters and the sample values are bound
afterwards, so all samples share one compiled structure (no gate is dropped or merged
because of a particular angle).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, Statevector

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("noise_model", "benchmark", "tools"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_noise_model import (assert_all_gates_noisy, noisy_simulator,  # noqa: E402
                                   prepare_for_noisy_run, SEED)
from qiskit_aer import AerSimulator  # noqa: E402


# ----------------------------------------------------------------------------- building
def build_circuit(ops, n_qubits, n_features, n_params, readout=(0,), measure=True, fold=1):
    """Parametrised circuit. x[k] = data angle k, t[k] = trainable angle k.

    fold = 1, 3, 5, ... : global unitary folding U (U^dagger U)^((fold-1)/2) for zero-noise
    extrapolation. Folding only repeats gates already in the noised basis.
    """
    x, t = ParameterVector("x", n_features), ParameterVector("t", n_params)
    qc = QuantumCircuit(n_qubits, len(readout) if measure else 0)

    def angle(ref):
        return x[ref[1]] if ref[0] == "x" else t[ref[1]]

    def forward(inverse=False):
        seq = reversed(ops) if inverse else ops
        for gate, qs, ref in seq:
            if gate == "cx":
                qc.cx(*qs)
            else:
                a = angle(ref)
                getattr(qc, gate)(-a if inverse else a, qs[0])

    forward()
    for _ in range((fold - 1) // 2):
        qc.barrier()                 # keep the transpiler from cancelling U^dagger U
        forward(inverse=True)
        qc.barrier()
        forward()
    if measure:
        for i, q in enumerate(readout):
            qc.measure(q, i)
    return qc, x, t


def z_string_from_counts(counts: dict):
    """<Z...Z> over all measured bits (parity) and its standard error."""
    n = sum(counts.values())
    even = sum(c for k, c in counts.items() if k.replace(" ", "").count("1") % 2 == 0)
    e = (2 * even - n) / n
    return e, float(np.sqrt(max(1 - e * e, 0.0) / n))


# ----------------------------------------------------------------------------- ideal reference
def ideal_expectations(ops, n_qubits, n_features, n_params, readout, A, theta):
    """Exact Qiskit Statevector <Z-string> (used to cross-check the NumPy simulator)."""
    qc, x, t = build_circuit(ops, n_qubits, n_features, n_params, readout, measure=False)
    label = ["I"] * n_qubits
    for q in readout:
        label[n_qubits - 1 - q] = "Z"            # Qiskit labels are little-endian
    obs = Pauli("".join(label))
    out = []
    for row in A:
        bound = qc.assign_parameters({**dict(zip(x, row)), **dict(zip(t, theta))})
        out.append(Statevector(bound).expectation_value(obs).real)
    return np.array(out)


# ----------------------------------------------------------------------------- noisy runs
def noisy_expectations(ops, n_qubits, n_features, n_params, readout, A, theta,
                       shots=4096, seed=SEED, fold=1, chunk=250, ideal=False,
                       coupling_map=None, basis_gates=None):
    """<Z-string> per sample from shot-based Aer runs.

    ideal=False : official noise model QFF2026-v2 on the fully-noised basis (the reported numbers).
    ideal=True  : same compiled circuits on a noise-free AerSimulator (shot noise only).
    Returns (values, standard_errors, compiled_circuit).
    """
    qc, x, t = build_circuit(ops, n_qubits, n_features, n_params, readout, fold=fold)
    tqc = prepare_for_noisy_run(qc, basis_gates=basis_gates, coupling_map=coupling_map)
    assert_all_gates_noisy(tqc)
    sim = AerSimulator(seed_simulator=seed) if ideal else noisy_simulator(seed)
    vals, ses = [], []
    for s in range(0, len(A), chunk):
        batch = [tqc.assign_parameters({**dict(zip(x, row)), **dict(zip(t, theta))})
                 for row in A[s:s + chunk]]
        res = sim.run(batch, shots=shots).result()
        for i in range(len(batch)):
            e, se = z_string_from_counts(res.get_counts(i))
            vals.append(e); ses.append(se)
    return np.array(vals), np.array(ses), tqc


def readout_calibration(n_qubits, readout, shots=8192, seed=SEED):
    """Per-qubit symmetric flip rate estimated from |0..0> and |1..1> preparation circuits that are
    run through the SAME noisy simulator (the standard calibration-matrix idea, 2 extra circuits)."""
    rates = []
    sim = noisy_simulator(seed)
    for q in readout:
        c0 = QuantumCircuit(n_qubits, 1); c0.measure(q, 0)
        c1 = QuantumCircuit(n_qubits, 1); c1.x(q); c1.measure(q, 0)
        counts = sim.run([prepare_for_noisy_run(c0), prepare_for_noisy_run(c1)],
                         shots=shots).result()
        p01 = counts.get_counts(0).get("1", 0) / shots          # prepared 0, read 1
        p10 = counts.get_counts(1).get("0", 0) / shots          # prepared 1, read 0
        rates.append((p01, p10))
    # the X gate is itself noisy (p=0.01 depolarising): the |1> estimate is slightly biased; we
    # therefore use the |0> estimate (no gate before measurement) as the flip rate.
    return rates


def compile_stats(tqc):
    ops = tqc.count_ops()
    return {"depth": tqc.depth(),
            "two_qubit_gates": sum(1 for i in tqc.data if i.operation.num_qubits == 2
                                   and i.operation.name != "barrier"),
            "gates": {k: int(v) for k, v in ops.items() if k not in ("measure", "barrier")}}
