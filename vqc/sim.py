"""NumPy simulators for small circuits (<= ~8 qubits), batched over data samples.

Circuits are *op lists* so that exactly the same description feeds
  (1) this fast simulator (used for training and for tests),
  (2) the Qiskit builder in vqc/qiskit_bridge.py (used for the official noisy runs).

op = (gate, qubits, ref)      gate in {"rx","ry","rz","cx"}
ref = None                    for cx
    = ("x", k)                angle = scale * x[:, k]  (data re-uploading)
    = ("t", k)                angle = theta[k]          (trainable)

Conventions are the standard ones used by Qiskit and PennyLane:
  R_P(a) = exp(-i a P / 2),  CX(control, target),  qubit q <-> tensor axis q.

`DensityMatrixSim` is a *development replica* of the organiser noise model
QFF2026-v2 (depolarising after every gate with lambda = p, symmetric readout flip).
It is used to pick designs quickly; every number that is reported as "noisy" in the
submission comes from qiskit-aer with the official model (see vqc/qiskit_bridge.py).
"""
from __future__ import annotations

import numpy as np

P1, P2, P_READOUT = 0.01, 0.05, 0.02   # organiser values (noise_model/benchmark_noise_model.py)


# ----------------------------------------------------------------------------- gates
def _rot(gate: str, a: np.ndarray) -> np.ndarray:
    """Return U of shape (B,2,2) (or (2,2) if a is scalar)."""
    a = np.asarray(a, dtype=float)
    c, s = np.cos(a / 2), np.sin(a / 2)
    z = np.zeros_like(c)
    if gate == "rx":
        u = np.stack([np.stack([c + 0j, -1j * s], -1), np.stack([-1j * s, c + 0j], -1)], -2)
    elif gate == "ry":
        u = np.stack([np.stack([c + 0j, -s + 0j], -1), np.stack([s + 0j, c + 0j], -1)], -2)
    elif gate == "rz":
        e = np.exp(-0.5j * a)
        u = np.stack([np.stack([e, z + 0j], -1), np.stack([z + 0j, np.conj(e)], -1)], -2)
    else:
        raise ValueError(gate)
    return u


def op_angle(ref, X, theta, scale):
    kind, k = ref
    if kind == "x":
        return scale * X[:, k]
    return np.asarray(theta[k], dtype=float)


# ----------------------------------------------------------------------------- statevector
class StatevectorSim:
    def __init__(self, n_qubits: int):
        self.n = n_qubits

    def run(self, ops, X, theta, scale=1.0) -> np.ndarray:
        """Return state of shape (B,) + (2,)*n."""
        n, B = self.n, len(X)
        s = np.zeros((B,) + (2,) * n, dtype=complex)
        s[(slice(None),) + (0,) * n] = 1.0
        for gate, qs, ref in ops:
            if gate == "cx":
                s = self._cx(s, *qs)
            else:
                s = self._apply1(s, qs[0], _rot(gate, op_angle(ref, X, theta, scale)))
        return s

    def _apply1(self, s, q, U):
        t = np.moveaxis(s, q + 1, -1)
        t = t @ U.T if U.ndim == 2 else np.einsum("b...i,bji->b...j", t, U)
        return np.moveaxis(t, -1, q + 1)

    def _cx(self, s, c, t):
        out = s.copy()
        sel = [slice(None)] * (self.n + 1)
        sel[c + 1] = 1
        ax = t + 1 if t < c else t
        out[tuple(sel)] = np.flip(s[tuple(sel)], axis=ax)
        return out

    def expz(self, s, qubits) -> np.ndarray:
        """<Z_q1 Z_q2 ...> for the given qubit tuple."""
        p = (np.abs(s) ** 2).reshape(len(s), -1)
        idx = np.arange(p.shape[1])
        sign = np.ones(p.shape[1])
        for q in qubits:
            bit = (idx >> (self.n - 1 - q)) & 1       # axis q is the q-th most significant
            sign = sign * (1 - 2 * bit)
        return p @ sign

    def unitary_probe(self, ops, X, theta, scale=1.0):
        return self.run(ops, X, theta, scale)


# ----------------------------------------------------------------------------- density matrix + noise
class DensityMatrixSim:
    """Batched density-matrix simulation with the QFF2026-v2 noise structure."""

    def __init__(self, n_qubits: int, p1=P1, p2=P2, p_readout=P_READOUT):
        self.n, self.p1, self.p2, self.pr = n_qubits, p1, p2, p_readout

    def run(self, ops, X, theta, scale=1.0, noisy=True) -> np.ndarray:
        n, B = self.n, len(X)
        rho = np.zeros((B,) + (2,) * (2 * n), dtype=complex)
        rho[(slice(None),) + (0,) * (2 * n)] = 1.0
        for gate, qs, ref in ops:
            if gate == "cx":
                rho = self._cx(rho, *qs)
                if noisy:
                    rho = self._depol(rho, qs, self.p2)
            else:
                U = _rot(gate, op_angle(ref, X, theta, scale))
                rho = self._apply1(rho, qs[0], U)
                if noisy:
                    rho = self._depol(rho, qs, self.p1)
        return rho

    def _apply1(self, rho, q, U):
        n = self.n
        r = np.moveaxis(rho, q + 1, -1)
        r = r @ U.T if U.ndim == 2 else np.einsum("b...i,bji->b...j", r, U)
        rho = np.moveaxis(r, -1, q + 1)
        r = np.moveaxis(rho, n + q + 1, -1)
        Uc = U.conj()
        r = r @ Uc.T if U.ndim == 2 else np.einsum("b...i,bji->b...j", r, Uc)
        return np.moveaxis(r, -1, n + q + 1)

    def _cx(self, rho, c, t):
        n = self.n
        for off in (0, n):                                  # rows then columns
            out = rho.copy()
            sel = [slice(None)] * (2 * n + 1)
            sel[off + c + 1] = 1
            sub = rho[tuple(sel)]
            ax = off + t + 1 if (off + t) < (off + c) else off + t
            out[tuple(sel)] = np.flip(sub, axis=ax)
            rho = out
        return rho

    def _depol(self, rho, qs, lam):
        """rho -> (1-lam) rho + lam * I/d (x) Tr_qs(rho)   (Qiskit depolarizing_error(lam, k))."""
        n, k = self.n, len(qs)
        d = 2 ** k
        row_axes = [q + 1 for q in qs]
        col_axes = [n + q + 1 for q in qs]
        t = rho
        # trace out: take diagonal over matching row/col axes
        t = np.einsum(self._trace_subscripts(n, qs), rho)
        out = (1 - lam) * rho
        for bits in np.ndindex(*(2,) * k):
            sel = [slice(None)] * (2 * n + 1)
            for q, b in zip(qs, bits):
                sel[q + 1] = b
                sel[n + q + 1] = b
            out[tuple(sel)] += (lam / d) * t
        return out

    @staticmethod
    def _trace_subscripts(n, qs):
        letters = "abcdefghijklmnopqrstuvwxyz"
        B = "B"
        row = [letters[i] for i in range(n)]
        col = [letters[n + i] for i in range(n)]
        for q in qs:
            col[q] = row[q]
        out = [row[i] for i in range(n) if i not in qs] + [col[i] for i in range(n) if i not in qs]
        return f"{B}{''.join(row)}{''.join(col)}->{B}{''.join(out)}"

    def expz(self, rho, qubits, readout=True) -> np.ndarray:
        """<Z...> from the diagonal, with the symmetric readout flip folded in exactly."""
        B, n = len(rho), self.n
        dim = 2 ** n
        diag = np.einsum("bii->bi", rho.reshape(B, dim, dim)).real
        idx = np.arange(dim)
        sign = np.ones(dim)
        for q in qubits:
            sign = sign * (1 - 2 * ((idx >> (n - 1 - q)) & 1))
        val = diag @ sign
        return val * ((1 - 2 * self.pr) ** len(qubits) if readout else 1.0)


def sample_expectation(exp: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Shot-noise emulation of a +-1 observable with mean `exp`."""
    p_plus = np.clip((1 + exp) / 2, 0, 1)
    return 2 * rng.binomial(shots, p_plus) / shots - 1
