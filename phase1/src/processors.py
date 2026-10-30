"""Processor A / B definitions (participant-designed), JSON I/O and the shared noise model.

Design rationale
----------------
The VQC entangler is a 4-qubit CX *ring* (0-1-2-3-0), so the question this benchmark asks is:
"what does a topology that lacks / contains a 4-cycle cost us?"

Processor A (5q) - "T-tree" : [(0,1),(1,2),(1,3),(3,4)]. Sparse, acyclic, degree-3 hub at q1.
    Acyclic => the closing ring CX(3,0) can never be native => guaranteed SWAP overhead.
Processor B (9q) - "Ring-core + bridges" : a native 4-cycle {0,1,2,3} (the ring ansatz embeds
    with ZERO swaps), a second 4-cycle through bridge qubits 4,5 (1-4-5-2) giving routing
    alternatives, and a 3-qubit tail 3-6-7-8 (heavy-hex-like low degree, max degree 3).
    10 edges, connected, degree <= 3 (realistic for fixed-frequency transmon chips).
Both use basis {rz, sx, x, cx} and the SAME noise rates, so differences isolate connectivity.
"""
import json
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

BASIS = ["rz", "sx", "x", "cx"]
P1Q, P2Q, PRO = 0.001, 0.005, 0.01

PROCESSORS = {
    "A": dict(name="Processor A (T-tree, 5q)", n_qubits=5,
              coupling_map=[(0, 1), (1, 2), (1, 3), (3, 4)],
              rationale="Sparse acyclic baseline; no 4-cycle so ring CX must be routed."),
    "B": dict(name="Processor B (ring-core + bridges + tail, 9q)", n_qubits=9,
              coupling_map=[(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2),
                            (3, 6), (6, 7), (7, 8)],
              rationale="Native 4-cycle for the ring ansatz, second cycle via bridges, low-degree tail."),
}

# Extra topologies used only for the ablation study (not the official A/B).
ABLATION = {
    "Linear-5 (ref. A)": dict(n_qubits=5, coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)]),
    "T-tree-5 (A)": dict(PROCESSORS["A"]),
    "Ring-5": dict(n_qubits=5, coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
    "Heavy-hex-9 (ref. B)": dict(n_qubits=9, coupling_map=[(0, 1), (1, 2), (1, 6), (2, 3), (2, 6), (2, 5),
                                                         (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (8, 0)]),
    "Ring-core-9 (B)": dict(PROCESSORS["B"]),
}


def to_cmap(proc):
    """Bidirectional CouplingMap from an undirected edge list."""
    edges = [tuple(e) for e in proc["coupling_map"]]
    return CouplingMap(edges + [(b, a) for a, b in edges])


def build_noise_model(p1=P1Q, p2=P2Q, pro=PRO):
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - pro, pro], [pro, 1 - pro]]))
    return nm


def save_processor_json(key, path):
    p = PROCESSORS[key]
    cfg = dict(p, basis_gates=BASIS, transpile=dict(optimization_level=1, seed_transpiler=42, shots=4096),
               noise=dict(depolarizing_1q=P1Q, depolarizing_cx=P2Q, readout=PRO))
    cfg["coupling_map"] = [list(e) for e in cfg["coupling_map"]]
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_processor(path):
    with open(path) as f:
        return json.load(f)
