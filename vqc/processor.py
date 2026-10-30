"""Processor D candidates and helpers (Task B). All measurements use the OFFICIAL code in
benchmark/qff_qml_benchmark.py; nothing here re-implements the benchmark."""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("noise_model", "benchmark"):
    sys.path.insert(0, os.path.join(_ROOT, sub))

from qiskit import transpile  # noqa: E402
from qff_qml_benchmark import (_coupling, _routing_swaps, task_b_qft3,  # noqa: E402
                               validate_processor_d)
from benchmark_noise_model import SEED  # noqa: E402


def both(edges):
    """Two-way links: [a,b] and [b,a] for every undirected edge."""
    out = []
    for a, b in edges:
        out += [[a, b], [b, a]]
    return out


GATESETS = {
    "rx,ry,rz,cx": ["rx", "ry", "rz", "cx"],
    "rx,ry,rz,h,cx": ["rx", "ry", "rz", "h", "cx"],
    "rz,ry,h,s,sdg,t,tdg,cx": ["rz", "ry", "h", "s", "sdg", "t", "tdg", "cx"],
    "rx,ry,rz,cz": ["rx", "ry", "rz", "cz"],
    "rz,sx,x,cx (Phase-1 basis)": ["rz", "sx", "x", "cx"],
}

# name -> (n_qubits, coupling_map, layout)
TOPOLOGIES = {
    "triangle-3 (two-way)": (3, both([(0, 1), (1, 2), (0, 2)]), "ring"),
    "triangle-3 (one-way ring)": (3, [[0, 1], [1, 2], [2, 0]], "ring"),
    "line-3 (two-way)": (3, both([(0, 1), (1, 2)]), "linear"),
    "ring-4 (two-way)": (4, both([(0, 1), (1, 2), (2, 3), (3, 0)]), "ring"),
    "diamond-4 (ring + chord 0-2)": (4, both([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]), "custom"),
    "line-5 (kit example)": (5, both([(0, 1), (1, 2), (2, 3), (3, 4)]), "linear"),
    "Phase-1 Processor A (T-tree, 5q)": (5, both([(0, 1), (1, 2), (1, 3), (3, 4)]), "custom"),
    "Phase-1 Processor B (ring core + tail, 9q)": (9, both([(0, 1), (1, 2), (2, 3), (3, 0), (1, 4), (4, 5), (5, 2), (3, 6), (6, 7), (7, 8)]), "custom"),
}


def make_config(name, topo, gates, **extra):
    n, cmap, layout = TOPOLOGIES[topo]
    cfg = {"processor_id": name, "n_qubits": n, "coupling_map": cmap,
           "native_gate_set": GATESETS[gates], "qubit_layout": layout,
           "noise_assumptions": {"single_qubit_error_rate": 0.001, "two_qubit_error_rate": 0.01,
                                 "readout_error_rate": 0.02, "notes": "placeholder"},
           "design_rationale": "x" * 60}
    cfg.update(extra)
    return cfg


def evaluate(cfg, vqc_logical=None):
    """Official QFT-3 benchmark (+ compilation cost of our VQC on the same processor)."""
    probs = validate_processor_d(cfg, require_results=False)
    if probs:
        return {"valid": False, "problems": probs}
    res = task_b_qft3(cfg)
    out = {"valid": True, **res}
    if vqc_logical is not None:
        cm, basis = _coupling(cfg), list(cfg["native_gate_set"])
        out["vqc_logical_depth"] = vqc_logical.depth()
        out["vqc_logical_two_qubit_gates"] = sum(1 for i in vqc_logical.data if i.operation.num_qubits == 2)
        if vqc_logical.num_qubits > cfg["n_qubits"]:       # processor too small for our circuit
            out.update({"vqc_depth": None, "vqc_two_qubit_gates": None, "vqc_swaps": None})
        else:
            t = transpile(vqc_logical, coupling_map=cm, basis_gates=basis, optimization_level=1,
                          seed_transpiler=SEED)
            out.update({"vqc_depth": t.depth(),
                        "vqc_two_qubit_gates": sum(1 for i in t.data if i.operation.num_qubits == 2),
                        "vqc_swaps": _routing_swaps(vqc_logical, cm, basis)})
    return out


def save_config(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
