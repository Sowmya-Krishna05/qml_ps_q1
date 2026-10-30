"""Circuit metrics: depth, CX count, SWAP overhead, distribution fidelity."""
import numpy as np
from qiskit.quantum_info import hellinger_fidelity
from .routing import transpile_logical, transpile_keep_swaps


def circuit_metrics(tc):
    ops = tc.count_ops()
    return dict(depth=tc.depth(), size=tc.size(), cx=ops.get("cx", 0),
                one_qubit=ops.get("rz", 0) + ops.get("sx", 0) + ops.get("x", 0))


def swap_overhead(circ, proc):
    """Direct SWAP count from routing + CX overhead vs the all-to-all baseline."""
    routed = transpile_keep_swaps(circ, proc)
    n_swap = routed.count_ops().get("swap", 0)
    cx_log = transpile_logical(circ).count_ops().get("cx", 0)
    return dict(swaps=n_swap, cx_logical=cx_log, cx_overhead_from_swaps=3 * n_swap)


def mean_hellinger(ideal_dists, noisy_counts):
    return float(np.mean([hellinger_fidelity(i, n) for i, n in zip(ideal_dists, noisy_counts)]))
