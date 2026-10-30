"""Transpilation, layout/routing and noisy execution."""
from qiskit import transpile
from qiskit_aer import AerSimulator
from .processors import BASIS, to_cmap, build_noise_model
from .problem import bind

OPT, SEED, SHOTS = 1, 42, 4096


def transpile_for(circ, proc):
    """Transpile onto a processor: Opt level 1, seed 42, basis {rz,sx,x,cx}."""
    return transpile(circ, coupling_map=to_cmap(proc), basis_gates=BASIS,
                     optimization_level=OPT, seed_transpiler=SEED)


def transpile_logical(circ):
    """All-to-all reference (no coupling map) used as the zero-routing baseline."""
    return transpile(circ, basis_gates=BASIS, optimization_level=OPT, seed_transpiler=SEED)


def transpile_keep_swaps(circ, proc):
    """Same routing but keeps 'swap' as a gate so SWAPs can be counted directly."""
    return transpile(circ, coupling_map=to_cmap(proc), basis_gates=BASIS + ["swap"],
                     optimization_level=OPT, seed_transpiler=SEED)


def make_sim(noisy=True):
    return AerSimulator(noise_model=build_noise_model()) if noisy else AerSimulator()


def run_counts(tcirc, x_params, w_params, X, w, sim, shots=SHOTS, seed=SEED):
    """Run one bound copy of the (transpiled, measured) template per sample; list of count dicts."""
    circs = [bind(tcirc, x_params, w_params, xi, w) for xi in X]
    res = sim.run(circs, shots=shots, seed_simulator=seed).result()
    return [res.get_counts(i) for i in range(len(circs))]


def counts_p1(counts):
    tot = sum(counts.values())
    return sum(v for k, v in counts.items() if bin(int(k, 2)).count("1") % 2 == 1) / tot
