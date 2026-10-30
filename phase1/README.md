# QFF 2026 Phase 1 — PS-Q1: Variational Quantum Classifier on Processor A vs B

**Problem statement selected: PS-Q1 (Quantum Machine Learning — Variational Quantum Classifier).**
Qiskit Fall Fest 2026 · SRM University-AP · Phase 1 (Processor A & B only).

## Team Nova — SRM University-AP
| Member | Role |
|---|---|
| Sowmya Balagam | Team leader |
| Rajrishi Akana | Member |


## What this repo does
A 4-qubit VQC (angle encoding → 2 variational layers with a CX ring → parity readout) is trained on the official synthetic dataset, then compiled and benchmarked on two **participant-designed** processors:

| | Processor A | Processor B |
|---|---|---|
| Qubits | 5 | 9 |
| Coupling map | T-tree `(0,1),(1,2),(1,3),(3,4)` | ring-core + bridges + tail `(0,1),(1,2),(2,3),(3,0),(1,4),(4,5),(5,2),(3,6),(6,7),(7,8)` |
| Idea | sparse, acyclic baseline → ring CX must be routed | native 4-cycle for the ring ansatz, degree ≤ 3, extra cycle for routing freedom |
| Basis / transpile / noise | `{rz,sx,x,cx}`, opt-level 1, seed 42, 4096 shots, 1q 0.1 %, CX 0.5 %, readout 1 % (identical on both) |

Coupling maps are our own designs (the guide's maps are reference-only); the guide's reference linear chain and heavy-hex are included in an ablation.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace main.ipynb   # ~1-2 min on CPU
```
The notebook regenerates every table in `results/tables/` and figure in `results/figures/`. Everything is seeded (42).

## Key results (test set = 40 samples)
| | Depth | CX | SWAPs | Noisy test acc | Hellinger fidelity |
|---|---|---|---|---|---|
| Ideal (exact) | – | – | – | 0.925 | 1.000 |
| **Processor A** | 43 | 17 | 3 | 0.900 | 0.991 |
| **Processor B** | 25 | 8 | 0 | 0.900 | 0.994 |

* Processor B compiles the VQC with **zero SWAPs** (CX = 8 = logical minimum, depth 25 = all-to-all depth); Processor A needs 3 SWAPs → **+9 CX, +72 % depth**.
* With only 0.5 % CX error the accuracy impact is small and **not statistically distinguishable** at n = 40 (1 sample = 2.5 pts); the clearer signal is in circuit cost and output-distribution fidelity.
* **Classical reference** (same split): logistic regression 0.875, RBF-SVM 0.850 test accuracy vs VQC 0.925 (ideal). The VQC's *train* accuracy (0.869) is below the classical models (0.925 / 0.950), so the higher test score is within noise on 40 samples and is **not** a claim of quantum advantage.
* Noisy test accuracy 95 % bootstrap CI: A and B both 0.90 [0.80, 0.975] — indistinguishable.
* Full analysis, ablation and limitations: [`report.md`](report.md).

## Layout
```
README.md  main.ipynb  requirements.txt  report.md
data/{raw,processed}/        generated dataset + scaled split
processors/processor_{A,B}.json
src/{problem,processors,routing,metrics,visualization}.py
results/{tables,figures}/
```

## Notes & open items
* The guide's dataset call `make_classification(n_samples=200, n_features=4, n_informative=3, random_state=42)` raises `ValueError` in current scikit-learn (default `n_redundant=2`). We use `n_redundant=0`.
* The Phase 1 guide describes PS-Q1 as the synthetic-data VQC; the PS-Q1 sheet mentions a "Biomedical Signal / Arrhythmia" framing with details in the Drive folder. **Check that Drive folder for any extra required dataset, metric or ansatz and adapt `src/problem.py` (only `load_data` and the ansatz need to change; the A/B benchmarking code is generic).**
