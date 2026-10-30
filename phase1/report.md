# Report — PS-Q1 VQC: Algorithm ↔ Architecture Co-Design (Processor A vs B)

## 1. Setup
Data: `make_classification(200, 4 features, 3 informative, n_redundant=0, seed 42)`, stratified 80/20, features min-max scaled to [0, π] on training statistics. Model: RY angle encoding; 2 × [RY, RZ per qubit + CX ring 0-1-2-3-0]; final RY layer (20 weights); output = parity of 4 measured bits = P(class 1); BCE loss; SPSA. Compilation: opt-level 1, seed 42, basis {rz, sx, x, cx}; noise: 1q depolarizing 0.1 %, CX 0.5 %, readout 1 %, 4096 shots, same on both processors.

## 2. Processor design
The ansatz entangles in a 4-cycle, so the core design question is *does the processor contain a 4-cycle?*
* **A (5q T-tree)** – acyclic, so the closing CX(3,0) is never native; it forces routing. It is a deliberately constrained baseline.
* **B (9q ring-core)** – contains the 4-cycle {0,1,2,3} natively, plus a second cycle (1-4-5-2) and a tail (3-6-7-8). Max degree 3, 10 edges: sparse like real heavy-hex-style devices, not a fully connected idealisation.

## 3. Results (from `results/tables/`)
| Metric | A | B | all-to-all |
|---|---|---|---|
| Depth | 43 | 25 | 25 |
| CX count | 17 | 8 | 8 |
| SWAPs (direct count) | 3 (= +9 CX) | 0 | 0 |
| Noisy test accuracy (ideal-trained weights) | 0.900 | 0.900 | ideal 0.925 |
| Mean \|ΔP(class 1)\| vs ideal | 0.0237 | 0.0166 | 0 |
| Hellinger fidelity (4-bit distribution) | 0.9912 | 0.9940 | 1 |
| Noisy-in-the-loop training: test acc | 0.850 | 0.875 | – |
| Noisy-in-the-loop training: mean loss, first 5 → last 5 iters | 0.713 → 0.639 | 0.716 → 0.641 | – |

**Classical reference** (`classical_baseline.csv`): logistic regression 0.875 / SVM-RBF 0.850 test accuracy (train 0.925 / 0.950) vs VQC ideal 0.925 (train 0.869). On a 40-sample test set this gap is within sampling noise; the VQC is competitive with, not demonstrably better than, classical models.

**Noisy accuracy 95 % bootstrap CI** (`noisy_accuracy_ci.csv`): A 0.90 [0.80, 0.975]; B 0.90 [0.80, 0.975].

**Topology ablation** (same circuit, `topology_ablation.csv`): linear-5 → 6 SWAPs, CX 24, depth 44; T-tree-5 → 3 SWAPs, CX 17, depth 43; ring-5 → 2 SWAPs, CX 14, depth 31; heavy-hex-9 (guide reference) → 0 SWAPs, CX 8, depth 25; ring-core-9 → 0 SWAPs, CX 8, depth 25.

## 4. Interpretation
1. **Connectivity, not qubit count, drives cost.** Going from 5 to 9 qubits only helps because the extra qubits create a 4-cycle; ring-5 (5 qubits, a 5-cycle with no 4-cycle) still needs 2 SWAPs. Edge *structure* matching the ansatz matters most.
2. **Routing cost is large and deterministic**: A pays +9 CX and +18 depth layers, i.e. 2.1× the CX count. Under p_CX = 0.5 % that is ≈ 4.5 % extra cumulative CX error budget.
3. **Noise effect is measurable in distribution fidelity and probability shift** (A: 0.0237 vs B: 0.0166 mean |ΔP1|; Hellinger 0.991 vs 0.994) but **accuracy differences are within sampling resolution** (test n = 40; one sample = 2.5 pts). We do not claim an accuracy advantage for B.
4. Noisy-in-the-loop training converged similarly on both (≈ 0.71 → 0.64 loss). The 0.850 vs 0.875 test accuracy is one sample; with only 40 SPSA steps, 16-sample batches and 1024 shots this is a convergence-trend comparison, not a final-performance claim.

## 5. Limitations
* Small test set (40) and a single seed; no confidence intervals. Multi-seed repeats would tighten the accuracy comparison.
* Synthetic uniform depolarizing noise has no qubit-dependent errors, crosstalk or T1/T2 idle decoherence, so deeper circuits are penalised only through gate count, which probably understates A's disadvantage.
* Opt-level 1 and a single transpiler seed; layout/routing is heuristic (SABRE) and could differ with other seeds.
* The model is shallow (2 layers); deeper ansätze would widen the A/B gap.
* Dataset is the Phase 1 guide's synthetic set; domain-specific biosignal features (see README open items) are not used.

## 6. Reproducibility
Everything is seeded (42); versions pinned in `requirements.txt`; `main.ipynb` regenerates all tables/figures.
