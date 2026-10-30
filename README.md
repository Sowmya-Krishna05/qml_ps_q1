# QFF 2026 · Phase 2 · QML track · PS-Q1 (ECG heartbeat classification)

A hybrid variational quantum classifier (VQC) with **6 qubits**, compared with classical baselines on the organiser-supplied ECG data.
Metric that ranks teams: **macro-F1**. Test labels are hidden, so the test set gets predictions only.

## 1. Results (real runs, exact NumPy statevector simulator)

| Model | Split | Accuracy | Precision (Arrhythmia) | Recall (Arrhythmia) | F1 (Arrhythmia) | Macro-F1 | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|
| VQC, 6 qubits | validation (1,500) | 0.9927 | 0.9820 | 1.0000 | 0.9909 | **0.9924** | 888 / 11 / 0 / 601 |
| VQC, 6 qubits | train (6,000) | 0.9915 | 0.9791 | 1.0000 | 0.9894 | 0.9912 | 3565 / 51 / 0 / 2384 |
| Logistic regression, 30 features | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 899 / 0 / 0 / 601 |
| Logistic regression, same 6 inputs | validation | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 899 / 0 / 0 / 601 |

Other validation numbers: ROC-AUC 0.9987, PR-AUC 0.9979, MCC 0.9849, specificity 0.9878, log-loss 0.0326, Brier 0.0073.
Loss falls from 0.236 to 0.038 (class-weighted BCE) over 40 epochs; train and validation curves track each other, so there is no overfitting.
The circuit never misses an arrhythmia (0 false negatives); its only errors are 11 false alarms on validation.
Test predictions: 879 Normal, 621 Arrhythmia; they agree with the 30-feature logistic regression on 99.6% of beats (1,494 of 1,500).
Noise preview: an exact density-matrix replica of the organiser noise model (**our own replica, not Aer**) gives macro-F1 0.980 on 500 validation beats, against 0.994 ideal on the same beats.

Everything above is in `results/` (see section 6). The official Aer numbers (noisy runs, Task A, Task B) are **not produced yet**; they need a run with Qiskit installed (section 7).

## 2. Data and split

Organiser data in `data/q1/`: `ecg_train.csv` (6,000 rows, labelled), `ecg_val.csv` (1,500, labelled), `ecg_test_features.csv` (1,500, no labels).
30 features per beat = 5 windows x 6 statistics (mean, std, min, max, skew, kurtosis). Roughly 60% Normal (0) and 40% Arrhythmia (1); there are no patient IDs.
The data are synthetic ("modelled on" MIT-BIH). Checks run on it: no duplicate rows within or across splits, train/validation/test are statistically indistinguishable, and the classes are nearly linearly separable (a single feature, `w5_mean`, separates the two classes on train). That is why classical baselines reach 1.000.

## 3. Method: how 30 features become 6 qubits

1. **StandardScaler** (fitted on train only).
2. **PCA to 6 components** (fitted on train only), keeping 87.1% of the variance (47.9%, 21.4%, 7.0%, 5.0%, 3.3%, 2.5%).
3. **Angle map**: each component is scaled between its 1st and 99th training percentile to [0, pi] and clipped.
4. **Encoding**: component q becomes `RY(x_q)` on qubit q (one feature per qubit).
5. **Trainable layer**: `RY(theta)` and `RZ(theta)` on every qubit (12 parameters).
6. **Entanglement**: a ring of 6 CX gates (0-1, 1-2, ..., 5-0).
7. **Read-out**: parity `<Z x Z x Z x Z x Z x Z>`, a number f in [-1, 1].
8. **Classical head**: `logit = w*f + b` (w = 14.97, b = 2.03), probability = sigmoid(logit), label = 1 if logit > 0.

Circuit size: 6 qubits, depth 9, 6 CX, 18 one-qubit gates, 12 trainable gates. Architecture drawing: `results/figures/vqc_architecture.png`.

## 4. Training

Class-weighted binary cross-entropy (Arrhythmia weight ~ N_normal / N_abnormal ~ 1.52), Adam, learning rate 0.05, batch 250, 40 epochs, exact parameter-shift gradients, seed 1 (best of seeds 0 and 1 on validation). Training takes ~225 s on 2 CPU cores. Only training data are used for fitting and validation data for choosing the seed; test labels are never used.

## 5. Classical baselines

Logistic regression on the 30 raw features (C = 0.01) and on the same 6 reduced inputs. Both score 1.000 on validation, as the organisers' own baseline does. We therefore claim **no quantum advantage**; the contribution is a compact, noise-aware circuit and an honest evaluation.

## 6. Where everything is

| File | Content |
|---|---|
| `predictions_q1.csv` | Submission file (`sample_id,label`); passes `tools/validate_predictions.py q1` |
| `results/test_samples_with_predictions.csv` | Copy of all 1,500 test samples + `predicted_label`, `predicted_class`, `p_arrhythmia`, `logit` |
| `results/metrics_summary.csv`, `results/extended_metrics.csv` | Accuracy, precision, recall, F1, MCC, ROC-AUC, log-loss, Brier, confusion counts |
| `results/training_history.csv` | Per-epoch loss and macro-F1, train and validation |
| `results/training_summary.json` | All hyper-parameters, trained angles, PCA variance, noise-replica result |
| `results/figures/` | Architecture, training/validation curves, confusion matrices + ROC, decision margins |

## 7. Reproduce

```bash
pip install -r requirements.txt            # pinned versions
python scripts/run_6qubit.py               # trains the 6-qubit model, rewrites results/ and predictions_q1.csv (~5 min, NumPy only)
python tools/validate_predictions.py q1 predictions_q1.csv
python tests/test_sim.py                   # simulator checks
```

`vqc/` holds the simulator, model, outputs, diagram and the Qiskit bridge. `noise_model/`, `benchmark/`, `tools/` and `schemas/` are the organiser kit, unmodified.

## 8. Still to do (not claimed as results)

* Official noisy simulation with `qiskit-aer` and the organiser noise model (every noisy circuit must be transpiled to the fully noised gate set `rx, ry, rz, cx`), with and without error mitigation.
* Task A (Bell-state fidelity from XX/YY/ZZ correlations) and Task B (Processor D design plus the QFT-3 benchmark), which need Qiskit; the code is in `vqc/` and `solution.ipynb`.
* `pip_freeze.txt` and `results.md` from that run. `solution.ipynb` still has to be re-run end to end with the 6-qubit configuration.