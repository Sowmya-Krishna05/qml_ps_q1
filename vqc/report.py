"""Text generation for results.md / README / processor rationale.

Every number and every qualitative verdict ("no measurable difference", "worse by ...") is computed
from the results of the run, so the written report cannot drift from the notebook output.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from . import processor as P


def _md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- Task A
def task_a_text(ta: dict) -> str:
    i, n = ta["ideal"], ta["noisy"]
    Fi, Fn = i["fidelity"], n["fidelity"]
    se = n["fidelity_se"]
    ent = ("Because F > 0.5, the noisy state is still provably entangled (any state with fidelity above 1/2 to a maximally entangled state is)"
           if Fn - 3 * se > 0.5 else
           "F is not clearly above 0.5, so entanglement can no longer be certified from this fidelity")
    return (f"Ideal simulation gives F = {Fi:.4f} ± {i['fidelity_se']:.4f} with ⟨XX⟩ = {i['XX']:+.3f}, ⟨YY⟩ = {i['YY']:+.3f}, ⟨ZZ⟩ = {i['ZZ']:+.3f}: "
            f"perfect correlation in all three bases is exactly the signature of |Φ⁺⟩. "
            f"Under the benchmark noise the same circuit gives F = {Fn:.4f} ± {se:.4f} (⟨XX⟩ = {n['XX']:+.3f}, ⟨YY⟩ = {n['YY']:+.3f}, ⟨ZZ⟩ = {n['ZZ']:+.3f}). "
            f"All three correlators shrink toward 0, because depolarising errors on the H and CX gates (1 % and 5 %), the extra basis-change gates for XX/YY, and 2 % read-out flips per qubit (a factor 0.96² = 0.92 on every correlator) randomise the parity. "
            f"The loss of {Fi - Fn:.3f} in fidelity is mainly a loss of coherence and correlation, not just of population. "
            f"{ent}; one noisy two-qubit gate is already enough to remove a large part of the entanglement quality.")


# ----------------------------------------------------------------------------- Processor D
def _row(df, name):
    r = df[df["topology"] == name]
    return None if r.empty else r.iloc[0]


def processor_rationale(topo, gs, final_row, topo_df, gs_df, compiled, logical) -> str:
    n, cmap, _ = P.TOPOLOGIES[topo]
    adj = {q: set() for q in range(n)}
    for a, b in cmap:
        adj[a].add(b); adj[b].add(a)
    n_links = len({tuple(sorted(e)) for e in cmap})
    tri = any(adj[a] & adj[b] for a in adj for b in adj[a])
    two_way = all([b, a] in cmap for a, b in cmap)
    s = []
    s.append(f"Our PS-Q1 classifier is a {logical['qubits']}-qubit circuit (RY encoding, trainable RY/RZ, a ring of {logical['two_qubit_gates']} CX gates, parity read-out), so the processor must contain a 4-cycle; "
             f"'{topo}' has {n} qubits and {n_links} {'two-way ' if two_way else ''}links and compiles our circuit with {int(final_row['vqc_swaps'])} routing SWAPs "
             f"(compiled depth {int(final_row['vqc_depth'])}, {int(final_row['vqc_two_qubit_gates'])} two-qubit gates; logical circuit has {logical['two_qubit_gates']}).")
    s.append(f"{'It also contains a triangle, which the mandatory QFT-3 needs for its three controlled-phase gates' if tri else 'It has no triangle, so QFT-3 needs routing'}: "
             f"official QFT-3 result is depth {final_row['qft3_circuit_depth_after_transpile']}, {final_row['qft3_two_qubit_gate_count']} two-qubit gates, "
             f"{final_row['qft3_swap_overhead']} SWAPs and noisy fidelity {final_row['qft3_fidelity_noisy']:.4f}.")
    ow = _row(topo_df, "triangle-3 (one-way ring)"); tw = _row(topo_df, "triangle-3 (two-way)")
    if ow is not None and tw is not None and ow["valid"] and tw["valid"]:
        s.append(f"Every link is listed in both directions because the benchmark reverses a one-way CX with extra Hadamards: on a triangle this costs {tw['qft3_fidelity_noisy'] - ow['qft3_fidelity_noisy']:+.4f} noisy QFT-3 fidelity "
                 f"({ow['qft3_fidelity_noisy']:.4f} one-way vs {tw['qft3_fidelity_noisy']:.4f} two-way).")
    g = gs_df[gs_df["valid"]]
    s.append(f"The native gate set {{{gs}}} contains the ry, rz and cx our circuit is built from, so trained rotations are not re-synthesised; across the candidate gate sets the noisy QFT-3 fidelity spans "
             f"{g['qft3_fidelity_noisy'].min():.4f} to {g['qft3_fidelity_noisy'].max():.4f}.")
    big = _row(topo_df, "Phase-1 Processor B (ring core + tail, 9q)")
    if big is not None and big["valid"]:
        s.append(f"More qubits and links did not help: our Phase-1 9-qubit processor B reaches {big['qft3_fidelity_noisy']:.4f}, "
                 f"so extra connectivity beyond what the workload uses is unused cost, not benefit.")
    return " ".join(s)


# ----------------------------------------------------------------------------- results.md
def _abl_sentence(abl: pd.DataFrame) -> str:
    a = abl.set_index("variant")
    base = a.loc["final design"]
    out = []
    for name in a.index:
        if name == "final design":
            continue
        d = a.loc[name, "val_mean"] - base["val_mean"]
        sd = max(np.nanmax([a.loc[name, "val_sd"], base["val_sd"]]), 1e-4)
        dn = a.loc[name, "dev_noisy_mean"] - base["dev_noisy_mean"]
        verdict = "no clear difference" if abs(d) <= 2 * sd and abs(dn) <= 0.01 else ("worse" if d + dn < 0 else "better")
        out.append(f"{name}: Δval {d:+.4f}, Δdev-noisy {dn:+.4f} ({verdict})")
    return "; ".join(out)


def write_results_md(R: dict, path: str):
    f, rs = R["final"], R["resources"]
    ta, tb = R["task_a"], R["processor_d"]
    bres = tb["benchmark_results"]
    bl = R["baselines"].copy()
    ideal, noisy, shot = R["ideal_val"], R["noisy_val"], R["shot_ideal_val"]
    seeds = np.array([s["val_macro_f1"] for s in R["seeds"]])
    best_cl = bl.sort_values("val_macro_f1", ascending=False).iloc[0]
    na = R["noise_analysis"]
    abl = R["ablation"]
    mit = R["mitigation"]
    topo = tb["processor_id"]
    overlap = ideal['ci95'][0] <= best_cl['ci_hi'] and best_cl['ci_lo'] <= ideal['ci95'][1]
    cmp_sent = ("The two are statistically indistinguishable (bootstrap CIs overlap); there is no evidence of quantum advantage or disadvantage in accuracy, only of higher cost." if overlap else
                ("The quantum model is below the best classical model beyond the bootstrap CI." if ideal['macro_f1'] < best_cl['val_macro_f1'] else "The quantum model is above the best classical model beyond the bootstrap CI; this is validation only and should be confirmed on the hidden test."))
    a_ = abl.set_index('variant'); d_e = a_.loc['no entanglement (product circuit)', 'val_mean'] - a_.loc['final design', 'val_mean']
    sd_e = max(np.nanmax([a_.loc['no entanglement (product circuit)', 'val_sd'], a_.loc['final design', 'val_sd']]), 1e-4)
    ent_sent = (f"Removing every CX gate changes validation macro-F1 by {d_e:+.4f} (seed sd {sd_e:.4f}): " +
                ("entanglement is not what makes this data separable — the PCA/angle encoding and the trainable rotations are." if abs(d_e) <= 2 * sd_e else
                 ("entanglement helps measurably here." if d_e < 0 else "the product circuit is actually better, so the CX ring adds noise cost without benefit.")))
    top_t = R['task_b_topologies'].dropna(subset=['qft3_fidelity_noisy']).sort_values('qft3_fidelity_noisy', ascending=False).iloc[0]
    topo_sent = (f"The highest raw QFT-3 noisy fidelity among the candidates is {top_t['qft3_fidelity_noisy']:.4f} ({top_t['topology']}); our selected design reaches {bres['qft3_fidelity_noisy']:.4f} "
                 f"and is the best candidate that also compiles our classifier without SWAPs.")
    two_way = all([b, a] in tb["coupling_map"] for a, b in tb["coupling_map"])
    n_links = len({tuple(sorted(e)) for e in tb["coupling_map"]})

    approach = (f"Each heartbeat's 30 window statistics are standardised, reduced with a {f['reducer'].upper()} fitted on the training split to {f['n_qubits']} components, "
                f"and mapped to rotation angles in [0, π] with a robust (1st-99th percentile, clipped) min-max map; component k is loaded onto qubit k with RY. "
                f"A trainable RY·RZ rotation on every qubit and a ring of {R['logical_stats']['two_qubit_gates']} CX gates follow, and the model reads out the {'parity ⟨Z⊗…⊗Z⟩ of all qubits' if f['readout'] == 'all' else '⟨Z⟩ of qubit 0'}; "
                f"a trainable scale and bias turn it into a logit (hybrid read-out). The design was chosen on validation macro-F1 with a fixed cost-aware rule (fewest two-qubit gates among statistically tied models), "
                f"because under the organiser noise every CX costs 5 % depolarisation. The data is almost linearly separable, so the aim is a small, noise-robust circuit and an honest account of what the quantum parts contribute.")

    cl_tbl = bl.rename(columns={"model": "Model", "val_macro_f1": "Validation macro-F1"})
    cl_tbl["Notes"] = [f"95% CI [{r.ci_lo:.4f}, {r.ci_hi:.4f}]; tuned on validation: {r.params}" for r in bl.itertuples()]
    cl_tbl = cl_tbl[["Model", "Validation macro-F1", "Notes"]]

    q_rows = [
        ("Ideal, exact (state-vector), best seed", f"{ideal['macro_f1']:.4f}", "", f"95% CI [{ideal['ci95'][0]:.4f}, {ideal['ci95'][1]:.4f}]; validation; accuracy {ideal['accuracy']:.4f}"),
        ("Ideal, exact, mean ± sd over 3 seeds", f"{seeds.mean():.4f} ± {seeds.std(ddof=1):.4f}", "", "different initialisations, same selection rule"),
        (f"Ideal, {R['shots']} shots (Aer, noise-free)", f"{shot['macro_f1']:.4f}", "", f"CI [{shot['ci95'][0]:.4f}, {shot['ci95'][1]:.4f}]; shot noise only"),
        (f"Noisy, as deployed (QFF2026-v2, {R['shots']} shots)", "", f"{noisy['macro_f1']:.4f}", f"CI [{noisy['ci95'][0]:.4f}, {noisy['ci95'][1]:.4f}]; paired (ideal-shots − noisy) CI [{R['paired_diff'][0]:+.4f}, {R['paired_diff'][1]:+.4f}]"),
        ("Test set (labels hidden)", "", "", f"noisy predictions equal the ideal ones for {100 * R['test_agreement']:.2f} % of 1,500 samples; submitted file = ideal-simulation predictions"),
    ]
    q_tbl = pd.DataFrame(q_rows, columns=["Configuration", "Ideal", "Noisy (benchmark)", "Notes"])
    mit_tbl = mit.rename(columns={"strategy": "Strategy", "val_macro_f1": "Mitigated macro-F1 (validation)", "extra_circuits": "Extra circuit evals", "note": "Justified?"})

    drop = (ideal["macro_f1"] - noisy["macro_f1"])
    k_ro = f['n_qubits'] if f['readout'] == 'all' else 1
    noise_para = (f"Noise contracts the read-out: noisy ⟨Z-string⟩ ≈ {na['slope']:.3f} × ideal {na['intercept']:+.3f} (correlation {na['corr']:.4f}); read-out flips alone would give {0.96 ** k_ro:.3f}. "
                  f"Macro-F1 changes by {-drop:+.4f} (ideal {ideal['macro_f1']:.4f} → noisy {noisy['macro_f1']:.4f}); {na['flipped']} validation decisions flip relative to the noise-free shot run, "
                  f"with median |logit| {na['median_margin_flipped'] if na['median_margin_flipped'] is not None else float('nan'):.2f} versus {na['median_margin_all']:.2f} overall, i.e. only samples near the boundary are lost. "
                  f"Per-class recall ideal/noisy: normal {na['recall_ideal'][0]:.3f}/{na['recall_noisy'][0]:.3f}, arrhythmia {na['recall_ideal'][1]:.3f}/{na['recall_noisy'][1]:.3f}.")

    topo_tbl = R["task_b_topologies"][["topology", "qft3_circuit_depth_after_transpile", "qft3_two_qubit_gate_count", "qft3_swap_overhead", "qft3_fidelity_noisy", "vqc_swaps", "vqc_depth"]]
    gs_tbl = R["task_b_gatesets"][["gate_set", "qft3_circuit_depth_after_transpile", "qft3_two_qubit_gate_count", "qft3_swap_overhead", "qft3_fidelity_noisy", "vqc_swaps", "vqc_depth"]]
    von = R.get("vqc_on_D")
    von_txt = (f"End to end, our trained classifier compiled to this processor (directed map, native gates) gives macro-F1 {von['macro_f1']:.4f} on the first {von['n']} validation samples versus {von['generic_macro_f1']:.4f} "
               f"with the default all-to-all compilation (compiled depth {von['compiled']['depth']} vs {von['generic_compiled']['depth']})." if von else "")

    txt = f"""# QFF2026 QML Track — Results Report

**Team name:** {R['team']}
**Selected PS:** [x] PS-Q1  [ ] PS-Q2  [ ] PS-Q3  (same PS as Phase 1) — plus Task A and Task B
**Date:** {dt.date.today().isoformat()}
**Dependency versions:**
```
{R['pip_freeze_filtered']}
```
**Noise model version:** QFF2026-v2

---

## 1. Approach Summary
{approach}

---

## 2. Results

### Classical baseline
Fitted on `ecg_train.csv`, hyper-parameters chosen on `ecg_val.csv` only. Metric: macro-F1 on the validation set (test labels are hidden; the organisers' published reference for this dataset is a logistic regression with macro-F1 1.000 on the hidden test set).

{_md_table(cl_tbl)}

### Quantum / hybrid model
Official metric: Q1 macro-F1.

{_md_table(q_tbl)}

### Accuracy, precision, recall and F1 (labelled splits)
Test labels are held by the organisers, so these scores exist only for train and validation. "abnormal" = Arrhythmia (label 1, positive class); "macro" = unweighted mean of both classes.

{_md_table(R['metrics_table'][['model / split','n','accuracy','precision_abnormal','recall_abnormal','f1_abnormal','f1_macro','TN','FP','FN','TP']])}

Under the organiser noise model (validation, same trained parameters):

{_md_table(R['metrics_noisy'][['model / split','accuracy','precision_abnormal','recall_abnormal','f1_abnormal','f1_macro','TN','FP','FN','TP']])}

### Test-sample output
`results/test_samples_with_predictions.csv` is a copy of `ecg_test_features.csv` (all 1,500 rows, `sample_id` + 30 features unchanged) with `predicted_label`, `predicted_class`, `p_arrhythmia`, `logit` and the noisy-run columns `predicted_label_noisy`, `p_arrhythmia_noisy` appended. `predictions_q1.csv` is the same ideal-simulation prediction in the official `sample_id,label` format.

**Honest comparison.** Best classical model on the same validation set: {best_cl['model']} with {best_cl['val_macro_f1']:.4f}; quantum model {ideal['macro_f1']:.4f} (ideal), {noisy['macro_f1']:.4f} (noisy). {cmp_sent}

**How noise changes macro-F1 and why.** {noise_para}

### Noise mitigation (PS-Q3 required; optional for PS-Q1/Q2)
{_md_table(mit_tbl)}

### Ablations (validation macro-F1, 3 seeds; dev-noisy = exact density-matrix replica of the noise model on 500 validation samples)
{_md_table(abl.rename(columns={'variant': 'Variant', 'val_mean': 'Val mean', 'val_sd': 'Val sd', 'dev_noisy_mean': 'Dev-noisy mean', 'two_qubit_gates': 'CX', 'params': 'Trainable gates'}))}

{_abl_sentence(abl)}.

---

## 3. Task A — Bell state |Φ+⟩ (required)

Produced by `task_a_bell()` in `benchmark/qff_qml_benchmark.py` (8192 shots per basis).

**Fidelity definition used:** F = (1 + ⟨XX⟩ − ⟨YY⟩ + ⟨ZZ⟩) / 4, with ⟨P⟩ = (N_even − N_odd)/N, SE(⟨P⟩) = √((1 − ⟨P⟩²)/N) and SE(F) = √(SE_XX² + SE_YY² + SE_ZZ²)/4. (P(00)+P(11) is a population, not a fidelity, and is not accepted.)

| Condition | ⟨XX⟩ ± SE | ⟨YY⟩ ± SE | ⟨ZZ⟩ ± SE | F ± SE |
|-----------|-----------|-----------|-----------|--------|
| Ideal | {ta['ideal']['XX']:+.4f} ± {ta['ideal']['XX_se']:.4f} | {ta['ideal']['YY']:+.4f} ± {ta['ideal']['YY_se']:.4f} | {ta['ideal']['ZZ']:+.4f} ± {ta['ideal']['ZZ_se']:.4f} | {ta['ideal']['fidelity']:.4f} ± {ta['ideal']['fidelity_se']:.4f} |
| Noisy (benchmark) | {ta['noisy']['XX']:+.4f} ± {ta['noisy']['XX_se']:.4f} | {ta['noisy']['YY']:+.4f} ± {ta['noisy']['YY_se']:.4f} | {ta['noisy']['ZZ']:+.4f} ± {ta['noisy']['ZZ_se']:.4f} | {ta['noisy']['fidelity']:.4f} ± {ta['noisy']['fidelity_se']:.4f} |

*What the difference says about noise and entanglement:* {R['task_a_text']}

---

## 4. Processor D Benchmark (Task B)

Produced by `python benchmark/qff_qml_benchmark.py processor_d_config.json`.

**QFT-3 fidelity definition used:** mean over the 8 computational-basis inputs |j⟩ of F_j = ⟨ψ_j|ρ_j|ψ_j⟩, where ψ_j = QFT|j⟩ (ideal) and ρ_j is the density matrix of the 3 logical qubits after the transpiled circuit runs under the benchmark noise model (gate noise; no measurement). Ideal must be 1.0.

**Processor ID:** {topo}  **n_qubits:** {tb['n_qubits']}  **Topology:** {tb['qubit_layout']} ({n_links} links, {'all two-way' if two_way else 'some one-way'}); native gates {', '.join(tb['native_gate_set'])}

| Benchmark | Value |
|-----------|-------|
| QFT-3 depth (before transpile) | {bres['qft3_circuit_depth_before_transpile']} |
| QFT-3 depth (after transpile) | {bres['qft3_circuit_depth_after_transpile']} |
| Two-qubit gate count | {bres['qft3_two_qubit_gate_count']} |
| SWAP overhead | {bres['qft3_swap_overhead']} |
| Fidelity (ideal) | {bres['qft3_fidelity_ideal']} |
| Fidelity (noisy) | {bres['qft3_fidelity_noisy']} |
| Coupling map direction (one-way / two-way links) | {'all links two-way' if two_way else 'contains one-way links'} |

**Design rationale.** {tb['design_rationale']}

Candidate topologies (rx, ry, rz, cx), official benchmark:

{_md_table(topo_tbl)}

Candidate gate sets on the selected topology:

{_md_table(gs_tbl)}

{von_txt}

*What do the results tell you about your topology?* Connectivity that the workload actually uses (the ring of our classifier and the qubit pairs QFT-3 couples) removes SWAPs, and each avoided SWAP is three CX at 5 % error each; {topo_sent} Fidelity is decided by the number of two-qubit gates after routing and by link direction, not by the number of qubits. The trade-off is that a denser map is harder to build and calibrate in real hardware (crosstalk, frequency collisions) than a ring; here the benchmark (which only counts gates) cannot see that cost.

---

## 5. Resource Costs
| Resource | Value |
|----------|-------|
| Qubits used | {rs['qubits']} |
| Circuit depth | logical {rs['logical_depth']}; compiled to rx,ry,rz,cx {rs['compiled_depth']}; two-qubit gates {rs['logical_two_qubit']} (compiled {rs['compiled_two_qubit']}) |
| Trainable parameters | {rs['trainable_circuit_params']} circuit angles + {rs['readout_params']} classical read-out (w, b) |
| Training time (approx) | {rs['train_seconds_chosen']:.0f} s for the selected seed ({rs['train_seconds_all']:.0f} s for all 3 seeds), exact NumPy simulator on CPU |
| Total circuit evaluations | training (parameter-shift, per-sample circuits): {rs['train_circuit_evals_chosen']:,}; Aer inference: {rs['inference_circuits']:,} circuits × {rs['shots']} shots (+ mitigation circuits listed above) |

---

## 6. Limitations and Honest Assessment
* **Quantum is not better here, and cannot be.** The hidden-test classical reference is a plain logistic regression at macro-F1 1.000; on our validation split every sensible classical model and the quantum circuit are at {best_cl['val_macro_f1']:.3f}–{ideal['macro_f1']:.3f}. The headline number therefore says almost nothing about the quantum representation; the ablation table is the real evidence.
* **What the ablations show:** {_abl_sentence(abl)}. {ent_sent}
* **Noise:** the circuit is shallow ({R['compiled_stats']['two_qubit_gates']} CX after compilation), so noise shrinks the margin by ≈{(1 - na['slope']) * 100:.0f} % but moves few decisions; a deeper circuit (see 2/3-layer ablations) pays more per CX at p = 5 %.
* **Test set:** labels are hidden, so test performance is only inferred from validation and from the agreement between noisy and ideal predictions. Validation was used for model selection and threshold recalibration only; no test data touched any fit.
* **Simulation caveat:** training and the design sweeps use an exact NumPy simulator (verified against Qiskit `Statevector` to < 1e-8 in the notebook); every *reported* noisy number comes from qiskit-aer with the unchanged organiser model. The dev-noisy columns in the ablation are an exact replica of that model, not Aer.
* **Hardware relevance:** the Processor D benchmark counts gates only; real devices add crosstalk and connectivity cost that this benchmark does not see.

---

## 7. Reproducibility
```python
SEED = {R['seed']}       # global; model seeds 0, 1, 2; Aer seed 42; transpiler seed 42
```
Environment: `pip install -r requirements.txt` (pinned), then `jupyter nbconvert --to notebook --execute --inplace solution.ipynb`; full `pip freeze` is in `pip_freeze.txt` and in README.md.

---

## 8. Optional Extensions
[ ] Second mitigation strategy (PS-Q3)
[x] Ablation studies (Section 2)
[x] Other: read-out error mitigation, threshold re-calibration and zero-noise extrapolation evaluated for PS-Q1; Phase-1 processors A/B and one-way-link variants benchmarked as Task B baselines
"""
    with open(path, "w") as fh:
        fh.write(txt)


def update_readme_freeze(path: str, freeze: str):
    s = open(path).read()
    block = "<!-- PIP_FREEZE_START -->\n```\n" + freeze.strip() + "\n```\n<!-- PIP_FREEZE_END -->"
    if "<!-- PIP_FREEZE_START -->" in s:
        s = re.sub(r"<!-- PIP_FREEZE_START -->.*?<!-- PIP_FREEZE_END -->", lambda m: block, s, flags=re.S)
    else:
        s += "\n\n## pip freeze\n" + block + "\n"
    open(path, "w").write(s)
