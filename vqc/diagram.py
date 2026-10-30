"""Draws the VQC from its op list with matplotlib (same list that feeds the simulator and the Qiskit builder),
so the figure always shows exactly the circuit that was trained and evaluated."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


def draw_vqc(ops, n_qubits, readout, theta=None, path=None, title=None):
    cols, free = [], [0] * n_qubits          # greedy column assignment (ASAP layering)
    for g, qs, ref in ops:
        c = max(free[q] for q in qs)
        for q in range(min(qs), max(qs) + 1) if g == "cx" else qs:
            free[q] = c + 1
        cols.append(c)
    ncol = max(cols) + 2
    fig, ax = plt.subplots(figsize=(1.0 * ncol + 1.5, 0.9 * n_qubits + 1.2))
    for q in range(n_qubits):
        ax.plot([-0.6, ncol - 0.2], [-q, -q], color="k", lw=0.8, zorder=0)
        ax.text(-0.9, -q, f"q{q}", ha="right", va="center", fontsize=9)
    colour = {"x": "#cfe8ff", "t": "#ffe3b3"}
    for (g, qs, ref), c in zip(ops, cols):
        if g == "cx":
            ctl, tgt = qs
            ax.plot([c, c], [-ctl, -tgt], color="k", lw=1.2, zorder=1)
            ax.add_patch(Circle((c, -ctl), 0.09, color="k", zorder=2))
            ax.add_patch(Circle((c, -tgt), 0.2, fill=True, fc="w", ec="k", zorder=2))
            ax.plot([c - 0.2, c + 0.2], [-tgt, -tgt], color="k", lw=1.2, zorder=3)
            ax.plot([c, c], [-tgt - 0.2, -tgt + 0.2], color="k", lw=1.2, zorder=3)
        else:
            q = qs[0]
            label = f"{g.upper()}(x{ref[1]})" if ref[0] == "x" else f"{g.upper()}(θ{ref[1]})"
            ax.add_patch(Rectangle((c - 0.38, -q - 0.28), 0.76, 0.56, fc=colour[ref[0]], ec="k", zorder=2))
            ax.text(c, -q, label, ha="center", va="center", fontsize=7, zorder=3)
    for i, q in enumerate(readout):
        ax.add_patch(Rectangle((ncol - 0.85, -q - 0.28), 0.7, 0.56, fc="#d9f2d9", ec="k", zorder=2))
        ax.text(ncol - 0.5, -q, "⟨Z⟩", ha="center", va="center", fontsize=8, zorder=3)
    ax.set_xlim(-1.6, ncol); ax.set_ylim(-n_qubits + 0.2, 0.7); ax.axis("off")
    ax.text(0, 0.55, "blue: angle encoding of PCA features (data)    orange: trainable rotations    ⊕: CX (ring entangler)    green: Z-string read-out",
            fontsize=7, va="bottom")
    ax.set_title(title or "Hybrid variational quantum classifier (data → RY encoding → trainable RY·RZ → CX ring → ⟨Z…Z⟩ → w·⟨Z⟩+b)", fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160)
    return fig
