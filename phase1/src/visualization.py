"""Plotting helpers."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

C = {"A": "#2563eb", "B": "#16a34a", "ideal": "#6b7280"}


def plot_coupling(proc, ax, color="#2563eb", title=None):
    g = nx.Graph(); g.add_nodes_from(range(proc["n_qubits"])); g.add_edges_from(proc["coupling_map"])
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, ax=ax, with_labels=True, node_color=color, font_color="white",
            node_size=650, edge_color="#94a3b8", width=2)
    ax.set_title(title or proc.get("name", ""), fontsize=10)


def plot_coupling_pair(procs, path):
    fig, axes = plt.subplots(1, len(procs), figsize=(5 * len(procs), 4))
    for ax, (k, p), col in zip(axes, procs.items(), [C["A"], C["B"]]):
        plot_coupling(p, ax, col, p["name"])
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_bars(df, cols, path, title):
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.8))
    for ax, c in zip(axes, cols):
        ax.bar(df.index, df[c], color="#64748b")
        ax.set_title(c); ax.tick_params(axis="x", rotation=35)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_ab_bars(df, cols, path, title):
    """Grouped A-vs-B bars; df index must be ['A','B'] (optionally plus 'ideal')."""
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.6))
    for ax, c in zip(axes, cols):
        ax.bar(df.index, df[c], color=[C.get(i, "#64748b") for i in df.index])
        ax.set_title(c)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def plot_curves(curves, path, title="Training convergence (BCE loss)"):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, (h, col) in curves.items():
        ax.plot(h, label=label, color=col, lw=2)
    ax.set_xlabel("SPSA iteration"); ax.set_ylabel("loss"); ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
