"""fig8: the measured GPU crossover, same host, same run (Colab T4).

Reads experiment_4d_results.json (produced by experiment_4d_colab.py on a
T4 runtime) and plots forward and adjoint-gradient cost per backend at the
experiment's two widths: 8 wires (the paper's OSCD-scale model) and 20
wires (the 4D amplitude-embedded model). All bars are single-circuit
timings on the same machine, so the ratios are clean.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "experiment_4d_results.json")))

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
COLORS = {"default.qubit": "#2a78d6", "lightning.qubit": "#eb6834",
          "lightning.gpu": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})

bench = {(r["wires"], r["backend"]): r for r in res["benchmark"]
         if "forward_ms" in r}
widths = sorted({w for w, _ in bench})
backends = ["default.qubit", "lightning.qubit", "lightning.gpu"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, key, title in [(axes[0], "forward_ms", "Forward pass"),
                       (axes[1], "gradient_ms", "Adjoint gradient (a training step)")]:
    for gi, w in enumerate(widths):
        for bi, b in enumerate(backends):
            v = bench[(w, b)][key]
            x = gi * (len(backends) + 1) + bi
            ax.bar(x, v, 0.82, color=COLORS[b], zorder=3,
                   label=b if gi == 0 else None)
            lab = f"{v/1000:.1f}s" if v >= 1000 else f"{v:.0f}ms"
            ax.text(x, v * 1.25, lab, ha="center", fontsize=8.6, color=INK_2)
        # ratio annotation: GPU vs best CPU backend at this width
        cpu = min(bench[(w, b)][key] for b in backends if b != "lightning.gpu")
        gpu = bench[(w, "lightning.gpu")][key]
        r = cpu / gpu
        note = f"GPU {r:.1f}× faster" if r > 1 else f"GPU {1/r:.1f}× slower"
        ax.text(gi * (len(backends) + 1) + 1, ax.get_ylim()[0] + 0.9,
                "", fontsize=8.5)
        ax.annotate(note, xy=(gi * (len(backends) + 1) + 2, gpu),
                    xytext=(gi * (len(backends) + 1) + 0.6, gpu * 6),
                    fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_yscale("log")
    ax.set_xticks([1, len(backends) + 2])
    ax.set_xticklabels(["8 wires\n(OSCD-scale model)",
                        "20 wires\n(4D amplitude model)"])
    ax.set_title(title, color=INK, loc="left")
    ax.xaxis.grid(False)
axes[0].set_ylabel("milliseconds per circuit  (log scale)")
axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle("The GPU crossover, measured on one machine — "
             f"{res['env'].get('nvidia_smi', 'GPU')}, PennyLane {res['env']['pennylane']}",
             x=0.055, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(HERE, "fig8_gpu_t4.png")
fig.savefig(out, dpi=200)
print("wrote", out)
