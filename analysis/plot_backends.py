"""Crossover figure: per-circuit cost vs qubit count, one line per simulator.

Reads backend_benchmark.json (CPU, this machine) and, if present,
gpu_benchmark.json (from the Colab run) and overlays both.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical order

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "legend.frameon": False,
})

series = {}
for fname in ("backend_benchmark.json", "gpu_benchmark.json"):
    path = os.path.join(HERE, fname)
    if not os.path.exists(path):
        continue
    blob = json.load(open(path))
    for r in blob["results"]:
        if r.get("kind", "forward") != "forward" or "per_circuit_us" not in r:
            continue
        series.setdefault(r["backend"], []).append((r["wires"], r["per_circuit_us"]))

fig, ax = plt.subplots(figsize=(9.6, 5.4))
order = ["default.qubit", "lightning.qubit", "lightning.gpu", "qiskit.aer"]
for name, colour in zip([n for n in order if n in series], SLOTS):
    pts = sorted(series[name])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.plot(xs, ys, color=colour, lw=2, marker="o", ms=5, label=name)
    ax.text(xs[-1] + 0.25, ys[-1], name, color=INK_2, fontsize=9, va="center")

ax.set_yscale("log")
ax.set_xlabel("qubits (circuit width)")
ax.set_ylabel("microseconds per circuit  (log scale)")

ax.axvline(8, color=INK, lw=1.2)
ax.text(8.15, ax.get_ylim()[1] * 0.35,
        "this project\n8 wires", fontsize=9, color=INK, va="top")

for n, label in ((18, "3×3 patch"), (32, "4×4 patch — 68 GB")):
    if n <= max(max(p[0] for p in v) for v in series.values()):
        ax.axvline(n, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.text(n + 0.2, ax.get_ylim()[0] * 2.2, label, fontsize=8.5, color=MUTED)

ax.legend(loc="upper left", fontsize=9.5)
ax.set_title("Simulator cost vs circuit width — accelerated backends only win once "
             "the state is large\nenough to amortise their per-call overhead",
             color=INK, loc="left", fontsize=12.5, pad=14)
fig.tight_layout()
out = os.path.join(HERE, "fig6_backend_crossover.png")
fig.savefig(out, dpi=200)
print("wrote", out)
print("series plotted:", list(series))
