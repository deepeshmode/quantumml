"""
Figures.

fig_pipeline.png  - the three subplots the project brief calls for: QNN schematic with the
                    pipeline, exploration of the imagery, and the result of
                    running the model on it.
fig_scaling.png   - accuracy against band count, over the hardware
                    state-preparation cost that band count controls.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from pipeline import (
    load_pavia, binary_task, split, wavelengths, rgb_composite, ndvi,
    select_bands, select_bands_decorrelated, CLASS_NAMES, BUILT_CLASSES, VEG_CLASSES,
)
from qnn import train_eval

# Validated categorical slots (light mode).
C_Q = "#2a78d6"       # quantum / built surface
C_C = "#eb6834"       # classical / vegetation
C_A = "#1baf7a"       # third series
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
})


_FS = 1.0


def fs(p):
    """Point size, scaled for the figure currently being drawn."""
    return p * _FS


def print_fonts(fig_width_in, placed_mm, target_pt=20):
    """
    rcParams that put a figure's text at target_pt once printed.

    A figure authored at fig_width_in and placed at placed_mm on the poster is
    scaled by placed_mm / (fig_width_in * 25.4). Point sizes scale with it, so
    authoring at a fixed 9 pt gives wildly different printed text depending on
    placement - which is how figure text ends up illegible on an A0 sheet even
    at 600 dpi.
    """
    scale = placed_mm / (fig_width_in * 25.4)
    base = target_pt / scale
    return {
        "font.size": base, "axes.labelsize": base,
        "axes.titlesize": base * 1.15, "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9, "legend.fontsize": base * 0.9,
        "lines.linewidth": max(2.0, base * 0.22),
        "lines.markersize": max(6.0, base * 0.6),
    }


def _box(ax, x, y, w, h, label, sub, face, edge, textcol="#ffffff"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h * 0.72, label, ha="center", va="center",
            fontsize=fs(8.5), fontweight="bold", color=textcol)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=fs(7), color=textcol, linespacing=1.35)


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.3, color=MUTED))


def panel_schematic(ax, k, n_qubits):
    """Subplot 1: the QNN and the pipeline that feeds it."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("A.  Pipeline and quantum circuit", loc="left",
                 fontweight="bold", color=INK)

    y, h = 0.62, 0.34
    stages = [
        (0.005, 0.150, "Hyperspectral", "103 bands\n430-860 nm", MUTED, MUTED),
        (0.185, 0.150, "Band select", "uniform\nspacing", MUTED, MUTED),
        (0.365, 0.150, "Multispectral", f"k = {k} bands", C_A, C_A),
        (0.545, 0.150, "Perceptron", f"{k} -> {2**n_qubits}", C_C, C_C),
        (0.725, 0.270, "Quantum layer", f"{n_qubits} qubits", C_Q, C_Q),
    ]
    for x, w, lab, sub, face, edge in stages:
        _box(ax, x, y, w, h, lab, sub, face, edge)
    for x0, x1 in [(0.155, 0.183), (0.335, 0.363), (0.515, 0.543), (0.695, 0.723)]:
        _arrow(ax, x0, x1, y + h / 2)

    # Circuit detail beneath the quantum-layer box.
    cy0, cy1 = 0.05, 0.46
    wires = np.linspace(cy1, cy0, n_qubits)
    for i, wy in enumerate(wires):
        ax.plot([0.30, 0.905], [wy, wy], color=INK2, linewidth=1.0, zorder=1)
        ax.text(0.285, wy, f"$q_{i}$", ha="right", va="center",
                fontsize=fs(7.5), color=INK2)

    ax.add_patch(FancyBboxPatch(
        (0.335, cy0 - 0.035), 0.16, (cy1 - cy0) + 0.07,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor=C_A, edgecolor=C_A, alpha=0.9, zorder=2))
    ax.text(0.415, (cy0 + cy1) / 2, "Amplitude\nembedding", ha="center",
            va="center", fontsize=fs(7.5), fontweight="bold", color="#ffffff", zorder=3)

    for j in range(3):
        bx = 0.545 + j * 0.135
        ax.add_patch(FancyBboxPatch(
            (bx, cy0 - 0.035), 0.115, (cy1 - cy0) + 0.07,
            boxstyle="round,pad=0.006,rounding_size=0.015",
            facecolor=C_Q, edgecolor=C_Q, alpha=0.92, zorder=2))
        ax.text(bx + 0.0575, (cy0 + cy1) / 2, f"STD\n{j+1}", ha="center",
                va="center", fontsize=fs(7.5), fontweight="bold",
                color="#ffffff", zorder=3)

    ax.add_patch(FancyBboxPatch(
        (0.905, wires[0] - 0.045), 0.058, 0.09,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=SURFACE, edgecolor=INK, linewidth=1.3, zorder=4))
    ax.text(0.934, wires[0], "Z", ha="center", va="center", fontsize=fs(8),
            fontweight="bold", color=INK, zorder=5)
    ax.text(0.934, wires[0] - 0.062, r"$\langle Z\rangle \to \sigma$",
            ha="center", va="top", fontsize=fs(7.5), color=INK2)

    ax.text(0.005, -0.02,
            f"Amplitude embedding packs k={k} bands into "
            f"$\\log_2 k$ = {n_qubits} qubits, so band selection sets circuit width.",
            fontsize=fs(7.5), color=INK2, va="top")


def panel_imagery(axes, cube, gt, sel_idx):
    """Subplot 2: exploration of the imagery."""
    ax_rgb, ax_ndvi, ax_spec = axes

    ax_rgb.imshow(rgb_composite(cube)); ax_rgb.axis("off"); ax_rgb.grid(False)
    ax_rgb.set_title("B.  Scene\n     (true colour)", loc="left",
                     fontweight="bold", color=INK)

    nd = ndvi(cube)
    im = ax_ndvi.imshow(nd, cmap="RdYlGn", vmin=-0.3, vmax=0.6)
    ax_ndvi.axis("off"); ax_ndvi.grid(False)
    ax_ndvi.set_title("NDVI\n ", loc="left", fontweight="bold", color=INK)
    cb = plt.colorbar(im, ax=ax_ndvi, fraction=0.055, pad=0.04)
    cb.ax.tick_params(labelsize=fs(7), colors=INK2)
    cb.outline.set_edgecolor(GRID)

    wl = wavelengths(cube.shape[-1])
    built = cube[np.isin(gt, BUILT_CLASSES)].mean(0)
    veg = cube[np.isin(gt, VEG_CLASSES)].mean(0)
    ax_spec.plot(wl, built, color=C_Q, linewidth=2.0, label="Built / disturbed")
    ax_spec.plot(wl, veg, color=C_C, linewidth=2.0, label="Vegetation")
    for b in wl[sel_idx]:
        ax_spec.axvline(b, color=MUTED, linewidth=0.8, alpha=0.55, zorder=0)
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Mean reflectance (scaled)")
    ax_spec.set_title("Class spectra; grey lines = selected bands",
                      loc="left", fontweight="bold", color=INK)
    ax_spec.legend(loc="upper left", fontsize=fs(8))
    ax_spec.set_xlim(wl[0], wl[-1])


def panel_result(ax_map, cube, gt, pred_map, acc, hist):
    """Subplot 3: the model run on the imagery."""
    H, W = gt.shape
    disp = np.full((H, W, 3), 0.94)
    labeled = np.isin(gt, BUILT_CLASSES + VEG_CLASSES)
    disp[labeled & (pred_map == 1)] = matplotlib.colors.to_rgb(C_Q)
    disp[labeled & (pred_map == 0)] = matplotlib.colors.to_rgb(C_C)
    ax_map.imshow(disp); ax_map.axis("off"); ax_map.grid(False)
    ax_map.set_title(f"C.  QNN prediction\n     (test accuracy {acc:.1%})",
                     loc="left", fontweight="bold", color=INK)
    ax_map.scatter([], [], c=C_Q, s=40, label="Built / disturbed")
    ax_map.scatter([], [], c=C_C, s=40, label="Vegetation")
    ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01),
                  ncol=1, fontsize=fs(7), handletextpad=0.5)


def figure_pipeline(k=16, method="uniform"):
    """
    Uniform selection by default: the sweep in run_experiment.py shows it beats
    decorrelated MI at every k >= 16, so the showcase figure should use the
    method that wins. Using decorr here would report a different accuracy at the
    same k than fig_scaling does.
    """
    cube, gt = load_pavia()
    X, y = binary_task(cube, gt)
    idx = (select_bands(X, y, k, "uniform") if method == "uniform"
           else select_bands_decorrelated(X, y, k, "mi"))
    n_qubits = int(np.log2(k))

    Xtr, ytr, Xte, yte = split(X[:, idx], y, seed=0)
    acc, hist, model = train_eval(Xtr, ytr, Xte, yte, k, epochs=15, seed=0)

    # Predict across every labeled pixel for the map.
    import torch
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    labeled = np.isin(gt, BUILT_CLASSES + VEG_CLASSES)
    Xall = (cube[labeled][:, idx] - mu) / sd
    with torch.no_grad():
        p = model(torch.tensor(Xall, dtype=torch.float32)).numpy()
    pred_map = np.zeros(gt.shape, dtype=int)
    pred_map[labeled] = (p > 0.5).astype(int)

    # Portrait, to match the poster column it sits in. A landscape figure has to
    # be scaled down to fit a 195 mm column, which is what made the scene images
    # small; this shape lets them render large.
    # Spans the full poster width as a band, so it is scaled UP in print rather
    # than down - which is what keeps the internal text legible at A0 and lets
    # the three scene images render large.
    global _FS
    W_IN, PLACED_MM = 19, 713
    _FS = 1.52
    with plt.rc_context(print_fonts(W_IN, PLACED_MM, target_pt=18)):
        fig = plt.figure(figsize=(W_IN, 10))
        gs = fig.add_gridspec(2, 4, height_ratios=[0.88, 2.4],
                              width_ratios=[0.52, 0.52, 0.52, 1.25],
                              hspace=0.14, wspace=0.34)
        panel_schematic(fig.add_subplot(gs[0, :]), k, n_qubits)
        panel_imagery([fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
                       fig.add_subplot(gs[1, 3])], cube, gt, idx)
        panel_result(fig.add_subplot(gs[1, 2]), cube, gt, pred_map, acc, hist)

        fig.suptitle(
            "Hyperspectral to multispectral to QNN: built-surface classification",
            x=0.006, y=0.998, ha="left", fontsize=fs(13), fontweight="bold",
            color=INK)
        fig.savefig("fig_pipeline.png", dpi=500, bbox_inches="tight")
    _FS = 1.0
    print(f"fig_pipeline.png  (k={k}, acc={acc:.4f})")
    return acc


def figure_scaling():
    r = json.load(open("results.json"))
    runs = r["runs"]
    meta = r["meta"]
    ks = sorted({x["k"] for x in runs})

    def series(method, key):
        return [next(x[key] for x in runs if x["method"] == method and x["k"] == k)
                for k in ks]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 7.4), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.16})

    for method, colour, lab in [("uniform", C_Q, "QNN, uniform bands"),
                                ("decorr", C_A, "QNN, decorrelated MI")]:
        m = np.array(series(method, "acc_mean")) * 100
        s = np.array(series(method, "acc_std")) * 100
        ax1.plot(ks, m, color=colour, linewidth=2.0, marker="o",
                 markersize=6, label=lab, zorder=3)
        ax1.fill_between(ks, m - s, m + s, color=colour, alpha=0.16,
                         linewidth=0, zorder=2)

    ax1.plot(ks, np.array(series("uniform", "classical_acc_mean")) * 100,
             color=C_C, linewidth=2.0, marker="s", markersize=5.5,
             linestyle="--", label="Logistic regression, uniform bands", zorder=3)
    ax1.axhline(meta["classical_acc_all_bands"] * 100, color=MUTED,
                linewidth=1.4, linestyle=":", zorder=1)
    ax1.annotate(f"Logistic regression, all 103 bands "
                 f"({meta['classical_acc_all_bands']*100:.1f}%)",
                 xy=(ks[0], meta["classical_acc_all_bands"] * 100),
                 xytext=(2, 6), textcoords="offset points",
                 fontsize=fs(7.6), color=INK2, ha="left")

    knee = next(x for x in runs if x["method"] == "uniform" and x["k"] == 16)
    peak = max((x for x in runs if x["method"] == "uniform"),
               key=lambda x: x["acc_mean"])
    ax1.annotate(f"k=16\n{knee['acc_mean']*100:.1f}%",
                 xy=(16, knee["acc_mean"] * 100), xytext=(7, -6),
                 textcoords="offset points", fontsize=fs(8.2), fontweight="bold",
                 color=INK, ha="left", va="top")
    ax1.annotate(f"k={peak['k']}\n{peak['acc_mean']*100:.1f}%",
                 xy=(peak["k"], peak["acc_mean"] * 100), xytext=(0, 9),
                 textcoords="offset points", fontsize=fs(8.2), fontweight="bold",
                 color=INK, ha="center")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_ylim(89.3, 96.6)
    ax1.set_title("Accuracy plateaus at 16-32 bands, then falls", loc="left",
                  fontweight="bold", color=INK)
    ax1.legend(loc="lower right", fontsize=fs(8))

    cn = series("uniform", "stateprep_cnots_hw")
    ax2.plot(ks, cn, color=C_Q, linewidth=2.0, marker="o", markersize=6, zorder=3)
    for k, v in zip(ks, cn):
        ax2.annotate(f"{v}", xy=(k, v), xytext=(0, 7),
                     textcoords="offset points", fontsize=fs(8), color=INK2,
                     ha="center")
    ax2.set_ylabel("State-prep CNOTs (modeled)")
    ax2.set_xlabel("Bands retained,  k   (qubits = $\\log_2 k$)")
    ax2.set_title("Hardware cost keeps doubling", loc="left",
                  fontweight="bold", color=INK)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(ks); ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_ylim(0, max(cn) * 1.25)

    fig.suptitle("Band count sets both accuracy and circuit cost",
                 x=0.012, ha="left", fontsize=fs(12.5), fontweight="bold", color=INK)
    fig.text(0.012, 0.005,
             f"Pavia University, built vs. vegetation. Mean +/- s.d. over "
             f"{len(meta['seeds'])} seeds. CNOT counts are modeled for exact "
             f"amplitude embedding\n(Mottonen et al.); the simulator applies "
             f"StatePrep directly and so hides this term entirely.",
             fontsize=fs(7.4), color=INK2, va="bottom")
    fig.savefig("fig_scaling.png", dpi=600, bbox_inches="tight")

    print("fig_scaling.png")
    for x in sorted(runs, key=lambda z: (z["method"], z["k"])):
        print(f"  {x['method']:8s} k={x['k']:3d}  qnn={x['acc_mean']*100:5.2f}%  "
              f"logreg={x['classical_acc_mean']*100:5.2f}%  "
              f"cnots={x['stateprep_cnots_hw']:3d}")


def figure_shot_noise():
    """Objective 2: the two frameworks agree; sampling is what differs."""
    r = json.load(open("sim_comparison.json"))
    exact = r["exact"]
    sweep = r["sweep"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 7.4),
        gridspec_kw={"height_ratios": [1.1, 1], "hspace": 0.42})

    # Spread of repeated estimates at a representative shot count.
    row = next(s for s in sweep if s["shots"] == 1024)
    bins = np.linspace(min(row["pl_samples"] + row["qk_samples"]) - 0.01,
                       max(row["pl_samples"] + row["qk_samples"]) + 0.01, 22)
    ax1.hist(row["pl_samples"], bins=bins, color=C_Q, alpha=0.75,
             label="PennyLane, 1024 shots")
    ax1.hist(row["qk_samples"], bins=bins, color=C_A, alpha=0.75,
             label="Qiskit Aer, 1024 shots")
    ax1.axvline(exact, color=INK, linewidth=2.0, zorder=5)
    ax1.annotate(f"analytic  {exact:.4f}\n(both frameworks, to 5e-16)",
                 xy=(exact, ax1.get_ylim()[1]), xytext=(8, -8),
                 textcoords="offset points", fontsize=fs(8), fontweight="bold",
                 color=INK, va="top")
    ax1.set_xlabel(r"estimated $\langle Z_0\rangle$")
    ax1.set_ylabel(f"count (of {len(row['pl_samples'])} runs)")
    ax1.set_title("Same circuit, same weights, same pixel - 40 runs each",
                  loc="left", fontweight="bold", color=INK)
    ax1.legend(loc="upper left", fontsize=fs(8))

    # Precision is bought with shots.
    shots = [s["shots"] for s in sweep]
    ax2.plot(shots, [s["pl_std"] for s in sweep], color=C_Q, linewidth=2.0,
             marker="o", markersize=6, label="PennyLane")
    ax2.plot(shots, [s["qk_std"] for s in sweep], color=C_A, linewidth=2.0,
             marker="s", markersize=5.5, label="Qiskit Aer")
    ax2.plot(shots, [s["theory_std"] for s in sweep], color=MUTED,
             linewidth=1.6, linestyle=":", label=r"binomial $1/\sqrt{N}$")
    ax2.set_xscale("log", base=2); ax2.set_yscale("log")
    ax2.set_xticks(shots); ax2.set_xticklabels([str(s) for s in shots])
    ax2.set_xlabel("shots per estimate,  N")
    ax2.set_ylabel("s.d. of estimate")
    ax2.set_title("Precision costs shots: 4x the shots buys 2x the precision",
                  loc="left", fontweight="bold", color=INK)
    ax2.legend(loc="lower left", fontsize=fs(8))

    fig.suptitle("PennyLane and Qiskit do not disagree - sampling does",
                 x=0.012, ha="left", fontsize=fs(12.5), fontweight="bold", color=INK)
    fig.text(0.012, 0.005,
             "Circuit built natively in each framework, not converted. On an "
             "identical circuit the analytic\nresults match to machine "
             "precision; what differs between runs is shot noise, which both "
             "share.",
             fontsize=fs(7.4), color=INK2, va="bottom")
    fig.savefig("fig_shot_noise.png", dpi=600, bbox_inches="tight")
    print(f"fig_shot_noise.png  (analytic {exact:.6f})")


if __name__ == "__main__":
    figure_pipeline(k=16)
    figure_scaling()
    figure_shot_noise()
