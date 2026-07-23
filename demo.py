"""
Monitor demo. Auto-looping, no input, no quantum work at runtime.

    python demo.py            -> writes demo.gif (open fullscreen in a browser)
    python demo.py --live     -> plays in a window

Three scenes:
  1. Band sweep    - k rises, the circuit widens, the map sharpens, cost doubles
  2. Shot noise    - one pixel, one circuit, 40 runs, 40 different answers
  3. Bottom line   - held, then loops

The narration that must accompany scene 2: the estimate is unbiased and
converges on the exact value. It is not unreliable - certainty is purchasable,
and that is the cost.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

C_Q, C_C, C_A = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df", "#fcfcfb"

HOLD = 8          # frames per band-count step
HOLD_END = 14     # frames on the closing card
FPS = 4

D = np.load("demo_data.npz")
KS, ACCS, CNOTS, QUBITS = D["ks"], D["accs"], D["cnots"], D["qubits"]
MAPS, SAMPLES, EXACT, SHOTS = D["maps"], D["samples"], float(D["exact"]), int(D["shots"])
SPECTRUM, BANDS16 = D["spectrum"], D["bands16"]

N1 = len(KS) * HOLD
N2 = len(SAMPLES)
N_TOTAL = N1 + N2 + HOLD_END

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "grid.color": GRID,
    "legend.frameon": False, "font.size": 13,
})


def _map_rgb(pred):
    h, w = pred.shape
    img = np.full((h, w, 3), 0.94)
    img[pred == 1] = matplotlib.colors.to_rgb(C_Q)
    img[pred == 0] = matplotlib.colors.to_rgb(C_C)
    return img


def _wires(ax, n, active=True):
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 6.5); ax.axis("off")
    for i in range(6):
        y = 5.5 - i
        if i < n:
            ax.plot([0.08, 0.92], [y, y], color=C_Q, linewidth=3.2, zorder=3)
            ax.text(0.03, y, f"$q_{i}$", ha="right", va="center",
                    fontsize=13, color=INK2)
        else:
            ax.plot([0.08, 0.92], [y, y], color=GRID, linewidth=2.0,
                    linestyle=":", zorder=1)


def scene_bands(fig, step):
    k, acc, cn, n = KS[step], ACCS[step], CNOTS[step], QUBITS[step]
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.7, 1.25],
                          left=0.04, right=0.97, top=0.84, bottom=0.08, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(_map_rgb(MAPS[step])); ax.axis("off")
    ax.set_title("prediction", loc="left", fontsize=15, color=INK2)

    axw = fig.add_subplot(gs[0, 1])
    _wires(axw, n)
    axw.set_title(f"{n} qubits", loc="left", fontsize=15, color=INK2)

    axt = fig.add_subplot(gs[0, 2]); axt.axis("off")
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.text(0, 0.98, f"{k}", fontsize=86, fontweight="bold", color=C_A,
             va="top", ha="left")
    axt.text(0, 0.70, "bands retained", fontsize=17, color=INK2, va="top")
    axt.text(0, 0.50, f"{acc*100:.1f}%", fontsize=48, fontweight="bold",
             color=INK, va="top")
    axt.text(0, 0.355, "test accuracy", fontsize=16, color=INK2, va="top")
    axt.text(0, 0.24, f"{cn}", fontsize=48, fontweight="bold", color=C_Q, va="top")
    axt.text(0, 0.095, "state-prep CNOTs on hardware", fontsize=16,
             color=INK2, va="top")

    fig.suptitle("Band count sets circuit width — and what it costs to run",
                 x=0.04, ha="left", fontsize=25, fontweight="bold", color=INK)


def scene_shots(fig, i):
    shown = SAMPLES[: i + 1]
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5],
                          left=0.06, right=0.97, top=0.84, bottom=0.14, wspace=0.24)

    axs = fig.add_subplot(gs[0, 0])
    axs.plot(BANDS16, SPECTRUM, color=C_A, linewidth=3.0, marker="o", markersize=7)
    axs.set_xlabel("wavelength (nm)", fontsize=14)
    axs.set_ylabel("standardized reflectance", fontsize=14)
    axs.set_title("one pixel, 16 bands", loc="left", fontsize=15, color=INK2)
    axs.grid(True, linewidth=0.6)

    axd = fig.add_subplot(gs[0, 1])
    axd.axhline(EXACT, color=INK, linewidth=2.6, zorder=4)
    axd.annotate(f"exact  {EXACT:.4f}", xy=(len(SAMPLES) * 0.99, EXACT),
                 xytext=(0, 9), textcoords="offset points", fontsize=15,
                 fontweight="bold", color=INK, ha="right")
    axd.scatter(np.arange(len(shown)), shown, s=95, color=C_Q,
                edgecolor=SURFACE, linewidth=1.4, zorder=5)
    if len(shown) > 1:
        axd.plot(np.arange(len(shown)), np.cumsum(shown) / np.arange(1, len(shown) + 1),
                 color=C_C, linewidth=2.4, zorder=6, label="running mean")
        axd.legend(loc="lower right", fontsize=14)
    axd.set_xlim(-1, len(SAMPLES)); axd.set_ylim(-0.29, -0.05)
    axd.set_xlabel(f"run number  (each is {SHOTS} shots)", fontsize=14)
    axd.set_ylabel(r"estimated $\langle Z_0\rangle$", fontsize=14)
    axd.set_title(f"run {i+1} of {len(SAMPLES)}", loc="left", fontsize=15, color=INK2)
    axd.grid(True, linewidth=0.6)

    fig.suptitle("Same pixel. Same circuit. Same weights. A different answer every run.",
                 x=0.04, ha="left", fontsize=25, fontweight="bold", color=INK)


def scene_end(fig):
    ax = fig.add_subplot(111); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.72,
            "Band selection, not qubit count,\nsets the cost.",
            ha="center", va="center", fontsize=44, fontweight="bold",
            color=INK, linespacing=1.35)
    ax.text(0.5, 0.40,
            "The estimate is unbiased — it converges on the exact value.\n"
            "Certainty is purchasable: 4× the shots buys 2× the precision.",
            ha="center", va="center", fontsize=21, color=INK2, linespacing=1.5)
    ax.add_patch(FancyBboxPatch((0.255, 0.13), 0.49, 0.105,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=C_Q, edgecolor=C_Q))
    ax.text(0.5, 0.1825, "github.com/deepeshmode/quantumml", ha="center",
            va="center", fontsize=17, color="#ffffff", fontweight="bold")


def draw(frame):
    fig = plt.gcf(); fig.clf()
    if frame < N1:
        scene_bands(fig, frame // HOLD)
    elif frame < N1 + N2:
        scene_shots(fig, frame - N1)
    else:
        scene_end(fig)
    return []


def main():
    fig = plt.figure(figsize=(16, 9), dpi=100)
    anim = animation.FuncAnimation(fig, draw, frames=N_TOTAL,
                                   interval=1000 // FPS, blit=False)
    if "--live" in sys.argv:
        matplotlib.use("MacOSX")
        plt.show()
    else:
        anim.save("demo.gif", writer=animation.PillowWriter(fps=FPS))
        print(f"demo.gif  {N_TOTAL} frames @ {FPS}fps  "
              f"= {N_TOTAL/FPS:.0f}s loop")


if __name__ == "__main__":
    main()
