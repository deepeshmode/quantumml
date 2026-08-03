"""Explainability (xAI) for the trained OSCD change-detection QNN.

Implements the two model-agnostic explainers studied in Steinmuller, Schulz,
Graf & Herr, "eXplainable AI for Quantum Machine Learning" (arXiv:2211.01441)
— Baseline SHAP and Integrated Gradients — on the ACTUAL trained pixel-level
QNN from the OSCD reproduction (run overjoyed-doe-477), plus the Fourier-mode
analysis their qSHAP method builds on.

Why this model is the tractable case, and the 4D model is not
-------------------------------------------------------------
The paper's central observation: xAI cost scales exponentially in the number
of FEATURES, not qubits. The OSCD model angle-encodes 8 features (4 PCA
components x 2 dates), so exact Baseline SHAP needs only 2^8 = 256 coalition
evaluations per explained pixel, and the model's quantum core is a truncated
Fourier series with spectrum {-1,0,1}^8 (Schuld et al., PRA 103 032430),
recoverable exactly from 3^8 = 6,561 circuit evaluations.

The 4D Arm B model (experiment_4d_colab.py) amplitude-encodes 2^20 features
into 20 qubits. Exact SHAP would need 2^(2^20) coalitions — a number with
~315,653 digits — and amplitude encoding is not rotation encoding, so the
Fourier structure behind qSHAP does not apply at all. The embedding that
makes 4D data simulable (and makes a GPU worthwhile) is the same embedding
that makes per-feature explanation intractable. This script quantifies both
sides; the 4D cost table is printed and stored alongside the OSCD results.

What is actually computed here
------------------------------
1. Exact Baseline SHAP for 8 features (all 2^8 subset values, factorial
   weights), on stratified samples of TP / FP / FN / TN pixels from the
   full-resolution inference. Verified per-pixel against the efficiency
   axiom  sum_e Sh(e) = f(x) - f(b)  to float precision.
2. Integrated Gradients via the paper's finite-difference form (their
   eq. 2), with the completeness check  sum_e IG(e) ~ f(x) - f(b).
3. A symmetry decomposition of the attributions. Features are
   [d1_pc1..4 | d2_pc1..4]. For each component i, split into
   symmetric  (Sh(d1_i)+Sh(d2_i))/2  — responds to the pixel's spectral
   LEVEL, same at both dates ("urban-ness") — and antisymmetric
   (Sh(d1_i)-Sh(d2_i))/2 — responds to the between-date CONTRAST, the
   part a change detector should use. The error analysis (fig2) showed
   false alarms tracing the built environment; if that reading is right,
   the symmetric share should dominate, especially for false positives.
4. The exact Fourier spectrum of the trained quantum core g(u) on its 8
   embedded angles: evaluated on the {0, 2pi/3, 4pi/3}^8 grid, FFT'd, and
   reported as coefficient mass by interaction order |omega|_0 (how much of
   the trained function is bias / additive / pairwise / higher-order).
   The replicated qnode is checked against the model's own quantum layer
   before use, and the truncated series is checked to reconstruct g at
   random points.

Run locally (needs the sibling qnn_change_detection checkout + its venv +
the OSCD dataset + the MLflow model artifact; ~4 min, CPU):

    cd ~/Downloads/qnn_change_detection
    ./.venv/bin/python ~/Downloads/quantumml/analysis/xai_oscd.py

Outputs: xai_results.json and fig7_xai.png next to this script.
"""

import json
import math
import os
import sys
import time
import warnings
from itertools import product as iproduct

warnings.filterwarnings("ignore")

REPO = os.path.expanduser("~/Downloads/qnn_change_detection")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.environ.setdefault("SIPWQNN_DATA_PATH", os.path.expanduser("~/Downloads/oscd_root"))

import numpy as np
import torch

ART = os.path.join(
    REPO, "mlruns/1/20cd6303631a4aa2bde0029415b45191/artifacts/model/data/model.pth")
CACHE = os.path.join(REPO, "analysis", ".inference_cache.npz")

N_FEAT = 8
N_PER_GROUP = 40          # explained pixels per outcome group
IG_NODES = 32             # path nodes for Integrated Gradients
IG_DELTA = 0.05           # finite-difference shift (paper eq. 2)
SEED = 11


# ------------------------------------------------------------ model access

def load_model():
    model = torch.load(ART, map_location="cpu", weights_only=False)
    model.eval()
    return model


def f_batch(model, X):
    """Model probability for a batch of 8-feature vectors [d1(4) | d2(4)]."""
    with torch.no_grad():
        t = torch.tensor(np.asarray(X, dtype=np.float32))
        return model(t[:, :4], t[:, 4:]).numpy().ravel()


# ------------------------------------------------------------ pixel samples

def gather_pixels(rng):
    """Stratified TP/FP/FN/TN pixels with their 8-feature vectors.

    Probabilities and ground truth come from the cached full-resolution
    inference (analyse_oscd.py); features are rebuilt from the dataset the
    same way that script does, aligned by (city, row, col)."""
    if not os.path.exists(CACHE):
        raise SystemExit(f"missing {CACHE} — run analyse_oscd.py first")
    z = np.load(CACHE)
    cities = sorted({k.split("_")[0] for k in z.files if k.endswith("_prob")})

    import torchvision.transforms as tr
    from src.oscd_dataloader import ChangeDetectionDataset
    print("  rebuilding test dataset for features ...")
    ds = ChangeDetectionDataset(
        os.environ["SIPWQNN_DATA_PATH"], train=False, patch_side=1, stride=10,
        transform=tr.Compose([]), band_type=2, normalise_img=True,
        fp_modifier=1, n_components=4)

    # The symmetric/antisymmetric decomposition pairs d1 component i with d2
    # component i, but the repo fits the two PCAs independently (the basis-
    # mismatch defect documented in ANSWERS.md). Per-component SIGN flips
    # cancel out of SHAP (attribution covaries with both the sensitivity and
    # the feature deviation), but component reordering or rotation mixing
    # would smear the decomposition — so measure the actual overlap and
    # report it as a validity diagnostic.
    O = ds.pca1.components_ @ ds.pca2.components_.T          # [4, 4]
    alignment = dict(
        overlap_diag_abs=[float(a) for a in np.abs(np.diag(O))],
        max_offdiag_abs=float(np.abs(O - np.diag(np.diag(O))).max()),
        note=("pairing valid where |diag| ~ 1 and off-diagonals are small; "
              "sign flips are harmless for the SHAP decomposition"),
    )

    groups = {g: [] for g in ("TP", "FP", "FN", "TN")}
    for city in cities:
        prob, gt = z[f"{city}_prob"], z[f"{city}_gt"]
        I1 = np.asarray(ds.imgs_1[city])          # [4, H, W]
        I2 = np.asarray(ds.imgs_2[city])
        pred = (prob >= 0.5)
        masks = dict(TP=pred & (gt == 1), FP=pred & (gt == 0),
                     FN=~pred & (gt == 1), TN=~pred & (gt == 0))
        for g, m in masks.items():
            ys, xs = np.where(m)
            if not len(ys):
                continue
            take = rng.choice(len(ys), min(N_PER_GROUP, len(ys)), replace=False)
            for i in take:
                y, x = ys[i], xs[i]
                groups[g].append(np.concatenate([I1[:, y, x], I2[:, y, x]]))
    for g in groups:
        arr = np.array(groups[g], dtype=np.float32)
        idx = rng.choice(len(arr), min(N_PER_GROUP, len(arr)), replace=False)
        groups[g] = arr[idx]
        print(f"  {g}: {len(groups[g])} pixels")
    return groups, alignment


# ------------------------------------------------------------ exact SHAP

def shap_exact(model, X, b):
    """Exact Baseline SHAP for n=8 features via all 2^8 subset values.

    For each pixel: evaluate f on every coalition vector g_S (one batched
    forward of 256), then combine marginal contributions with the exact
    factorial weights |S|!(n-|S|-1)!/n!. Efficiency axiom checked."""
    n = X.shape[1]
    masks = np.array(list(iproduct([0, 1], repeat=n)), dtype=np.float64)  # [256, n]
    fact = [math.factorial(k) for k in range(n + 1)]
    sizes = masks.sum(axis=1).astype(int)

    # weight for the pair (S, S+e): |S|!(n-|S|-1)!/n! with S excluding e
    w_excl = np.array([fact[s] * fact[n - s - 1] / fact[n] for s in range(n)])

    mask_index = {tuple(m.astype(int)): i for i, m in enumerate(masks)}
    shap = np.zeros((len(X), n))
    max_eff_err = 0.0
    for j, x in enumerate(X):
        G = b[None, :] + masks * (x - b)[None, :]
        v = f_batch(model, G)                      # all 256 subset values
        for e in range(n):
            off = masks[:, e] == 0                 # subsets S excluding e
            for i in np.where(off)[0]:
                m2 = masks[i].copy()
                m2[e] = 1
                i2 = mask_index[tuple(m2.astype(int))]
                shap[j, e] += w_excl[sizes[i]] * (v[i2] - v[i])
        eff = abs(shap[j].sum() - (v[-1] - v[0]))
        max_eff_err = max(max_eff_err, eff)
    return shap, max_eff_err


# ------------------------------------------------------------ IG (paper eq. 2)

def integrated_gradients(model, X, b):
    """Finite-difference Integrated Gradients along the straight path b->x,
    exactly the estimator in arXiv:2211.01441 eq. (2)."""
    n = X.shape[1]
    ig = np.zeros((len(X), n))
    comp_err = np.zeros(len(X))
    for j, x in enumerate(X):
        s = (np.arange(1, IG_NODES) / IG_NODES)[:, None]
        gamma = s * x[None, :] + (1 - s) * b[None, :]      # interior nodes
        batch = []
        for e in range(n):
            for sgn in (+1, -1):
                g = gamma.copy()
                g[:, e] += sgn * IG_DELTA
                batch.append(g)
        v = f_batch(model, np.concatenate(batch, axis=0))
        v = v.reshape(n, 2, IG_NODES - 1)
        for e in range(n):
            ig[j, e] = (x[e] - b[e]) / (2 * IG_NODES * IG_DELTA) \
                * (v[e, 0] - v[e, 1]).sum() * 2
        fx, fb = f_batch(model, np.stack([x, b]))
        comp_err[j] = abs(ig[j].sum() - (fx - fb))
    return ig, float(comp_err.mean())


# ------------------------------------------------------------ Fourier core

def fourier_spectrum(model):
    """Exact Fourier coefficients of the trained quantum core g(u) on its 8
    embedded angles (spectrum {-1,0,1}^8 for single RY encoding).

    Replicates the repo circuit (AngleEmbedding rotation='Y', 3-layer
    SimplifiedTwoDesign with zero initial layer, <Z_1> readout) with the
    trained weights, verifies it against the model's own quantum layer,
    samples the {0, 2pi/3, 4pi/3}^8 grid in one broadcast call, and FFTs."""
    import pennylane as qml
    sd = model.state_dict()
    W = sd["quantum_layer.weights"].numpy()

    dev = qml.device("default.qubit", wires=N_FEAT)

    @qml.qnode(dev)
    def g(u):
        qml.templates.AngleEmbedding(u, wires=range(N_FEAT), rotation="Y")
        qml.SimplifiedTwoDesign(initial_layer_weights=np.zeros(N_FEAT),
                                weights=W, wires=range(N_FEAT))
        return qml.expval(qml.PauliZ(1))           # wire 1, as in the repo

    # replication check against the model's own TorchLayer
    u_test = np.random.default_rng(0).uniform(-2, 2, (16, N_FEAT))
    ours = np.array([float(g(u)) for u in u_test])
    with torch.no_grad():
        theirs = model.quantum_layer(
            torch.tensor(u_test, dtype=torch.float32)).numpy().ravel()
    repl_err = float(np.abs(ours - theirs).max())
    if repl_err > 1e-4:
        raise SystemExit(f"circuit replication failed (max err {repl_err:.2e})")

    pts = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    grid = np.array(list(iproduct(pts, repeat=N_FEAT)))       # [6561, 8]
    vals = np.array([float(g(u)) for u in grid]).reshape((3,) * N_FEAT)
    coeff = np.fft.fftn(vals) / vals.size                     # freqs {0,1,-1}

    # reconstruction check at random points
    freqs = np.array(list(iproduct([0, 1, -1], repeat=N_FEAT)))
    flat = coeff.ravel()
    order = np.abs(freqs).sum(axis=1)
    u_chk = np.random.default_rng(1).uniform(0, 2 * np.pi, (8, N_FEAT))
    recon = np.real(flat[None, :] * np.exp(1j * u_chk @ freqs.T)).sum(axis=1)
    exact = np.array([float(g(u)) for u in u_chk])
    recon_err = float(np.abs(recon - exact).max())

    mass = np.abs(flat) ** 2
    by_order = {int(k): float(mass[order == k].sum() / mass.sum())
                for k in range(N_FEAT + 1)}
    n_active = int((np.abs(flat) > 1e-9).sum())
    return dict(mass_by_interaction_order=by_order,
                active_modes=n_active, total_modes=len(flat),
                replication_max_err=repl_err, reconstruction_max_err=recon_err)


# ------------------------------------------------------------ 4D contrast

def cost_4d():
    """The same explainers priced for the 4D amplitude-embedded model."""
    m = 2 ** 20                                    # features (amplitudes)
    n_wires = 20
    shap_coalitions_log10 = (m - 1) * math.log10(2)
    ig_evals_hw = 2 * m * IG_NODES                 # FD needs 2m evals per node
    stateprep = 2 ** n_wires - n_wires - 1
    return dict(
        features=m,
        exact_shap_coalitions="10^%d" % round(shap_coalitions_log10),
        qshap_applicable=False,
        qshap_note=("qSHAP (arXiv:2211.01441) requires rotation-encoded "
                    "features so the model is a truncated Fourier series in "
                    "them (Schuld et al. theorem); amplitude encoding enters "
                    "through state preparation, not rotations, so the "
                    "framework does not apply — and even if it did, qSHAP "
                    "scales in the number of FEATURES, which amplitude "
                    "embedding makes exponential in qubit count."),
        ig_simulation=("feasible: d<Z>/d(amplitudes) for all 2^20 inputs is "
                       "one backward pass per path node in simulation"),
        ig_hardware_circuit_execs=ig_evals_hw,
        ig_hardware_cnots=ig_evals_hw * stateprep,
        ig_hardware_note=("state-prep inputs admit no parameter-shift rule, "
                          "so hardware IG needs finite differences: 2m "
                          "executions per node, each paying the O(2^n) "
                          "state-preparation cost"),
        group_shap_path=("attribute groups, not amplitudes: dates (4 groups, "
                         "2^4 coalitions) or octant-x-date supervoxels "
                         "(32 groups, sampled SHAP) — the standard rescue, "
                         "at the price of coarse explanations"),
    )


# ------------------------------------------------------------ figure

def make_figure(res, groups_order=("TP", "FP", "FN", "TN")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
    MUTED, GRID = "#898781", "#e1e0d9"
    S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": "#c3c2b7",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    feat_labels = [f"d1·pc{i}" for i in range(1, 5)] + \
                  [f"d2·pc{i}" for i in range(1, 5)]
    ax = axes[0]
    xs = np.arange(N_FEAT)
    w = 0.2
    cols = [S1, S2, S3, "#c3c2b7"]
    for i, g in enumerate(groups_order):
        vals = res["groups"][g]["mean_abs_shap"]
        ax.bar(xs + (i - 1.5) * w, vals, w * 0.9, color=cols[i], label=g, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(feat_labels, rotation=45, fontsize=8.5)
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    ax.set_title("Mean |SHAP| per feature, by outcome", color=INK, loc="left")
    ax.xaxis.grid(False)

    ax = axes[1]
    sym = [res["groups"][g]["symmetric_share"] for g in groups_order]
    ax.bar(range(4), sym, 0.55, color=S2, zorder=3,
           label="symmetric (spectral level, 'urban-ness')")
    ax.bar(range(4), [1 - s for s in sym], 0.55, bottom=sym, color=S1,
           zorder=3, label="antisymmetric (between-date contrast)")
    ax.axhline(0.5, color=INK, lw=1, ls=(0, (3, 3)))
    for i, s in enumerate(sym):
        ax.text(i, s / 2, f"{100*s:.0f}%", ha="center", va="center",
                color="#0b0b0b", fontsize=9.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(groups_order)
    ax.set_ylim(0, 1.18); ax.legend(fontsize=8.6, loc="upper left")
    ax.set_title("Attribution symmetry: level vs change", color=INK, loc="left")
    ax.xaxis.grid(False)

    ax = axes[2]
    orders = sorted(res["fourier"]["mass_by_interaction_order"])
    mass = [res["fourier"]["mass_by_interaction_order"][k] for k in orders]
    ax.bar(orders, mass, 0.7, color=S1, zorder=3)
    ax.set_xlabel("interaction order (non-zero frequencies)")
    ax.set_title("Fourier mass of the trained quantum core", color=INK,
                 loc="left")
    ax.xaxis.grid(False)

    fig.suptitle("xAI on the trained OSCD QNN — exact SHAP, IG, and the "
                 "Fourier structure behind qSHAP (arXiv:2211.01441)",
                 x=0.055, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(HERE, "fig7_xai.png")
    fig.savefig(out, dpi=200)
    print(f"  wrote {out}")


# ------------------------------------------------------------ main

def main():
    rng = np.random.default_rng(SEED)
    print("[1/5] loading model and pixels ...")
    model = load_model()
    groups, alignment = gather_pixels(rng)

    # baseline: the mean correctly-rejected unchanged pixel
    b = groups["TN"].mean(axis=0).astype(np.float64)

    res = dict(
        model_run="overjoyed-doe-477",
        n_per_group={g: int(len(v)) for g, v in groups.items()},
        baseline="mean of sampled TN pixels",
        pca_alignment=alignment,
        symmetric_share_null=0.5,
        checks={}, groups={},
    )

    print("[2/5] exact Baseline SHAP (256 coalitions/pixel) ...")
    t0 = time.perf_counter()
    max_eff = 0.0
    for g, X in groups.items():
        shap, eff = shap_exact(model, X.astype(np.float64), b)
        max_eff = max(max_eff, eff)
        sym = 0.5 * (shap[:, :4] + shap[:, 4:])
        asym = 0.5 * (shap[:, :4] - shap[:, 4:])
        res["groups"][g] = dict(
            mean_abs_shap=np.abs(shap).mean(axis=0).tolist(),
            mean_shap=shap.mean(axis=0).tolist(),
            symmetric_share=float(np.abs(sym).sum()
                                  / (np.abs(sym).sum() + np.abs(asym).sum())),
        )
    res["checks"]["shap_efficiency_max_err"] = float(max_eff)
    print(f"  done in {time.perf_counter()-t0:.1f}s — "
          f"efficiency axiom max err {max_eff:.2e}")

    print("[3/5] Integrated Gradients (paper eq. 2) ...")
    t0 = time.perf_counter()
    ig_comp = {}
    for g, X in groups.items():
        ig, comp = integrated_gradients(model, X.astype(np.float64), b)
        res["groups"][g]["mean_abs_ig"] = np.abs(ig).mean(axis=0).tolist()
        # rank agreement between the two explainers
        sh = np.array(res["groups"][g]["mean_abs_shap"])
        igm = np.abs(ig).mean(axis=0)
        rho = float(np.corrcoef(np.argsort(np.argsort(sh)),
                                np.argsort(np.argsort(igm)))[0, 1])
        res["groups"][g]["ig_shap_rank_corr"] = rho
        ig_comp[g] = comp
    res["checks"]["ig_completeness_mean_err"] = {k: float(v)
                                                for k, v in ig_comp.items()}
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    print("[4/5] Fourier spectrum of the quantum core (3^8 grid) ...")
    res["fourier"] = fourier_spectrum(model)
    fr = res["fourier"]
    print(f"  replication err {fr['replication_max_err']:.1e}, "
          f"reconstruction err {fr['reconstruction_max_err']:.1e}, "
          f"{fr['active_modes']}/{fr['total_modes']} modes active")

    print("[5/5] pricing the same explainers for the 4D model ...")
    res["cost_4d"] = cost_4d()
    print(f"  exact SHAP coalitions at 2^20 features: "
          f"{res['cost_4d']['exact_shap_coalitions']}")

    out = os.path.join(HERE, "xai_results.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote {out}")
    make_figure(res)

    print("\nsummary:")
    for g in ("TP", "FP", "FN", "TN"):
        r = res["groups"][g]
        print(f"  {g}: symmetric share {100*r['symmetric_share']:.0f}%   "
              f"IG/SHAP rank corr {r['ig_shap_rank_corr']:.2f}")


if __name__ == "__main__":
    main()
