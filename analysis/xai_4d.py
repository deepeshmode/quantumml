"""Group-level SHAP for the 4D amplitude-embedded QNN.

Per-feature explanation of the 20-wire model is intractable (2^20 features;
xai_oscd.py prices exact SHAP at ~10^315,653 coalitions, and the Fourier
structure behind qSHAP does not exist for amplitude encoding). The standard
rescue is GROUP attribution: treat semantically meaningful blocks of inputs
as single players. Exact Shapley over G groups costs only 2^G coalition
forwards, so it stays exact for the two decompositions that matter here:

  dates    — players are dates 1..3 (date 0 defines the baseline), 2^3 = 8
             coalitions: how much does each date's deviation from "nothing
             ever changed" drive the change score?
  octants  — players are the 8 spatial octants of the volume (their full
             time series), 2^8 = 256 coalitions: is the score driven by the
             octant that actually contains the edit, or diffuse?

The baseline is the frozen-time counterfactual: the sequence's own first
frame tiled across all dates. f(b) is then the model's answer to "this
exact structure, never changing" — the cleanest no-change reference
available, and per-sequence rather than global.

The Colab run did not save weights, so a twin Arm B is trained locally with
the same code path, config and seed (dataset is procedural here — ModelNet
is a 450 MB download; provenance is recorded, and the twin's accuracy is
reported next to the Colab run's). Explaining a model whose test accuracy
is at chance is still informative mechanically: the attribution shows WHAT
the circuit responds to — if octant attributions ignore the edit location,
the model is reading bulk occupancy, exactly the shortcut the count
baseline was built to expose.

Run:  ~/Downloads/qnn_change_detection/.venv/bin/python xai_4d.py
Outputs: xai_4d_results.json, fig9_xai_4d.png (next to this script).
"""

import json
import math
import os
import sys
import time
import warnings
from itertools import product as iproduct

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WORK = os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "tmp", "xai4d")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)                     # dataset cache lands here, not in the repo

import numpy as np
import torch

import experiment_4d_colab as e4

N_OCTANT_SEQS = 8                  # sequences given the 256-coalition octant pass
SEED = 7                           # same as the Colab run


def shapley_groups(f, x, b, group_masks):
    """Exact Shapley values over G groups for a scalar model f.

    group_masks: list of boolean arrays over the flat input; coalition S
    takes x on the union of its groups' masks and b elsewhere. Returns
    (values, efficiency_error)."""
    G = len(group_masks)
    fact = [math.factorial(k) for k in range(G + 1)]
    coalitions = list(iproduct([0, 1], repeat=G))
    vecs = []
    for c in coalitions:
        g = b.copy()
        for i, on in enumerate(c):
            if on:
                g[group_masks[i]] = x[group_masks[i]]
        vecs.append(g)
    vals = f(np.stack(vecs))
    v = {c: vals[i] for i, c in enumerate(coalitions)}
    sh = np.zeros(G)
    for e in range(G):
        for c in coalitions:
            if c[e] == 1:
                continue
            s = sum(c)
            c_on = tuple(ci if i != e else 1 for i, ci in enumerate(c))
            sh[e] += fact[s] * fact[G - s - 1] / fact[G] * (v[c_on] - v[c])
    eff = abs(sh.sum() - (v[tuple([1] * G)] - v[tuple([0] * G)]))
    return sh, eff


def main():
    cfg = dict(e4.DEFAULTS)
    cfg["no_download"] = True      # procedural structures; provenance recorded
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    import pennylane as qml
    print("[1/4] dataset (same generator/config/seed as the Colab run,")
    print("       procedural bases — ModelNet is a 450 MB download) ...")
    cached = e4.load_cache(cfg)
    if cached:
        train_seq, test_seq, provenance = cached
    else:
        train_seq, test_seq, provenance = e4.build_dataset(cfg, rng)
        e4.save_cache(train_seq, test_seq, provenance, cfg)
    print(f"  {len(train_seq)} train / {len(test_seq)} test ({provenance})")

    n_feat = cfg["dates"] * cfg["res"] ** 3
    n_wires = max(1, math.ceil(math.log2(n_feat)))
    dev = qml.device("lightning.qubit", wires=n_wires)
    model, _ = e4.make_arm_b_model(n_feat, cfg["layers"], dev, torch)

    print(f"[2/4] training the twin Arm B locally ({n_wires} wires, CPU) ...")
    Xtr = np.stack([s["seq"].ravel() for s in train_seq]).astype(np.float32)
    ytr = np.array([s["label"] for s in train_seq], dtype=np.float32)
    Xte = np.stack([s["seq"].ravel() for s in test_seq]).astype(np.float32)
    yte = np.array([s["label"] for s in test_seq], dtype=np.float32)
    twin = e4.train_binary(model, Xtr, ytr, Xte, yte, cfg["epochs_b"],
                           cfg["batch_b"], 5e-2, "adam", torch)
    torch.save(model.state_dict(), os.path.join(HERE, "arm_b_twin_weights.pt"))
    print(f"  twin acc {twin['accuracy']:.3f} F1 {twin['f1']:.3f} "
          f"(Colab run: acc 0.500 F1 0.400)")

    model.eval()

    def f(batch):
        with torch.no_grad():
            return model(torch.tensor(batch, dtype=torch.float32)).numpy().ravel()

    res, dates, res3 = cfg["res"], cfg["dates"], cfg["res"] ** 3
    idx = np.arange(n_feat).reshape(dates, res, res, res)

    # date groups: dates 1..3 (date 0 is identical to the frozen baseline)
    date_masks = []
    for t in range(1, dates):
        m = np.zeros(n_feat, dtype=bool)
        m[idx[t].ravel()] = True
        date_masks.append(m)

    # octant groups: each octant's full time series
    h = res // 2
    oct_masks, oct_keys = [], []
    for ox, oy, oz in iproduct([0, 1], repeat=3):
        m = np.zeros(n_feat, dtype=bool)
        m[idx[:, ox*h:(ox+1)*h, oy*h:(oy+1)*h, oz*h:(oz+1)*h].ravel()] = True
        oct_masks.append(m)
        oct_keys.append((ox, oy, oz))

    print("[3/4] exact group-SHAP (frozen-time baseline per sequence) ...")
    t0 = time.perf_counter()
    date_rows, oct_rows, max_eff = [], [], 0.0
    oct_done = {0: 0, 1: 0}
    for si, s in enumerate(test_seq):
        x = s["seq"].ravel().astype(np.float32)
        b = np.tile(s["seq"][0], (dates, 1, 1, 1)).ravel().astype(np.float32)
        sh, eff = shapley_groups(f, x, b, date_masks)
        max_eff = max(max_eff, eff)
        date_rows.append(dict(label=int(s["label"]),
                              when=s.get("when"), kind=s.get("kind"),
                              shap_dates=[float(v) for v in sh],
                              f_x=float(f(x[None])[0]), f_b=float(f(b[None])[0])))
        # octant pass on a balanced subset (256 coalitions each)
        if oct_done[s["label"]] < N_OCTANT_SEQS // 2:
            sh_o, eff_o = shapley_groups(f, x, b, oct_masks)
            max_eff = max(max_eff, eff_o)
            # which octants actually contain changed voxels?
            diff = (s["seq"][0] != s["seq"][-1])
            edit_frac = []
            for (ox, oy, oz) in oct_keys:
                d = diff[ox*h:(ox+1)*h, oy*h:(oy+1)*h, oz*h:(oz+1)*h]
                edit_frac.append(float(d.mean()))
            oct_rows.append(dict(label=int(s["label"]), kind=s.get("kind"),
                                 shap_octants=[float(v) for v in sh_o],
                                 edit_fraction=edit_frac))
            oct_done[s["label"]] += 1
        if (si + 1) % 6 == 0:
            print(f"  {si+1}/{len(test_seq)} sequences "
                  f"({time.perf_counter()-t0:.0f}s)")
    print(f"  done in {time.perf_counter()-t0:.0f}s — "
          f"efficiency axiom max err {max_eff:.2e}")

    # spatial specificity: within octant-passed CHANGE sequences, does
    # attribution rank the edit-bearing octant first?
    spec = []
    for r in oct_rows:
        if r["label"] != 1 or max(r["edit_fraction"]) == 0:
            continue
        target = int(np.argmax(r["edit_fraction"]))
        rank = int(np.argsort(-np.abs(r["shap_octants"])).tolist().index(target))
        spec.append(dict(edit_octant=target, attribution_rank=rank,
                         top_octant=int(np.argmax(np.abs(r["shap_octants"])))))

    print("[4/4] writing outputs ...")
    out = dict(
        note=("Twin of the Colab Arm B (same code/config/seed, procedural "
              "bases, local lightning.qubit); Colab weights were not saved. "
              "Group SHAP is exact over its groups; the per-amplitude "
              "problem stays intractable by design."),
        twin_metrics={k: twin[k] for k in
                      ("accuracy", "precision", "recall", "f1", "confusion")},
        colab_metrics=dict(accuracy=0.5, f1=0.4),
        provenance=provenance, n_wires=n_wires,
        efficiency_axiom_max_err=float(max_eff),
        date_shap=date_rows, octant_shap=oct_rows,
        spatial_specificity=spec,
        coalition_costs=dict(dates=2 ** len(date_masks),
                             octants=2 ** len(oct_masks),
                             per_amplitude="10^315653 (intractable)"),
    )
    with open(os.path.join(HERE, "xai_4d_results.json"), "w") as fjs:
        json.dump(out, fjs, indent=2)

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
    MUTED, GRID = "#898781", "#e1e0d9"
    S1, S2 = "#2a78d6", "#eb6834"
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "text.color": INK, "axes.labelcolor": INK_2,
        "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    for lab, col, name in [(1, S2, "change sequences"),
                           (0, S1, "no-change sequences")]:
        rows = [r["shap_dates"] for r in date_rows if r["label"] == lab]
        m = np.abs(np.array(rows)).mean(axis=0)
        xs = np.arange(1, dates)
        ax.bar(xs + (0.19 if lab else -0.19), m, 0.34, color=col,
               label=name, zorder=3)
    ax.set_xticks(range(1, dates))
    ax.set_xticklabels([f"date {t}" for t in range(1, dates)])
    ax.set_ylabel("mean |group SHAP|")
    ax.legend(fontsize=9)
    ax.set_title("Date-group attribution vs the frozen-time baseline",
                 color=INK, loc="left")
    ax.xaxis.grid(False)

    ax = axes[1]
    chg = [r for r in oct_rows if r["label"] == 1]
    for i, r in enumerate(chg):
        sh = np.abs(r["shap_octants"])
        ef = np.array(r["edit_fraction"])
        order = np.argsort(-ef)
        ax.plot(range(8), sh[order] / (sh.max() or 1), "-o", ms=4.5, lw=1.6,
                color=[S2, "#1baf7a", "#eda100", "#e87ba4"][i % 4],
                label=f"seq {i+1} ({r['kind']})")
    ax.set_xticks(range(8))
    ax.set_xticklabels(["edit\noctant"] + [f"{k}" for k in range(2, 9)],
                       fontsize=8.5)
    ax.set_xlabel("octants, sorted by actual changed-voxel fraction")
    ax.set_ylabel("|SHAP| (normalised per sequence)")
    ax.legend(fontsize=8.2)
    ax.set_title("Spatial specificity: does attribution find the edit?",
                 color=INK, loc="left")
    ax.xaxis.grid(False)

    fig.suptitle("Group-level SHAP for the 20-wire 4D model — exact over "
                 "groups, where per-amplitude xAI is impossible",
                 x=0.055, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(HERE, "fig9_xai_4d.png"), dpi=200)
    print("  wrote xai_4d_results.json and fig9_xai_4d.png")

    if spec:
        ranks = [s["attribution_rank"] for s in spec]
        print(f"\nspatial specificity: edit-octant attribution rank per "
              f"change sequence: {ranks} (0 = top)")


if __name__ == "__main__":
    main()
