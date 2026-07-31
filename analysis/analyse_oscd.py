"""Deep-dive analysis of the trained OSCD change-detection QNN.

Runs full-resolution inference on the three held-out cities, then produces:
  fig1_training_dynamics.png  - train/val accuracy and loss over 20 epochs
  fig2_change_maps.png        - probability, ground truth and error map per city
  fig3_threshold.png          - precision/recall/F1 vs decision threshold, PR curve
  fig4_per_city.png           - metric breakdown by city
  fig5_separability.png       - are the input features even separable?
plus metrics.json with every number the figures assert.
"""
import os, sys, json, time, sqlite3, warnings
warnings.filterwarnings("ignore")

REPO = os.path.expanduser("~/Downloads/qnn_change_detection")
OUT = os.path.join(REPO, "analysis")
sys.path.insert(0, REPO)
os.environ.setdefault("SIPWQNN_DATA_PATH", os.path.expanduser("~/Downloads/oscd_root"))
os.makedirs(OUT, exist_ok=True)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
import torchvision.transforms as tr
from src.oscd_dataloader import ChangeDetectionDataset

# ---------------------------------------------------------------- palette
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
AXIS      = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"   # validated all-pairs (CVD dE 9.2)
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGE   = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a9a8", "#e34948", "#8f2020"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})

ART = os.path.join(REPO, "mlruns/1/20cd6303631a4aa2bde0029415b45191/artifacts/model/data/model.pth")
RUN_UUID = "20cd6303631a4aa2bde0029415b45191"

# ---------------------------------------------------------------- 1. curves
con = sqlite3.connect(os.path.join(REPO, "mlflow.db"))
def series(key):
    rows = con.execute(
        "SELECT step, value FROM metrics WHERE key=? AND run_uuid=? ORDER BY step",
        (key, RUN_UUID)).fetchall()
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])

ep, tr_acc = series("train_acc")
_,  va_acc = series("val_acc")
_,  tr_los = series("train_loss")
_,  va_los = series("val_loss")
best_ep = int(ep[np.argmax(va_acc)]); best_va = float(va_acc.max())
print(f"best val acc {best_va:.4f} @ epoch {best_ep};  final {va_acc[-1]:.4f}")

# ---------------------------------------------------------------- 2. inference
print("building test dataset (3 cities)...")
ds = ChangeDetectionDataset(
    os.environ["SIPWQNN_DATA_PATH"], train=False, patch_side=1, stride=10,
    transform=tr.Compose([]), band_type=2, normalise_img=True,
    fp_modifier=1, n_components=4,
)
model = torch.load(ART, map_location="cpu", weights_only=False)
model.eval()

cities, results = list(ds.imgs_1.keys()), {}
CACHE = f"{OUT}/.inference_cache.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE)
    for city in cities:
        results[city] = {"prob": z[f"{city}_prob"], "gt": z[f"{city}_gt"],
                         "featdiff": z[f"{city}_fd"]}
    print("loaded cached inference")
    cities = list(results.keys())
for city in ([] if results else cities):
    I1 = np.asarray(ds.imgs_1[city]); I2 = np.asarray(ds.imgs_2[city])
    gt = np.asarray(ds.change_maps[city]).astype(np.uint8)
    C, H, W = I1.shape
    x1 = torch.from_numpy(I1.reshape(C, -1).T.astype(np.float32))
    x2 = torch.from_numpy(I2.reshape(C, -1).T.astype(np.float32))

    t0, probs = time.time(), []
    with torch.no_grad():
        for i in range(0, x1.shape[0], 1000):
            probs.append(model(x1[i:i+1000], x2[i:i+1000]).numpy())
    p = np.concatenate(probs).reshape(H, W)
    print(f"  {city:<10} {H}x{W} = {H*W:>9,} px  in {time.time()-t0:6.1f}s")

    results[city] = {
        "prob": p, "gt": gt,
        "featdiff": np.linalg.norm(I1 - I2, axis=0),
    }

if not os.path.exists(CACHE):
    np.savez_compressed(CACHE, **{
        f"{c}_{k}": results[c][{"prob": "prob", "gt": "gt", "fd": "featdiff"}[k]]
        for c in cities for k in ("prob", "gt", "fd")})

def confusion(gt, pred):
    tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum()); tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                accuracy=(tp + tn) / (tp + tn + fp + fn))

all_p  = np.concatenate([results[c]["prob"].ravel() for c in cities])
all_gt = np.concatenate([results[c]["gt"].ravel() for c in cities])

ths = np.linspace(0.05, 0.95, 91)
sweep = [confusion(all_gt, (all_p >= t).astype(np.uint8)) for t in ths]
f1s = np.array([s["f1"] for s in sweep])
best_t = float(ths[np.argmax(f1s)]); best_f1 = float(f1s.max())
at_half = confusion(all_gt, (all_p >= 0.5).astype(np.uint8))
print(f"aggregate @0.50: F1 {at_half['f1']:.4f}   best F1 {best_f1:.4f} @ threshold {best_t:.2f}")

per_city = {c: confusion(results[c]["gt"], (results[c]["prob"] >= 0.5).astype(np.uint8))
            for c in cities}
for c, m in per_city.items():
    print(f"  {c:<10} acc {m['accuracy']:.3f}  P {m['precision']:.3f}  "
          f"R {m['recall']:.3f}  F1 {m['f1']:.3f}  changed {100*m['tp']+0:.0f}")

# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
for ax, (a, b, la, lb, ttl) in zip(axes, [
        (tr_acc, va_acc, "train", "held-out", "Accuracy"),
        (tr_los, va_los, "train", "held-out", "BCE loss")]):
    ax.plot(ep, a, color=S1, lw=2, label=la)
    ax.plot(ep, b, color=S2, lw=2, label=lb)
    ax.set_xlabel("epoch"); ax.set_title(ttl, color=INK, loc="left")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.text(ep[-1] + 0.3, a[-1], la, color=INK_2, va="center", fontsize=9)
    ax.text(ep[-1] + 0.3, b[-1], lb, color=INK_2, va="center", fontsize=9)
    ax.set_xlim(0.5, 23.5)
axes[0].axvline(best_ep, color=MUTED, lw=1, ls=(0, (3, 3)))
axes[0].annotate(f"best held-out {best_va:.3f} @ epoch {best_ep}\nreported value is epoch 20 ({va_acc[-1]:.3f})",
                 xy=(best_ep, best_va), xytext=(7.4, 0.775), fontsize=9, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
axes[0].axvspan(1, 9, color=GRID, alpha=0.55, lw=0)
axes[0].text(5, 0.60, "9 epochs flat", ha="center", fontsize=9, color=MUTED)
fig.suptitle("Training dynamics — pixel-level QNN on OSCD, 20 epochs",
             x=0.062, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig1_training_dynamics.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 2
seq = LinearSegmentedColormap.from_list("blues", BLUE_RAMP)
err_cmap = ListedColormap(["#eeeeea", S1, S2, S3])   # TN, TP, FP, FN
fig, axes = plt.subplots(len(cities), 3, figsize=(11.5, 3.7 * len(cities)))
for r, city in enumerate(cities):
    p, gt = results[city]["prob"], results[city]["gt"]
    pred = (p >= 0.5).astype(np.uint8)
    err = np.zeros_like(gt); err[(pred == 1) & (gt == 1)] = 1
    err[(pred == 1) & (gt == 0)] = 2; err[(pred == 0) & (gt == 1)] = 3
    m = per_city[city]
    for c, (img, cmap, ttl, vmax) in enumerate([
            (p, seq, "predicted change probability", 1),
            (gt, ListedColormap(["#eeeeea", INK]), "ground truth", 1),
            (err, err_cmap, "errors", 3)]):
        ax = axes[r, c]; ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0: ax.set_title(ttl, color=INK_2, fontsize=10, loc="left")
    axes[r, 0].set_ylabel(f"{city}\n{gt.shape[0]}×{gt.shape[1]}", color=INK,
                          fontsize=10.5, rotation=0, ha="right", va="center", labelpad=34)
    axes[r, 2].text(1.02, 0.5, f"F1 {m['f1']:.3f}\nprecision {m['precision']:.3f}\n"
                    f"recall {m['recall']:.3f}", transform=axes[r, 2].transAxes,
                    fontsize=9, color=INK_2, va="center")
fig.legend(handles=[Patch(facecolor=S1, label="true positive — change found"),
                    Patch(facecolor=S2, label="false positive — false alarm"),
                    Patch(facecolor=S3, label="false negative — change missed"),
                    Patch(facecolor="#eeeeea", label="true negative")],
           loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, -0.004))
fig.suptitle("Full-resolution predictions on the three held-out cities (threshold 0.5)",
             x=0.045, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=[0.02, 0.035, 1, 0.955])
fig.savefig(f"{OUT}/fig2_change_maps.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- fig 3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ax = axes[0]
ax.plot(ths, [s["precision"] for s in sweep], color=S1, lw=2, label="precision")
ax.plot(ths, [s["recall"] for s in sweep], color=S2, lw=2, label="recall")
ax.plot(ths, f1s, color=S3, lw=2, label="F1")
ax.legend(loc="upper right", fontsize=9.5, ncol=1)
ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.axvline(best_t, color=INK, lw=1.2)
ax.annotate(f"F1-optimal {best_t:.2f}\nF1 {best_f1:.3f}", xy=(best_t, best_f1),
            xytext=(best_t + 0.05, 0.40), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.text(0.5, 0.03, " default 0.5", color=MUTED, fontsize=9)
ax.set_xlabel("decision threshold"); ax.set_ylim(0, 1.05); ax.set_xlim(0.05, 0.95)
ax.set_title("Metrics vs threshold", color=INK, loc="left")

ax = axes[1]
ax.plot([s["recall"] for s in sweep], [s["precision"] for s in sweep], color=S1, lw=2)
ax.scatter([at_half["recall"]], [at_half["precision"]], s=64, color=INK, zorder=5)
ax.annotate(f"threshold 0.5\nF1 {at_half['f1']:.3f}",
            xy=(at_half["recall"], at_half["precision"]),
            xytext=(at_half["recall"] - 0.34, at_half["precision"] + 0.13), fontsize=9,
            color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
base = all_gt.mean()
ax.axhline(base, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(0.02, base + 0.015, f"no-skill baseline ({base:.3f} of pixels changed)",
        color=MUTED, fontsize=9)
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1); ax.set_xlim(0, 1)
ax.set_title("Precision–recall", color=INK, loc="left")
fig.suptitle("The 0.5 threshold is not the operating point that maximises F1",
             x=0.055, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f"{OUT}/fig3_threshold.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 4
fig, ax = plt.subplots(figsize=(8.4, 4.2))
metrics = ["accuracy", "precision", "recall", "f1"]
xs = np.arange(len(metrics)); w = 0.26
for i, (city, col) in enumerate(zip(cities, [S1, S2, S3])):
    vals = [per_city[city][m] for m in metrics]
    ax.bar(xs + (i - 1) * w, vals, w * 0.92, color=col, label=city, zorder=3)
    for x, v in zip(xs + (i - 1) * w, vals):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5, color=INK_2)
ax.set_xticks(xs); ax.set_xticklabels(["accuracy", "precision", "recall", "F1"])
ax.set_ylim(0, 1); ax.legend(ncol=3, loc="upper right", fontsize=9.5)
ax.set_title("Per-city performance at threshold 0.5 — the aggregate hides the spread",
             color=INK, loc="left", fontsize=12)
ax.xaxis.grid(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_per_city.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- fig 5
fig, axes = plt.subplots(1, len(cities), figsize=(4.0 * len(cities), 3.9), sharey=True)
for ax, city in zip(np.atleast_1d(axes), cities):
    d, gt = results[city]["featdiff"].ravel(), results[city]["gt"].ravel()
    bins = np.linspace(0, np.percentile(d, 99), 60)
    ax.hist(d[gt == 0], bins=bins, color=S1, alpha=0.75, density=True,
            label="unchanged", lw=0)
    ax.hist(d[gt == 1], bins=bins, color=S2, alpha=0.75, density=True,
            label="changed", lw=0)
    med0, med1 = np.median(d[gt == 0]), np.median(d[gt == 1])
    ax.set_title(f"{city}   medians {med0:.2f} / {med1:.2f}", color=INK_2,
                 fontsize=10, loc="left")
    ax.set_xlabel("‖PCA(t₂) − PCA(t₁)‖₂"); ax.yaxis.grid(True); ax.xaxis.grid(False)
np.atleast_1d(axes)[0].set_ylabel("density")
np.atleast_1d(axes)[0].legend(fontsize=9.5)
fig.suptitle("Do the input features separate change from no-change before the QNN sees them?",
             x=0.02, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(f"{OUT}/fig5_separability.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- json
json.dump({
    "run": "overjoyed-doe-477",
    "best_epoch": best_ep, "best_val_acc": best_va, "final_val_acc": float(va_acc[-1]),
    "aggregate_at_0.5": at_half,
    "best_threshold": best_t, "best_threshold_f1": best_f1,
    "per_city_at_0.5": per_city,
    "changed_fraction": float(base),
    "note": "full-resolution inference over every pixel of the 3 held-out cities",
}, open(f"{OUT}/metrics.json", "w"), indent=2)
print("\nwrote figures + metrics.json to", OUT)
