"""4D change detection (3D structures + time) with the paper's QNN, on GPU.

WHAT THIS IS
------------
The OSCD work is 2D pixels at two dates. The supervisor's forward-looking
question is 4D: 3D structures evolving in time. This experiment lifts the
paper's pixel-level QNN (arXiv:2503.08962: linear layer -> embedding ->
SimplifiedTwoDesign -> <Z> -> sigmoid) to 4D voxel data and measures what
happens — including where GPU simulation helps and where it cannot.

Two arms, chosen from the wire-budget analysis (see wire_budget.py):

  Arm A — paper-faithful. Per-voxel classification. Each voxel is described
          by its 3x3x3 neighbourhood at first and last date, PCA-reduced to
          4 components per date, angle-embedded on 8 wires. Same architecture
          and width as the 2D OSCD run. Training draws a class-stratified
          sample through the paper's inverse-frequency sampler; EVALUATION is
          a uniform voxel sample at natural prevalence, with the balanced-
          resample view reported alongside — the 2D reproduction showed the
          balanced resample is what inflated F1 from 0.153 to 0.691.
          8 wires is far below the GPU crossover: the GPU is expected to
          LOSE here, and that measurement is part of the point.

  Arm B — GPU-justified. Whole-volume classification. The full 4-date voxel
          sequence (4 x res^3 occupancy values) is amplitude-embedded on
          log2(4 * res^3) wires — 20 wires at res=64, the width where GPU
          statevector simulation starts to pay.

          Two caveats are load-bearing and recorded in the JSON. First,
          amplitude state preparation costs O(2^n) CNOTs on real hardware
          (2^n - n - 1), so this arm is simulable, not executable. Second,
          because edits add or remove voxel mass and the flattening is
          date-major, total-occupancy differences between early and late
          dates separate the classes on their own — and <Z> on wire 0 (the
          most significant address bit) reads exactly that early-vs-late
          mass statistic. A count-only logistic baseline is therefore
          trained on per-date occupancy totals and reported alongside:
          Arm B only evidences spatial change detection to the extent it
          beats that baseline.

DATA
----
Real 3D structures: ModelNet10 (Princeton, ~450 MB zip of CAD meshes),
voxelized to occupancy shells. Temporal evolution is SYNTHESIZED — half the
sequences get a structural edit (a cuboid of the structure demolished, or a
new cuboid built) at a random date, half get only occupancy noise. Edits are
redrawn until they change a minimum number of voxels, so a label=1 sequence
is guaranteed to contain real change (naive uniform placement makes the
demolition a no-op on hollow shells more than half the time). Ground truth
masks are computed directly as clean-first vs clean-last differences, at
each resolution an arm consumes — never max-pooled across resolutions.
If the download fails, procedural structures (unions of cuboids) are used;
provenance is recorded in the JSON either way.

No public 4D change-detection benchmark with voxel labels exists; synthetic
time is the honest option and is labelled as such everywhere.

RUNNING
-------
Colab (T4 GPU runtime), two cells:

    !nvidia-smi
    !pip install -q pennylane pennylane-lightning pennylane-lightning-gpu custatevec-cu12
    # Runtime -> Restart session, then:

    !curl -fsSL -o experiment_4d.py https://raw.githubusercontent.com/deepeshmode/quantumml/main/analysis/experiment_4d_colab.py && python experiment_4d.py

(-f makes a 404 fail loudly instead of leaving an empty file behind.)

Local smoke test (CPU, ~2 min):  python experiment_4d_colab.py --smoke

experiment_4d_results.json is (re)written after every stage, so a Colab
disconnect mid-run keeps everything finished so far; the built dataset is
cached to disk and reused on rerun. Accuracy is allowed to be poor; the
deliverable is the measurement, not a leaderboard number.
"""

import argparse
import json
import math
import os
import platform
import subprocess
import time
import urllib.request
import warnings
import zipfile

warnings.filterwarnings("ignore")

import numpy as np

# Heavy imports are deferred until after arg parsing so --help is instant.


# ============================================================ configuration

MODELNET_URLS = [
    "https://modelnet.cs.princeton.edu/ModelNet10.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
]

DEFAULTS = dict(
    res=64,              # voxel resolution for Arm B (64^3 * 4 dates -> 20 wires)
    res_a=16,            # resolution Arm A operates at (own noise, own mask)
    dates=4,             # timestamps per sequence
    n_train_seq=60,      # sequences for Arm B training
    n_test_seq=24,
    n_voxel_train=3000,  # stratified per-voxel TRAINING samples for Arm A
    n_voxel_test=4000,   # uniform per-voxel TEST samples (natural prevalence)
    epochs_a=3,
    epochs_b=4,
    batch_a=64,
    batch_b=8,
    layers=3,            # SimplifiedTwoDesign layers, as in the paper
    noise=0.004,         # occupancy flip probability per voxel per date
    seed=7,
)

SMOKE = dict(
    res=8, res_a=8, dates=4, n_train_seq=12, n_test_seq=8,
    n_voxel_train=200, n_voxel_test=400, epochs_a=1, epochs_b=1,
    batch_a=32, batch_b=4, layers=3, noise=0.004, seed=7,
)

RESULTS_JSON = "experiment_4d_results.json"
DATASET_CACHE = "dataset_4d_cache.npz"


def flush(results):
    """Checkpoint: rewrite the results JSON. Called after every stage so a
    Colab disconnect loses at most the stage in progress."""
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)


# ============================================================ data: meshes

def _download(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def fetch_modelnet(root="modelnet10"):
    """Download and unpack ModelNet10. Returns the extracted dir or None."""
    if os.path.isdir(os.path.join(root, "ModelNet10")):
        return os.path.join(root, "ModelNet10")
    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(root, "ModelNet10.zip")
    for url in MODELNET_URLS:
        try:
            print(f"  downloading {url} ...")
            _download(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(root)
            out = os.path.join(root, "ModelNet10")
            if os.path.isdir(out):
                return out
        except Exception as e:
            print(f"  download failed ({type(e).__name__}: {str(e)[:80]}) — trying next")
    return None


def read_off(path):
    """Tolerant OFF reader. Handles ModelNet's 'OFF490 518 0' header quirk."""
    with open(path, "r", errors="ignore") as f:
        tokens = f.read().split()
    if not tokens:
        raise ValueError("empty file")
    if tokens[0] == "OFF":
        tokens = tokens[1:]
    elif tokens[0].startswith("OFF"):
        tokens[0] = tokens[0][3:]          # counts glued to the magic word
    else:
        raise ValueError("not an OFF file")
    nv, nf = int(tokens[0]), int(tokens[1])
    tokens = tokens[3:]
    verts = np.array(tokens[: 3 * nv], dtype=np.float64).reshape(nv, 3)
    tokens = tokens[3 * nv:]
    tris = []
    i = 0
    for _ in range(nf):
        k = int(tokens[i])
        idx = [int(t) for t in tokens[i + 1: i + 1 + k]]
        for j in range(1, k - 1):          # fan-triangulate polygons
            tris.append((idx[0], idx[j], idx[j + 1]))
        i += 1 + k
    return verts, np.array(tris, dtype=np.int64)


def voxelize_mesh(verts, tris, res, n_points=60000, rng=None):
    """Surface-sample the mesh and bin points into a res^3 occupancy grid."""
    rng = rng or np.random.default_rng(0)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    if areas.sum() <= 0:
        raise ValueError("degenerate mesh")
    pick = rng.choice(len(tris), size=n_points, p=areas / areas.sum())
    r1, r2 = rng.random((2, n_points))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * a[pick] + (s * (1 - r2))[:, None] * b[pick] \
        + (s * r2)[:, None] * c[pick]
    lo, hi = pts.min(0), pts.max(0)
    span = (hi - lo).max() or 1.0
    ijk = ((pts - lo) / span * (res - 2)).astype(int) + 1   # 1-voxel margin
    ijk = np.clip(ijk, 0, res - 1)
    grid = np.zeros((res, res, res), dtype=np.float32)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1.0
    return grid


def procedural_structure(res, rng):
    """Fallback 'building': union of 2-5 cuboids on a ground plane."""
    grid = np.zeros((res, res, res), dtype=np.float32)
    for _ in range(rng.integers(2, 6)):
        sx, sy, sz = rng.integers(res // 6, res // 2, size=3)
        x, y = rng.integers(0, res - sx), rng.integers(0, res - sy)
        grid[x:x + sx, y:y + sy, 0:sz] = 1.0
    return grid


# ============================================================ data: 4D time

def _draw_edit(base, rng, min_delta, tries=50):
    """Draw a structural edit that actually changes >= min_delta voxels.

    Demolition centres the cuboid on a randomly chosen OCCUPIED voxel —
    uniform placement makes demolition a no-op on hollow shells more than
    half the time (verified empirically), which would leave label=1
    sequences containing no change at all. Falls back to construction if
    demolition cannot find enough mass.
    Returns (edit_grid, kind, n_changed)."""
    res = base.shape[0]
    occ = np.argwhere(base > 0)
    for _ in range(tries):
        demolish = rng.random() < 0.5 and len(occ) > 0
        sx, sy, sz = rng.integers(max(2, res // 8), max(3, res // 3), size=3)
        if demolish:
            cx, cy, cz = occ[rng.integers(len(occ))]
            x = int(np.clip(cx - sx // 2, 0, res - sx))
            y = int(np.clip(cy - sy // 2, 0, res - sy))
            z = int(np.clip(cz - sz // 2, 0, res - sz))
        else:
            x = rng.integers(0, res - sx)
            y = rng.integers(0, res - sy)
            z = rng.integers(0, max(1, res - sz))
        edit = np.zeros_like(base)
        edit[x:x + sx, y:y + sy, z:z + sz] = 1.0
        delta = int(((base > 0) & (edit > 0)).sum()) if demolish \
            else int(((base == 0) & (edit > 0)).sum())
        if delta >= min_delta:
            return edit, ("demolition" if demolish else "construction"), delta
    # Guaranteed fallback: a construction cuboid in the largest empty corner.
    edit = np.zeros_like(base)
    s = max(2, res // 4)
    edit[:s, :s, :s] = 1.0
    delta = int(((base == 0) & (edit > 0)).sum())
    return edit, "construction", delta


def evolve_clean(base, dates, change, rng, min_delta):
    """Noise-free temporal evolution. Returns (frames, when, kind, delta).

    The change mask is NOT tracked incrementally — downstream it is computed
    directly as clean_first != clean_last at whatever resolution an arm
    consumes, so labels are by construction a function of the observables."""
    frames, when, kind, delta = [], None, None, 0
    cur = base.copy()
    if change:
        when = int(rng.integers(1, dates))
        edit, kind, delta = _draw_edit(base, rng, min_delta)
    for t in range(dates):
        if change and t == when:
            cur = cur * (1 - edit) if kind == "demolition" \
                else np.maximum(cur, edit)
        frames.append(cur.copy())
    return np.stack(frames), when, kind, delta


def apply_noise(frames, noise, rng):
    out = frames.copy()
    flips = rng.random(out.shape) < noise
    out[flips] = 1.0 - out[flips]
    return out


def pool(frames, res_out):
    """Block-max downsample [dates, r, r, r] -> [dates, res_out^3]."""
    f = frames.shape[1] // res_out
    if f == 1:
        return frames.copy()
    return frames.reshape(frames.shape[0], res_out, f, res_out, f, res_out, f)\
                 .max(axis=(2, 4, 6))


def build_dataset(cfg, rng):
    """Assemble base grids (ModelNet if possible), evolve them in time, and
    derive each arm's view at its own resolution.

    Noise is applied independently per resolution — max-pooling noisy fine
    frames would light up an empty coarse block whenever ANY of its fine
    voxels flips (a one-sided 56x noise amplification, verified), so Arm A
    gets clean frames pooled first, then coarse-resolution noise."""
    n_total = cfg["n_train_seq"] + cfg["n_test_seq"]
    bases = []
    mn = fetch_modelnet() if not cfg.get("no_download") else None
    if mn:
        offs = []
        for cls in sorted(os.listdir(mn)):
            d = os.path.join(mn, cls, "train")
            if os.path.isdir(d):
                offs += [os.path.join(d, f) for f in sorted(os.listdir(d))
                         if f.endswith(".off")]
        rng.shuffle(offs)
        for path in offs:
            if len(bases) >= n_total:
                break
            try:
                v, t = read_off(path)
                bases.append(voxelize_mesh(v, t, cfg["res"], rng=rng))
            except Exception:
                continue
    n_real = len(bases)
    while len(bases) < n_total:
        bases.append(procedural_structure(cfg["res"], rng))
    provenance = (f"ModelNet10 ({n_real} meshes @ {cfg['res']}^3)"
                  if n_real >= n_total else
                  f"mixed ({n_real} ModelNet10 + {n_total - n_real} procedural)"
                  if n_real else "procedural")

    min_delta = max(8, cfg["res"] ** 3 // 8192)   # 32 fine voxels at res=64
    data = []
    for i, base in enumerate(bases):
        change = i % 2 == 0                       # balanced by construction
        clean, when, kind, delta = evolve_clean(
            base, cfg["dates"], change, rng, min_delta)
        clean_a = pool(clean, cfg["res_a"])
        item = dict(
            seq=apply_noise(clean, cfg["noise"], rng),          # Arm B view
            label=int(change), when=when, kind=kind, delta=delta,
            a_first=apply_noise(clean_a[:1], cfg["noise"], rng)[0],
            a_last=apply_noise(clean_a[-1:], cfg["noise"], rng)[0],
            a_mask=(clean_a[0] != clean_a[-1]),   # label from observables
        )
        data.append(item)
    rng.shuffle(data)
    return data[: cfg["n_train_seq"]], data[cfg["n_train_seq"]:], provenance


# ============================================================ quantum models

def make_arm_a_model(n_layers, quantum_device, torch):
    """Paper-faithful 8-wire model: Linear(8,8) -> AngleEmbedding ->
    SimplifiedTwoDesign -> <Z0> -> sigmoid. Initial RY layer is TRAINABLE
    (the repo passes zeros as a constant — dead capacity, fixed here)."""
    import pennylane as qml
    n_wires = 8

    @qml.qnode(quantum_device, diff_method="adjoint")
    def circuit(inputs, initial, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        qml.SimplifiedTwoDesign(initial_layer_weights=initial, weights=weights,
                                wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    shapes = {"initial": (n_wires,),
              "weights": qml.SimplifiedTwoDesign.shape(n_layers, n_wires)[1]}
    qlayer = qml.qnn.TorchLayer(circuit, shapes)

    class ArmA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pre = torch.nn.Linear(n_wires, n_wires)
            self.q = qlayer

        def forward(self, x):
            return torch.sigmoid(self.q(self.pre(x)))

    return ArmA(), n_wires


def make_arm_b_model(n_features, n_layers, quantum_device, torch):
    """Whole-volume model: AmplitudeEmbedding(dates x res^3) ->
    SimplifiedTwoDesign -> <Z0> -> scale/bias -> sigmoid.

    Note recorded in the JSON: with date-major flattening, wire 0 is the
    most significant address bit, so before any trained layers <Z0> equals
    the normalized early-vs-late-dates mass difference. The count-only
    baseline below is the control for that shortcut."""
    import pennylane as qml
    n_wires = max(1, math.ceil(math.log2(n_features)))

    @qml.qnode(quantum_device, diff_method="adjoint")
    def circuit(inputs, initial, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_wires),
                               normalize=True, pad_with=0.0)
        qml.SimplifiedTwoDesign(initial_layer_weights=initial, weights=weights,
                                wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    shapes = {"initial": (n_wires,),
              "weights": qml.SimplifiedTwoDesign.shape(n_layers, n_wires)[1]}
    qlayer = qml.qnn.TorchLayer(circuit, shapes)

    class ArmB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = qlayer
            # <Z> lives in [-1,1]; sigmoid alone maps that to [0.27, 0.73].
            # A trainable affine readout removes that ceiling. (The paper
            # applies sigmoid directly; noted as a deliberate deviation.)
            self.scale = torch.nn.Parameter(torch.tensor(3.0))
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            return torch.sigmoid(self.scale * self.q(x) + self.bias)

    return ArmB(), n_wires


# ============================================================ arm A pipeline

def _neighbourhoods(item, coords):
    """3^3 neighbourhoods at first and last date for given coarse coords."""
    first = np.pad(item["a_first"], 1)
    last = np.pad(item["a_last"], 1)
    feats = []
    for x, y, z in coords:
        nb1 = first[x:x + 3, y:y + 3, z:z + 3].ravel()
        nb2 = last[x:x + 3, y:y + 3, z:z + 3].ravel()
        feats.append(np.concatenate([nb1, nb2]))
    return feats


def arm_a_train_samples(seqs, n_target, rng):
    """TRAINING set: stratified pos/neg quota per sequence (a deliberate,
    documented training-side choice — evaluation never uses this path)."""
    feats, labels = [], []
    n_pos = max(1, n_target // (len(seqs) * 2))
    for item in seqs:
        mask = item["a_mask"]
        pos = np.argwhere(mask)
        neg = np.argwhere(~mask)
        take = []
        if len(pos):
            take += [(tuple(p), 1) for p in
                     pos[rng.choice(len(pos), min(n_pos, len(pos)), replace=False)]]
        take += [(tuple(p), 0) for p in
                 neg[rng.choice(len(neg), n_pos, replace=False)]]
        feats += _neighbourhoods(item, [c for c, _ in take])
        labels += [l for _, l in take]
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)


def arm_a_test_samples(seqs, n_target, rng):
    """TEST set: uniform random voxels — natural prevalence, no
    stratification. The 2D reproduction showed a class-balanced evaluation
    resample inflated F1 from 0.153 to 0.691; this experiment does not
    repeat that mistake."""
    feats, labels = [], []
    n_per = max(1, n_target // len(seqs))
    for item in seqs:
        res_a = item["a_mask"].shape[0]
        coords = rng.integers(0, res_a, size=(n_per, 3))
        feats += _neighbourhoods(item, [tuple(c) for c in coords])
        labels += [int(item["a_mask"][tuple(c)]) for c in coords]
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.float32)


def fit_pca_pair(Xtr, Xte):
    """ONE PCA basis, fit on train only, shared across both dates.
    (The OSCD repo refits per date — unchanged voxels get different features
    at t1 and t2 — and refits on evaluation data; both fixed here.)"""
    from sklearn.decomposition import PCA
    half = Xtr.shape[1] // 2
    pca = PCA(n_components=4).fit(
        np.concatenate([Xtr[:, :half], Xtr[:, half:]], axis=0))
    def project(X):
        return np.concatenate(
            [pca.transform(X[:, :half]), pca.transform(X[:, half:])],
            axis=1).astype(np.float32)
    return project(Xtr), project(Xte)


# ============================================================ training loops

def train_binary(model, Xtr, ytr, Xte, yte, epochs, batch, lr, opt_name, torch,
                 balance=False):
    """Shared train/eval loop.

    balance=True reproduces the paper's WeightedRandomSampler on the
    TRAINING side only: indices drawn with replacement, probability inverse
    to class frequency. Test metrics are computed on (Xte, yte) exactly as
    given — the caller controls what distribution that is."""
    opt = (torch.optim.RMSprop if opt_name == "rmsprop" else torch.optim.Adam)(
        model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte)
    hist = []
    n = len(Xtr_t)
    if balance:
        pos = float(ytr.sum())
        w = torch.tensor(np.where(ytr > 0.5, 1.0 / max(pos, 1.0),
                                  1.0 / max(n - pos, 1.0)), dtype=torch.double)
    for ep in range(epochs):
        t0 = time.perf_counter()
        perm = torch.multinomial(w, n, replacement=True) if balance \
            else torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(idx)
        dt = time.perf_counter() - t0
        hist.append(dict(epoch=ep + 1, loss=ep_loss / n, seconds=dt,
                         samples_per_s=n / dt))
        print(f"    epoch {ep+1}/{epochs}  loss {ep_loss/n:.4f}  "
              f"{dt:.1f}s  ({n/dt:.1f} samples/s)")
    with torch.no_grad():
        pv = model(Xte_t).numpy().ravel()
    return dict(history=hist, **metrics(pv, yte))


def metrics(probs, y):
    pred = (probs >= 0.5).astype(int)
    yv = y.astype(int)
    tp = int(((pred == 1) & (yv == 1)).sum()); fp = int(((pred == 1) & (yv == 0)).sum())
    fn = int(((pred == 0) & (yv == 1)).sum()); tn = int(((pred == 0) & (yv == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(accuracy=(tp + tn) / max(1, len(yv)), precision=prec,
                recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
                prevalence=float(yv.mean()), probs_kept=False)


def balanced_view(probs, y, rng, draws=20):
    """Metrics on class-balanced resamples of the SAME predictions — the
    paper's evaluation protocol, reported for comparison, never as the
    headline."""
    yv = y.astype(int)
    pos = np.where(yv == 1)[0]
    neg = np.where(yv == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    k = min(len(pos), len(neg))
    f1s, accs = [], []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(pos, k, replace=False),
                              rng.choice(neg, k, replace=False)])
        m = metrics(probs[idx], yv[idx])
        f1s.append(m["f1"]); accs.append(m["accuracy"])
    return dict(f1_mean=float(np.mean(f1s)), accuracy_mean=float(np.mean(accs)),
                draws=draws, note="paper-style balanced resample of the same "
                "predictions; inflated relative to natural metrics by design")


def count_baseline(train_seq, test_seq):
    """Logistic regression on per-date occupancy totals ONLY. Arm B must
    beat this to claim it detects spatial change rather than counting."""
    from sklearn.linear_model import LogisticRegression
    def feats(seqs):
        return np.array([[s["seq"][t].mean() for t in range(s["seq"].shape[0])]
                         for s in seqs])
    Xtr, ytr = feats(train_seq), np.array([s["label"] for s in train_seq])
    Xte, yte = feats(test_seq), np.array([s["label"] for s in test_seq])
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return metrics(clf.predict_proba(Xte)[:, 1], yte)


# ============================================================ micro-benchmark

def micro_benchmark(widths, torch):
    """Forward + adjoint gradient per backend at the experiment's widths.
    The initial layer is created with requires_grad=False so gradient
    timings cover exactly the (3, n-1, 2) trainable weights."""
    import pennylane as qml
    from pennylane import numpy as pnp
    out = []
    backends = []
    for name in ["default.qubit", "lightning.qubit", "lightning.gpu"]:
        try:
            qml.device(name, wires=2)
            backends.append(name)
        except Exception:
            pass
    for n in widths:
        for b in backends:
            try:
                dev = qml.device(b, wires=n)
                shape = qml.SimplifiedTwoDesign.shape(3, n)
                init = pnp.zeros(n, requires_grad=False)

                @qml.qnode(dev, diff_method="adjoint")
                def c(w):
                    qml.SimplifiedTwoDesign(init, w, wires=range(n))
                    return qml.expval(qml.PauliZ(0))

                w = pnp.array(np.random.uniform(0, np.pi, shape[1]),
                              requires_grad=True)
                c(w)                                   # warm up
                tf = min(_t(c, w) for _ in range(3))
                g = qml.grad(c)
                g(w)
                tg = min(_t(g, w) for _ in range(3))
                out.append(dict(wires=n, backend=b, forward_ms=1e3 * tf,
                                gradient_ms=1e3 * tg))
                print(f"    {n:>2} wires  {b:<16} fwd {1e3*tf:9.2f} ms   "
                      f"grad {1e3*tg:9.2f} ms")
            except Exception as e:
                out.append(dict(wires=n, backend=b, error=type(e).__name__))
                print(f"    {n:>2} wires  {b:<16} {type(e).__name__}")
    return out


def _t(fn, *a):
    t0 = time.perf_counter()
    fn(*a)
    return time.perf_counter() - t0


# ============================================================ dataset cache

def cache_key(cfg):
    return "_".join(str(cfg[k]) for k in
                    ("res", "res_a", "dates", "n_train_seq", "n_test_seq",
                     "noise", "seed"))


def save_cache(train_seq, test_seq, provenance, cfg):
    def pack(seqs):
        return (np.stack([s["seq"] for s in seqs]).astype(np.uint8),
                np.stack([s["a_first"] for s in seqs]).astype(np.uint8),
                np.stack([s["a_last"] for s in seqs]).astype(np.uint8),
                np.stack([s["a_mask"] for s in seqs]),
                np.array([s["label"] for s in seqs]))
    tr, te = pack(train_seq), pack(test_seq)
    np.savez_compressed(DATASET_CACHE, key=cache_key(cfg), provenance=provenance,
                        **{f"tr{i}": a for i, a in enumerate(tr)},
                        **{f"te{i}": a for i, a in enumerate(te)})


def load_cache(cfg):
    if not os.path.exists(DATASET_CACHE):
        return None
    z = np.load(DATASET_CACHE, allow_pickle=True)
    if str(z["key"]) != cache_key(cfg):
        return None
    def unpack(p):
        seqs, a1, a2, m, lab = (z[f"{p}{i}"] for i in range(5))
        return [dict(seq=seqs[i].astype(np.float32),
                     a_first=a1[i].astype(np.float32),
                     a_last=a2[i].astype(np.float32),
                     a_mask=m[i], label=int(lab[i])) for i in range(len(lab))]
    return unpack("tr"), unpack("te"), str(z["provenance"])


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny local CPU run")
    ap.add_argument("--no-download", action="store_true",
                    help="skip ModelNet, use procedural structures")
    ap.add_argument("--arm", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    cfg = dict(SMOKE if args.smoke else DEFAULTS)
    cfg["no_download"] = args.no_download or args.smoke
    rng = np.random.default_rng(cfg["seed"])

    import torch
    import pennylane as qml
    torch.manual_seed(cfg["seed"])

    gpu = False
    try:
        qml.device("lightning.gpu", wires=2)
        gpu = True
    except Exception:
        pass
    qdev_name = "lightning.gpu" if gpu else "lightning.qubit"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        smi = None
    print(f"platform  : {platform.platform()}")
    print(f"pennylane : {qml.__version__}  |  torch {torch.__version__}")
    print(f"quantum   : {qdev_name}  (gpu={'yes' if gpu else 'NO — CPU fallback'})"
          + (f"  [{smi}]" if smi else ""))

    results = dict(
        config={k: v for k, v in cfg.items()},
        env=dict(platform=platform.platform(), pennylane=qml.__version__,
                 torch=torch.__version__, quantum_device=qdev_name, gpu=gpu,
                 nvidia_smi=smi),
        caveats=[
            "Temporal evolution is synthetic (structural edits on real 3D "
            "models); no public 4D change benchmark with voxel labels exists.",
            "Amplitude state preparation is O(2^n) CNOTs on hardware "
            "(2^n - n - 1), so Arm B is simulable, not executable.",
            "Arm B's classes differ in total occupancy, and with date-major "
            "flattening <Z0> reads the early-vs-late mass statistic; Arm B "
            "evidences spatial change detection only where it beats "
            "count_baseline.",
        ],
    )
    flush(results)

    print("\n[1/4] building 4D dataset ...")
    t0 = time.perf_counter()
    cached = load_cache(cfg)
    if cached:
        train_seq, test_seq, provenance = cached
        print("  loaded from cache")
    else:
        train_seq, test_seq, provenance = build_dataset(cfg, rng)
        try:
            save_cache(train_seq, test_seq, provenance, cfg)
        except Exception as e:
            print(f"  (cache write skipped: {type(e).__name__})")
    print(f"  {len(train_seq)} train / {len(test_seq)} test sequences "
          f"({provenance})  in {time.perf_counter()-t0:.1f}s")
    results["data"] = dict(
        provenance=provenance, dates=cfg["dates"], res=cfg["res"],
        n_train=len(train_seq), n_test=len(test_seq),
        edit_kinds={k: sum(1 for s in train_seq + test_seq
                           if s.get("kind") == k)
                    for k in ("demolition", "construction")},
        min_edit_voxels=max(8, cfg["res"] ** 3 // 8192))
    flush(results)

    if args.arm in ("a", "both"):
        print(f"\n[2/4] Arm A — paper-faithful per-voxel QNN, 8 wires "
              f"({qdev_name})")
        Xtr_raw, ytr = arm_a_train_samples(train_seq, cfg["n_voxel_train"], rng)
        Xte_raw, yte = arm_a_test_samples(test_seq, cfg["n_voxel_test"], rng)
        Xtr, Xte = fit_pca_pair(Xtr_raw, Xte_raw)
        print(f"  train {len(Xtr)} voxels ({100*ytr.mean():.0f}% changed, "
              f"stratified)  |  test {len(Xte)} voxels "
              f"({100*yte.mean():.1f}% changed, natural)")
        dev = qml.device(qdev_name, wires=8)
        model, _ = make_arm_a_model(cfg["layers"], dev, torch)
        arm_a = train_binary(model, Xtr, ytr, Xte, yte, cfg["epochs_a"],
                             cfg["batch_a"], 1e-3, "rmsprop", torch,
                             balance=True)
        with torch.no_grad():
            probs = model(torch.tensor(Xte)).numpy().ravel()
        arm_a["balanced_resample"] = balanced_view(probs, yte, rng)
        arm_a.update(wires=8, device=qdev_name, embedding="angle (paper)",
                     n_params=sum(p.numel() for p in model.parameters()))
        results["arm_a"] = arm_a
        flush(results)
        print(f"  natural: acc {arm_a['accuracy']:.3f}  P {arm_a['precision']:.3f}  "
              f"R {arm_a['recall']:.3f}  F1 {arm_a['f1']:.3f}")
        if arm_a["balanced_resample"]:
            print(f"  balanced resample (paper protocol): "
                  f"F1 {arm_a['balanced_resample']['f1_mean']:.3f} — "
                  f"inflated, for comparison only")

    if args.arm in ("b", "both"):
        n_feat = cfg["dates"] * cfg["res"] ** 3
        n_wires = max(1, math.ceil(math.log2(n_feat)))
        print(f"\n[3/4] Arm B — whole-volume amplitude QNN, {n_wires} wires "
              f"({qdev_name})")
        Xtr = np.stack([s["seq"].ravel() for s in train_seq]).astype(np.float32)
        ytr = np.array([s["label"] for s in train_seq], dtype=np.float32)
        Xte = np.stack([s["seq"].ravel() for s in test_seq]).astype(np.float32)
        yte = np.array([s["label"] for s in test_seq], dtype=np.float32)
        dev = qml.device(qdev_name, wires=n_wires)
        model, _ = make_arm_b_model(n_feat, cfg["layers"], dev, torch)
        arm_b = train_binary(model, Xtr, ytr, Xte, yte, cfg["epochs_b"],
                             cfg["batch_b"], 5e-2, "adam", torch)
        arm_b["count_baseline"] = count_baseline(train_seq, test_seq)
        arm_b.update(wires=n_wires, device=qdev_name, embedding="amplitude",
                     stateprep_cnots_hw=2 ** n_wires - n_wires - 1,
                     n_params=sum(p.numel() for p in model.parameters()))
        results["arm_b"] = arm_b
        flush(results)
        print(f"  QNN: acc {arm_b['accuracy']:.3f}  F1 {arm_b['f1']:.3f}   |   "
              f"count-only baseline: acc {arm_b['count_baseline']['accuracy']:.3f}"
              f"  F1 {arm_b['count_baseline']['f1']:.3f}")
        print(f"  (state prep on hardware: {2**n_wires - n_wires - 1:,} CNOTs)")

    print("\n[4/4] micro-benchmark at the experiment's widths ...")
    widths = [8]
    if args.arm in ("b", "both"):
        widths.append(max(1, math.ceil(math.log2(cfg["dates"] * cfg["res"] ** 3))))
    results["benchmark"] = micro_benchmark(widths, torch)
    flush(results)
    print(f"\nwrote {RESULTS_JSON} — download this and hand it back")


if __name__ == "__main__":
    main()
