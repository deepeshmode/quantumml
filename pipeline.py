"""
Hyperspectral -> band selection -> multispectral -> QNN.

Implements the preprocessing pipeline set out in the project brief: reduce a hyperspectral
cube to a small set of informative bands, producing a multispectral stack that
the QNN change-detection code can consume.

Proxy task
----------
No hyperspectral scene over a hyperscale data center is publicly available.
Pavia University (ROSIS, 103 bands, 430-860 nm) is used instead because its
material classes are the ones a data center campus is built from: painted metal
sheets (roofing), asphalt and gravel (hardstanding), bitumen (roofing), bare
soil (cleared ground). The binary task below - built/disturbed surface vs.
undisturbed vegetation - is the spectral signal that Krawec's construction
taxonomy tracks at stages 1 (site clearing) through 6 (cladding).
"""

import numpy as np
import scipy.io as sio

# Pavia University ground-truth classes (1-9); 0 is unlabeled.
CLASS_NAMES = {
    1: "Asphalt", 2: "Meadows", 3: "Gravel", 4: "Trees",
    5: "Painted metal sheets", 6: "Bare Soil", 7: "Bitumen",
    8: "Self-Blocking Bricks", 9: "Shadows",
}

# Materials a hyperscale data center campus is made of.
BUILT_CLASSES = [1, 3, 5, 6, 7, 8]
# Undisturbed vegetation - what stage-1 site clearing removes.
VEG_CLASSES = [2, 4]

# ROSIS-3 spectral range for the Pavia scenes.
WL_MIN, WL_MAX = 430.0, 860.0


def wavelengths(n_bands=103):
    """Centre wavelength (nm) of each retained ROSIS band."""
    return np.linspace(WL_MIN, WL_MAX, n_bands)


def load_pavia(data_dir="data"):
    """Return (cube [H,W,B] float32 reflectance-scaled, labels [H,W] int)."""
    cube = sio.loadmat(f"{data_dir}/PaviaU.mat")["paviaU"].astype(np.float32)
    gt = sio.loadmat(f"{data_dir}/PaviaU_gt.mat")["paviaU_gt"].astype(int)
    cube /= cube.max()
    return cube, gt


def binary_task(cube, gt):
    """Flatten to (X, y) over labeled pixels only. y=1 built, y=0 vegetation."""
    mask = np.isin(gt, BUILT_CLASSES + VEG_CLASSES)
    X = cube[mask]
    y = np.isin(gt[mask], BUILT_CLASSES).astype(int)
    return X, y


def ndvi(cube, red_nm=670.0, nir_nm=850.0):
    """
    NDVI from the two nearest ROSIS bands.

    ROSIS tops out at 860 nm, so the NIR shoulder is only partly sampled. This
    is a weaker NDVI than Sentinel-2 (842 nm) would give; it separates
    vegetation from built surfaces but the absolute values are not comparable
    to standard NDVI products.
    """
    wl = wavelengths(cube.shape[-1])
    red = cube[..., int(np.argmin(np.abs(wl - red_nm)))]
    nir = cube[..., int(np.argmin(np.abs(wl - nir_nm)))]
    return (nir - red) / (nir + red + 1e-8)


def rgb_composite(cube):
    """True-colour composite, percentile-stretched for display."""
    wl = wavelengths(cube.shape[-1])
    idx = [int(np.argmin(np.abs(wl - c))) for c in (640.0, 550.0, 460.0)]
    img = cube[..., idx]
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    return np.clip((img - lo) / (hi - lo), 0, 1)


def select_bands(X, y, k, method="mi", seed=0):
    """
    Choose k bands from the full hyperspectral stack.

    mi      - mutual information with the class label (supervised)
    fscore  - ANOVA F statistic (supervised, cheap)
    uniform - evenly spaced across the spectrum; this is the baseline, and is
              roughly what a fixed multispectral sensor hands you for free
    pca     - bands with the largest loading on the leading components
    """
    from sklearn.feature_selection import mutual_info_classif, f_classif

    n_bands = X.shape[1]
    if method == "uniform":
        return np.linspace(0, n_bands - 1, k).round().astype(int)

    if method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(n_components=min(k, n_bands), random_state=seed).fit(X)
        loading = np.abs(p.components_).max(axis=0)
        return np.sort(np.argsort(loading)[::-1][:k])

    if method == "fscore":
        score, _ = f_classif(X, y)
    else:
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(X), size=min(4000, len(X)), replace=False)
        score = mutual_info_classif(X[sub], y[sub], random_state=seed)

    score = np.nan_to_num(score)
    return np.sort(np.argsort(score)[::-1][:k])


def select_bands_decorrelated(X, y, k, method="mi", max_corr=0.95, seed=0):
    """
    Greedy band selection that rejects redundant neighbours.

    Univariate scoring alone picks clusters of adjacent bands, which in a
    hyperspectral cube are near-duplicates - it spends the band budget on one
    part of the spectrum. This takes the highest-scoring band, then keeps
    adding the next-best band whose correlation with everything already chosen
    stays below max_corr.
    """
    from sklearn.feature_selection import mutual_info_classif, f_classif

    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(4000, len(X)), replace=False)
    if method == "fscore":
        score, _ = f_classif(X, y)
    else:
        score = mutual_info_classif(X[sub], y[sub], random_state=seed)
    score = np.nan_to_num(score)

    corr = np.abs(np.corrcoef(X[sub].T))
    order = np.argsort(score)[::-1]

    chosen = [int(order[0])]
    for b in order[1:]:
        if len(chosen) >= k:
            break
        if corr[b, chosen].max() < max_corr:
            chosen.append(int(b))
    # If the correlation floor was too strict to fill the budget, relax it.
    for b in order:
        if len(chosen) >= k:
            break
        if b not in chosen:
            chosen.append(int(b))
    return np.sort(np.array(chosen[:k]))


def split(X, y, n_train=3000, n_test=1500, seed=0):
    """Balanced train/test subsample, standardized on training statistics."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        tr.append(idx[: n_train // 2])
        te.append(idx[n_train // 2 : n_train // 2 + n_test // 2])
    tr = rng.permutation(np.concatenate(tr))
    te = rng.permutation(np.concatenate(te))

    Xtr, Xte = X[tr], X[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, y[tr], (Xte - mu) / sd, y[te]
