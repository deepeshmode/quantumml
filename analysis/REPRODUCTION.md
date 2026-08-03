# Reproduction notes — Tomev/qnn_change_detection

Reproducing the pixel-level QNN change-detection result of Rybotycki, Gupta &
Gawron (arXiv:2503.08962v3; Zenodo 14784888) on the OSCD dataset.

Date: 2026-07-24. Machine: macOS, CPU only.

## Result

**Reproduced.** 20 epochs, pixel-level, 4 qubits.

> **Update 2026-07-31 — read the headline below with this caveat.** The
> validation numbers here are computed on a **class-balanced resample**, not
> on the images as they are. `prepare_data_loader` wraps the evaluation set
> in the same inverse-class-frequency `WeightedRandomSampler` as training
> (`src/pixel_level/utils.py`), so the eval sample is ~49% positive (the
> confusion matrix below sums to 25,500 with 12,463 positives) while the
> real test cities are **2.56%** positive. Running the same trained model
> over every pixel of the three held-out cities gives **accuracy 0.821,
> F1 0.153** (precision 0.087, recall 0.629). Both numbers are real; they
> answer different questions — but only the full-resolution one is
> comparable to Daudt's baseline or the OSCD literature, so the paper's
> "~15 points worse than classical" claim cannot be checked against the
> 0.691 below. Full analysis, figures and per-city numbers:
> github.com/deepeshmode/quantumml — `ANSWERS.md` and `analysis/`.
>
> Also: held-out accuracy peaked at **epoch 18 (73.60%)** and decayed to
> 72.74% by epoch 20; only the final model was saved. `src/pixel_level/
> train.py` now carries a local patch that checkpoints the best-val model
> (`best_model.pth`) as it happens.

| epoch | this run (val acc) | Zenodo (val acc) |
|-------|--------------------|------------------|
| untrained | 49.4% | — |
| 1 | 63.2% | 63.17% |
| 5 | 63.4% | 71.35% |
| 10 | 64.7% | 71.19% |
| 17 | 73.5% | — |
| 20 (final) | **72.8%** | ~71% |

Final validation: **accuracy 72.8%, F1 0.691**. Confusion matrix:

```
                 pred no-change   pred change
actual no-change     10,792           2,245
actual change         4,700           7,763
```

The model learned change rather than collapsing to the majority class. Epoch 1
matches the published curve to three decimals (63.2% vs 63.17%). The trajectory
then diverges — this run stays near 63% until ~epoch 15, then climbs to the same
~71–73% plateau by epoch 17, where the published run reaches it by epoch 5. Same
start, same destination, slower path. See "Deviations" for the likely cause.

## Environment

Python 3.11. The current PyPI resolutions do not work; the code needs the
authors' pinned versions from Zenodo `pip_freeze.txt`. Key pins:

```
PennyLane==0.38.1         # NOTE: now YANKED on PyPI
PennyLane-qiskit==0.39.0
autoray==0.6.12           # 0.8.x breaks PennyLane 0.38 import
qiskit==1.1.0
qiskit-aer==0.14.2
qiskit-iqm==13.13         # >=18 refuses to import ("package is obsolete")
qiskit-machine-learning==0.8.2
numpy==1.26.4  scipy==1.13.1  scikit-learn==1.5.1  scikit-image==0.24.0
niapy==2.3.1  mlflow==2.17.0  torcheval==0.0.7
```

### Five breakages before a single epoch ran

This is first-hand evidence for the paper's own thesis about QML software
fragility. In order encountered:

1. `qiskit-iqm==18.2` raises `RuntimeError: package is obsolete` on import.
   Fixed by `qiskit-iqm<18`.
2. `qml.operation.AnyWires` removed from current PennyLane. Fixed by pinning
   PennyLane 0.38.1.
3. `autoray==0.8.4` breaks PennyLane 0.38: `module 'autoray.autoray' has no
   attribute 'NumpyMimic'`. Fixed by `autoray==0.6.12`.
4. MLflow tracking URI hardcoded to `http://127.0.0.1:8080` in `train.py`;
   without a server the run retries and stalls. Fixed by running a local
   `mlflow server`.
5. `PennyLane==0.38.1` is yanked on PyPI, so the authors' published environment
   is no longer cleanly installable from scratch — a reproducibility finding in
   its own right.

## How to run

```bash
uv venv --python 3.11 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt
VIRTUAL_ENV=.venv uv pip install "qiskit-iqm<18" \
  "PennyLane==0.38.1" "PennyLane-qiskit==0.39.0" "autoray==0.6.12" \
  "qiskit==1.1.0" "qiskit-aer==0.14.2" "qiskit-iqm==13.13" \
  "qiskit-machine-learning==0.8.2" "numpy==1.26.4" "scipy==1.13.1" \
  "scikit-learn==1.5.1" "scikit-image==0.24.0" "niapy==2.3.1" \
  "mlflow==2.17.0" "torcheval==0.0.7"

# tracking server (separate shell); URI is hardcoded to 8080
.venv/bin/mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns --host 127.0.0.1 --port 8080

# dataset path comes from an env var, not config.py
SIPWQNN_DATA_PATH=/path/to/oscd_root KMP_DUPLICATE_LIB_OK=TRUE \
  .venv/bin/python -m src.pixel_level.train
```

Config defaults (`src/pixel_level/config.py`): `PATCH_SIDE 1`, `TYPE BELOW20M`
— which loads **10 Sentinel-2 bands** (B02-B04, B08 at 10 m; B05-B07, B8A,
B11, B12 upsampled from 20 m; see `read_sentinel_img_below20`), not the 5 the
config's `NUM_BANDS = TYPE + 3` formula suggests. That enum arithmetic is
correct for RGB (0→3) and RGBIR (1→4) only; it is off for BELOW20M (actual
10) and ALLBANDS (actual 13). Harmless in this pixel-level path (any value
>4 routes into PCA), but live in `src/patch/`, where `NUM_BANDS` is passed
to the model constructor. `PCA_COMPONENTS 4` → **4 qubits**, `N_EPOCHS 20`,
`BATCH_SIZE 500`, `DEVICE_TYPE DEFAULT` (`default.qubit`, noiseless).

### Dataset layout the loader expects

`oscd_dataloader.py` reads `<Labels>/../Images/<city>/imgs_{1,2}/*.tif` and
`<Labels>/<city>/cm/cm.png`, so Images and Labels must be real sibling
directories (a symlinked Labels dir breaks the `..` traversal):

```
oscd_root/
  Onera Satellite Change Detection dataset - Images/
    train.txt, test.txt, <city>/imgs_1/*.tif, <city>/imgs_2/*.tif
  Onera Satellite Change Detection dataset - Train Labels/<city>/cm/cm.png
  Onera Satellite Change Detection dataset - Test Labels/<city>/cm/cm.png
```

## Deviations from the paper

- **Split.** OSCD ships test-set labels separately and they were not in the
  local archive. The paper trains on 14 labelled cities and tests on 10; only 14
  labelled cities were available here, so the split is 11 train / 3 test
  (`cupertino, beirut, mumbai` held out). Fewer training cities and a smaller,
  differently-composed validation set is the most likely reason the learning
  curve is slower than the published one even though the plateau matches.
- **CPU only**, so device-oriented (noisy) training was not attempted — that is
  where the paper itself hit out-of-memory and could not complete an epoch.
- Wall-clock 3h11m, dominated by machine throttling (≈39 s/epoch early, rising
  to ≈1764 s/epoch), not computation.

## Finding: the PennyLane/Qiskit discrepancy is a circuit bug

The repo README lists an unresolved issue:

> "While pennylane and qiskit implementations of our model are meant to do the
> same computations, they do not. Their outputs are different. We were not able
> to assert why."

`compare_std.py` (in this directory) builds `pennylane.SimplifiedTwoDesign` and
the repo's `SimplifiedTwoDesignQiskit` with the same weights and diffs them gate
by gate. They are **different circuits** — not a simulator or endianness issue:

1. **Missing initial rotation layer.** PennyLane's `SimplifiedTwoDesign` opens
   with an `RY(initial_layer_weights)` on every wire. The Qiskit
   reimplementation omits it entirely (4 gates gone for N=4).
2. **Wrong even-block RY targets.** PennyLane applies RY to both qubits of each
   even pair (wires 0,1,2,3). The reimplementation loops `range(n_qubits - 1)`,
   so wire 3 receives no even-block RY and the RY/CZ interleaving differs.

Net: 18 gates vs 22 for N=4, L=2. Either defect alone makes the two
implementations compute different functions; together they fully account for the
documented discrepancy. On genuinely identical circuits the two frameworks agree
analytically to ~5e-16 (see the quantumml repo's `sim_comparison.py`).

## Next

- **Hyperspectral.** Preprocess hyperspectral imagery, select informative bands,
  build a multispectral stack, and feed this pipeline — the flagged follow-on to
  reproduction.
- **FP_MODIFIER fairness test.** This config uses `FP_MODIFIER 1`; Daudt's
  classical baseline uses 10. Rerunning the QNN with 10 tests how much of the
  paper's "~15 points worse than classical" gap is class-imbalance handling
  rather than the quantum model.
- **Matched classical baseline.** `fully_convolutional_change_detection-master`
  holds Daudt's official Siam-U-Net (reference [12] in the paper). Running it on
  this same split would test the paper's 15-point claim directly.
