# Answers — QNN change detection on OSCD

Five questions, answered against the reproduction run `overjoyed-doe-477`
(20 epochs, finished 2026-07-24, 3 h 10 m 58 s) and a full-resolution re-analysis
of that model run on 2026-07-31.

Everything below is measured on the machine that did the work: **Apple M4,
10 cores, 24 GB, macOS**. Code under test is `Tomev/qnn_change_detection` at
`c049636`, pixel-level path, `BandType.BELOW20M`, `PCA_COMPONENTS = 4`.

Figures referenced live in [`analysis/`](analysis/); every number is in
[`analysis/metrics.json`](analysis/metrics.json).

---

## First, a correction that reframes the rest

**The headline 72.8 % / F1 0.691 is computed on a class-balanced resample, not on
the images as they actually are.**

`prepare_data_loader` wraps *both* the training and the evaluation set in a
`WeightedRandomSampler` (`src/pixel_level/utils.py:51-56`) whose weights are
inverse class frequency, `[1/true_pix, 1/(n_pix − true_pix)]`
(`src/oscd_dataloader.py:196`). The evaluation sample therefore comes out near
50/50. The logged confusion matrix confirms it: 12,463 positives out of 25,500
samples, 48.9 %.

In the real images, **2.56 % of pixels are change**.

Running the same trained model over every pixel of the three held-out cities:

| | balanced resample (reported) | full resolution (actual) |
|---|---|---|
| accuracy | 0.728 | **0.821** |
| F1 | 0.691 | **0.153** |
| precision | — | 0.087 |
| recall | — | 0.629 |

Accuracy *rises* and F1 *collapses*, which is the signature of a rare-positive
problem being scored on a balanced sample. Both numbers are real; they answer
different questions. The consequence that matters: **the balanced figure is not
comparable to Daudt's classical baseline or to most OSCD literature**, which
report on the natural distribution. The "~15 points worse" comparison in the
source paper cannot be checked against 0.691.

`analysis/fig2_change_maps.png` shows why. The false alarms are not scattered
noise — they trace the built environment itself. The model has largely learned
"this pixel is urban," not "this pixel changed."

---

## 1) How is memory usage?

**Peak 1.27 GB.** Comfortably inside 24 GB; memory is not a constraint at this
scale.

| Stage | Peak RSS |
|---|---|
| Baseline | 15 MB |
| After imports (torch + PennyLane) | 425 MB |
| + train set, 11 cities | 989 MB |
| + test set, 3 cities | **1,272 MB** |

Data actually retained is far smaller than the peak:

| | Train (11 cities) | Test (3 cities) |
|---|---|---|
| `imgs_1` + `imgs_2`, post-PCA | 121.4 MB | 77.6 MB |
| change maps | 3.8 MB | 2.4 MB |
| patch coordinates | 2.8 MB | 1.8 MB |
| **resident** | **128.0 MB** | **81.7 MB** |
| patches indexed | 40,322 | 25,500 |

The gap between the 564 MB jump and the 128 MB retained is **transient PCA
cost**. Each image is reshaped to `(H·W, 10)` and passed through
`StandardScaler.fit_transform` then `PCA.fit_transform`
(`src/oscd_dataloader.py:106-117`); scikit-learn promotes to float64 and holds
several full-image copies at once. Beirut alone is 1180 × 1070 × 10 at float64 ≈
101 MB per copy.

**The quantum simulation is negligible**: 8 wires = 256 amplitudes = 0.004 MB.
Statevector memory doubles per wire, so 24 GB tops out near 30 wires — roughly
22 beyond what this experiment uses.

Memory is dominated entirely by classical preprocessing, and all imagery is held
resident rather than streamed. Scaling to more cities calls for float32 PCA or
lazy loading, not anything quantum.

## 2) Does the code allow GPU acceleration? Nvidia or AMD?

**No GPU was used, and neither Nvidia nor AMD has been tested.** The hardware is
an Apple M4 — there is no CUDA device present:

```
cuda available : False      cuda built : False
mps  available : True       mps  built : True      (available, never used)
```

Three code-level facts:

- `TORCH_DEVICE` is hardcoded `"cpu"` (`src/pixel_level/config.py:131`); the run
  log confirms `Using cpu torch device`.
- The quantum device is `default.qubit`, PennyLane's CPU simulator.
  `lightning.qubit` — the C++ backend — is **commented out** with the note
  *"There's some kind of problem with the lightning.qubit device"*
  (`config.py:165-167`). `pennylane-lightning 0.38.0` is installed;
  `pennylane-lightning-gpu` is not.
- CUDA code exists but in a different entry point:
  `src/run_on_real_device.py:276` selects `cuda` when available. That path was
  not used.

One latent defect: `ConvolutionalQuantumNeuralNetwork` hardcodes
`self.torch_device = "cuda"` with no availability check
(`src/quantum_classifier.py:613`). That class cannot run on this machine as
written.

**Would a GPU help? Measured answer: not at this circuit width.** The run
performed **1,367,440 circuit executions across 2,742 batches**. At 8 wires each
state is 256 amplitudes — 4 KB — far too small to saturate a GPU.

This is not speculation; the backend crossover is measurable on CPU alone.
Per-circuit microseconds for this project's own circuit (AngleEmbedding +
SimplifiedTwoDesign, 3 layers), swept across width —
`analysis/fig6_backend_crossover.png`:

| wires | default.qubit | lightning.qubit | qiskit.aer |
|---|---|---|---|
| 4 | **17** | 362 | 9,513 |
| 8 | **186** | 792 | 18,960 |
| 12 | **1,627** | 2,551 | 37,741 |
| 14 | 8,398 | **6,367** | 50,337 |
| 16 | 33,035 | **18,777** | 64,400 |
| 18 | 140,154 | 84,889 | **88,534** |
| 20 | 678,818 | 430,791 | **175,398** |

Every accelerated backend carries a fixed per-call overhead and only wins once
the state is large enough to amortise it. `lightning.qubit` — a C++ backend — is
**4.2× slower than plain Python at 8 wires** and does not overtake until 14.
`qiskit.aer`, 50× slower at small widths, wins above 18. The upstream choice of
`default.qubit` is therefore correct for this workload, and the commented-out
`lightning.qubit` line in `config.py:165-167` costs nothing at 8 wires.

Two updates from later measurement (2026-08-03), which sharpen and partly
supersede the table above:

**For gradients — what training actually computes — the C++ backend wins at
every width.** The forward-pass crossover does not carry over to adjoint
differentiation: on this machine `lightning.qubit` beats `default.qubit` at
4 wires already (2.2 vs 9.3 ms/gradient) and by 44× at 20 wires (0.38 vs
16.5 s). The "default.qubit wins below 14 wires" rule applies to broadcast
forward passes only.

**The GPU crossover is now measured, not projected** — Colab Tesla T4,
PennyLane 0.45.1, same host for all backends, single-circuit timings
(`analysis/fig8_gpu_t4.png`, data in `analysis/experiment_4d_results.json`):

| | 8 wires fwd / grad | 20 wires fwd / grad |
|---|---|---|
| default.qubit | 65 / 75 ms | 11,635 / 12,873 ms |
| lightning.qubit | **8.6 / 10.5 ms** | 1,408 / 1,406 ms |
| lightning.gpu | 16.2 / 17.9 ms | **151 / 155 ms** |

At 8 wires — this project's model — the GPU is **1.7–1.9× slower** than the
same machine's CPU backend. At 20 wires it is **9.1–9.3× faster**. The same
inversion shows up in the real training loops, not just microbenchmarks: the
8-wire Arm A of the 4D experiment trained at 38 samples/s on the T4 versus
~400 samples/s for the identical loop on a laptop CPU, while the 20-wire
Arm B trained at 11.7 s/epoch on the GPU — a workload that projects to
roughly 9× longer on that host's CPU backend and ~80× longer on pure Python.

The exponential wall is not something a GPU moves. Circuit width for the
patch-based model is `2 × PATCH_SIDE²`:

| patch | wires | statevector |
|---|---|---|
| 2×2 | 8 | 4 KB |
| 3×3 | 18 | 4 MB |
| 4×4 | 32 | **68 GB** |

24 GB of RAM reaches ~30 wires; an 80 GB A100 reaches ~32. **A GPU buys roughly
two qubits**, not a new regime.

What does help, measured: **batch size and idle cores.** Inference runs at
~9,700 px/s at batch 1000 but collapses to ~650 px/s at batch 20,000 — a 15×
penalty from batch size alone. On the QCNN circuit, batch 1 → 1000 moves
throughput 167 → 31,634 samples/s, a **189× swing**. And the 3 h 11 m training
run held ~104 % CPU on a 10-core machine, leaving ~90 % of the hardware idle.
Parallelism and batching dominate any hardware change at this scale.

## 3) Were different qubit counts tested, and how do they affect training?

**Yes — on the Pavia task**, sweeping 2→6 qubits by varying retained band count
k, 3 seeds each:

| k | qubits | QNN acc | s.d. | LogReg | state-prep CNOTs | sim depth |
|---|---|---|---|---|---|---|
| 4 | 2 | 91.96 % | 0.38 | 91.62 % | 1 | 8 |
| 8 | 3 | 91.84 % | 0.22 | 90.84 % | 4 | 14 |
| 16 | 4 | 95.16 % | 0.30 | 93.09 % | 11 | 14 |
| 32 | 5 | **95.64 %** | 0.26 | 93.87 % | 26 | 14 |
| 64 | 6 | 94.27 % | 0.49 | 94.31 % | 57 | 14 |

- The only real gain is **3→4 qubits**: +3.3 pp, well clear of the ~0.4–0.5 pp
  run-to-run noise floor.
- **5 qubits is the nominal peak but not a defensible one** — 4→5 gains 0.48 pp,
  inside the noise floor.
- **6 qubits is worse**, and is where logistic regression overtakes the QNN.
- Cost grows exactly as **2ⁿ − n − 1** state-preparation CNOTs. Going 16→64
  bands costs 5× the CNOTs and *loses* 0.9 pp.
- **Simulation hides this entirely** — circuit depth is flat at 14 for every
  k ≥ 8. Qubit count looks free in simulation and is not.

**On OSCD, qubit count has not been swept.** It is fixed at 4
(`PCA_COMPONENTS = 4` → 8 device wires), and a sweep is currently *blocked by a
hardcoded literal* — see answer 4.

## 4) Is this a CNN? Do architecture changes matter? Is the setting optimal?

**It is not a CNN. There is no convolution and no spatial context whatsoever.**

Architecture as trained (`src/quantum_classifier.py:303-327`, `:169-180`):

```
x1, x2   4 PCA components per timestamp
  concat                          -> 8 features
  Linear(8, 8)                    classical,  72 params
  AngleEmbedding(rotation="Y")    8 wires, one feature per wire
  SimplifiedTwoDesign(n_layers=3) 8 wires,    42 params
  <Z> on wire 1
  Sigmoid
```

**114 trainable parameters total.** With `PATCH_SIDE = 1`, every pixel is
classified from its own spectra alone — no neighbourhood, no texture. For change
detection that is a severe handicap and is the most likely reason performance
saturates. A `ConvolutionalQuantumNeuralNetwork` and a `src/patch/` pipeline
exist but were not used.

**Architecture changes matter, and two are currently blocked or wasted:**

- **Qubit count is pinned by a literal.**
  `initial_layer_weights=[0,0,0,0,0,0,0,0]` (line 322) is a hardcoded 8-element
  list, correct only when `2 × NUM_QUBITS = 8`. Any qubit sweep on OSCD needs
  this fixed first.
- **The initial rotation layer is dead capacity.** Those weights are zeros *and*
  non-trainable — passed as a constant rather than registered in
  `weight_shapes` — so the first RY layer is the identity. Our Pavia
  implementation makes that layer trainable.
- **Readout is one wire of eight.** `qml.expval(qml.PauliZ(1))` reads wire 1
  only; the other seven contribute solely through entanglement. The choice is
  undocumented and un-ablated.
- `n_layers = 3` has never been varied.

**Is the current setting optimal? There is no evidence that it is.** Nothing has
been swept on the change-detection task — not depth, qubit count, readout wire,
nor embedding type. Two concrete findings from the training history
(`analysis/fig1_training_dynamics.png`):

- **Held-out accuracy peaked at 73.60 % at epoch 18** and fell to 72.74 % by
  epoch 20. The script reports the final epoch, so ~0.9 pp was discarded for
  want of checkpoint selection.
- **Training and validation decouple for 9 epochs.** Train accuracy climbs
  60.8 % → 69.9 % while held-out sits flat near 63 %, then climbs sharply from
  epoch 10. A third of the run produced no generalisation.

And from `analysis/fig3_threshold.png`: **0.5 is not the right operating point.**
Sweeping the threshold gives F1 0.197 at 0.69, versus 0.153 at 0.5 — a 29 %
relative improvement for a one-line change and no retraining.

## 5) For hyperspectral, were limited band counts tested for change detection?

**Two separate results, and the honest answer distinguishes them.**

**Limited bands were tested on hyperspectral — but for a different task.** On
Pavia University (ROSIS, 103 bands) we swept k ∈ {4, 8, 16, 32, 64} across five
selection strategies — uniform, PCA, mutual information, F-score, decorrelated
MI. Accuracy saturates at 16–32 bands: 16 bands on 4 qubits reaches 95.16 %,
beating logistic regression on all 103 bands (94.33 %). A small, well-chosen
subset retains essentially all the discriminative signal.

That task is **single-date per-pixel classification** — closer to abnormality
detection than to change detection. It is a different purpose, and the numbers
should not be read as change-detection results.

**Reduced-dimensionality change detection does work, but via PCA rather than
informed selection.** On OSCD the 10 Sentinel-2 ≤20 m bands are compressed to 4
PCA components per timestamp before reaching the QNN. That is a genuine
band-limited change-detection result — with the caveat in the correction above
about how it is scored.

**What has never been tested** is whether *selected* bands beat *PCA-compressed*
bands for change detection. Two issues with the current PCA make this worth
doing:

1. `pca1` and `pca2` are fit **independently per timestamp**
   (`src/oscd_dataloader.py:108, 115`), so t₁ and t₂ land in different bases.
   PCA components carry arbitrary sign and ordering, so an unchanged pixel is
   not guaranteed the same feature vector at both dates — some apparent "change"
   is basis rotation. `analysis/fig5_separability.png` quantifies how weakly
   ‖PCA(t₂) − PCA(t₁)‖ separates changed from unchanged pixels before the QNN
   sees anything.
2. The evaluation branch calls `fit_transform` rather than `transform`
   (`:121-134`), refitting on evaluation data. No label leakage — PCA is
   unsupervised — but the evaluation is transductive. The `if`/`elif` branches
   are byte-identical.

---

## Figures

| File | Shows |
|---|---|
| `analysis/fig1_training_dynamics.png` | Train/held-out accuracy and loss, 20 epochs; the 9-epoch plateau and the epoch-18 peak |
| `analysis/fig2_change_maps.png` | Full-resolution predicted probability, ground truth, TP/FP/FN error map per city |
| `analysis/fig3_threshold.png` | Precision/recall/F1 vs threshold, and the precision–recall curve |
| `analysis/fig4_per_city.png` | Per-city metric breakdown |
| `analysis/fig5_separability.png` | Whether input features separate change before the model |

Per-city results at threshold 0.5, full resolution — the aggregate hides a wide
spread:

| City | size | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|
| cupertino | 1015 × 788 | 0.881 | 0.108 | 0.553 | 0.180 |
| beirut | 1180 × 1070 | 0.795 | 0.090 | 0.723 | 0.159 |
| mumbai | 858 × 557 | 0.792 | 0.060 | 0.489 | 0.107 |

## Postscript — the 4D experiment and explainability (2026-08-03)

Two follow-up studies extend these answers; full data in
`analysis/experiment_4d_results.json` and `analysis/xai_results.json`.

**4D change detection (3D structures + time), run on a Colab T4**
(`analysis/experiment_4d_colab.py`). Real ModelNet10 meshes voxelized at
64³, synthetic structural edits (29 constructions, 13 demolitions), half the
sequences unchanged. Findings:

- **The evaluation-inflation mechanism replicated in 4D.** The
  paper-faithful 8-wire arm, evaluated at natural prevalence (0.73%
  changed voxels), scores **F1 0.060** (precision 0.031, recall 0.966 —
  the flag-everything syndrome again). The same predictions under the
  paper's class-balanced protocol score **F1 0.876** — a 14.7× inflation,
  larger than the 4.5× found on OSCD. The protocol, not the task,
  manufactures the headline number.
- **The 20-wire amplitude arm trained on GPU but did not learn**: accuracy
  0.500 at n=24 (chance). The pre-registered count-only baseline also
  failed (F1 0.0) — the minimum-edit-size control kept small edits below
  the occupancy-noise floor — so by the experiment's own criterion no
  spatial change detection is claimed. What the arm does establish is
  feasibility and cost: 11.7 s/epoch at 20 wires on a T4, against a
  hardware state-preparation price of 2²⁰−21 = 1,048,555 CNOTs per sample.
  Simulable, not executable.

**Explainability (`analysis/xai_oscd.py`, per arXiv:2211.01441).** Exact
Baseline SHAP and Integrated Gradients on the trained OSCD model — the
tractable case, 8 angle-encoded features:

- False positives are **74% driven by spectral level** (identical at both
  dates — "urban-ness") vs a 50% null; even true positives are 62%
  level-driven. The fig2 error-map reading is now a per-decision quantity.
- Attribution concentrates almost entirely on pc1 of both dates, and the
  exact Fourier spectrum of the quantum core is 52% additive / 96% within
  interaction order ≤ 3: the entangled 114-parameter circuit learned a
  function a small classical additive model represents.
- The 4D model inverts the tractability: at 2²⁰ amplitude-encoded
  features, exact SHAP needs ~10^315,653 coalitions and the Fourier
  structure behind qSHAP does not exist (amplitude encoding is not
  rotation encoding). **The embedding that makes 4D simulable — and the
  GPU worthwhile — is the embedding that makes per-feature explanation
  intractable.** The fallback is group-level attribution (per-date,
  per-supervoxel).

## Open items

1. Re-score against Daudt's baseline on the natural distribution — the current
   comparison is not valid.
2. Add checkpoint selection; recover the ~0.9 pp thrown away at epoch 20.
3. Move the operating point off 0.5, or calibrate it.
4. Fix the hardcoded `initial_layer_weights` so qubit count can be swept on OSCD.
5. Make the initial RY layer trainable.
6. Share one PCA basis across both timestamps, and use `transform` at
   evaluation.
7. Introduce spatial context (`PATCH_SIDE > 1`) — the single largest expected
   gain.
