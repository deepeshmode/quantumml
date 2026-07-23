# Hyperspectral → multispectral → QNN

Builds the preprocessing pipeline set out in the project brief — reduce hyperspectral
imagery to its informative bands, produce a multispectral stack, run the QNN change
detection on it — and asks what that reduction costs and buys.

## Findings

**1. Band count is a quantum cost lever, not just a data-reduction step.**
Amplitude embedding packs `k` bands into `log2(k)` qubits, so band selection sets
circuit width directly. Exact state preparation costs O(2ⁿ) CNOTs on hardware
(Möttönen et al.), which is the term the simulator hides — `default.qubit` applies
`StatePrep` in one op, so measured simulator depth stays flat at 14 while real
hardware cost doubles with every added qubit.

**2. Accuracy plateaus at 16–32 bands while cost keeps doubling.**

| k | qubits | QNN acc | LogReg acc | state-prep CNOTs |
|---|--------|---------|------------|------------------|
| 4 | 2 | 91.96% | 91.62% | 1 |
| 8 | 3 | 91.84% | 90.84% | 4 |
| **16** | **4** | **95.16%** | 93.09% | **11** |
| 32 | 5 | 95.64% | 93.87% | 26 |
| 64 | 6 | 94.27% | 94.31% | 57 |

Going 16 → 64 bands *loses* 0.9 pp accuracy and costs 5× the CNOTs. k=32 is the
peak; k=16 gives up 0.5 pp for 2.4× less state-prep cost.

**3. Uniformly spaced bands beat supervised selection.** Decorrelated mutual
information tops out at 93.78% — below uniform at every k ≥ 16. This is good news
for the pipeline: a fixed multispectral sensor hands you evenly-spaced bands for
free, so the expensive hyperspectral step may not be buying much for this task.

**4. Naive univariate selection picks redundant neighbours.** Plain mutual
information chose 8 adjacent blue bands (468–506 nm) — near-duplicates in a
hyperspectral cube. `select_bands_decorrelated` rejects candidates correlated
above 0.95 with what's already chosen, which spreads selection across 430–860 nm
and picks up the red edge (717–746 nm) where the vegetation/built split lives.

**5. The QNN's margin over logistic regression is real but small**, and only at
low band counts: +2.1 pp at k=16, +1.8 pp at k=32, and gone by k=64 (−0.04 pp).
Both sit near logistic regression on all 103 bands (94.33%).

### Correction to the source paper

Rybotycki et al. report "2 minutes and 13 seconds" per sample on `ibm_brisbane`.
Their own spreadsheet (`14784888/computations_cost_estimation.xlsx`) computes
`1815 samples / 851 min = 2.13`, which is **samples per minute** — the cell label
is the reciprocal. Their totals only support the other reading:

- **28.1 s/sample** (851 × 60 / 1815), not 2 m 13 s
- **$45.01/sample** at $96/min
- Full 31,256-sample test set ≈ **$1.4M**, not the $6.4M reported

The batch durations in the spreadsheet sum to 852 min, consistent with 851.

## Proxy task, and its limits

No hyperspectral scene over a hyperscale data center is publicly available. Pavia
University (ROSIS, 103 bands, 430–860 nm) stands in because its material classes
are what a data center campus is built from: painted metal sheets (roofing),
asphalt and gravel (hardstanding), bitumen (roofing), bare soil (cleared ground).

The binary task — built/disturbed surface vs. undisturbed vegetation — is the
spectral signal Krawec's construction taxonomy tracks at stage 1 (site clearing)
through stage 6 (cladding).

**This is a proxy, not a data center result.** It is a single-date scene, so this
is per-pixel classification, not true bi-temporal change detection; the change
framing is inherited from the task definition, not demonstrated over time. ROSIS
also tops out at 860 nm, so NDVI here samples only the near shoulder of the NIR
plateau and is weaker than Sentinel-2 (842 nm) would give — it separates the
classes, but the values aren't comparable to standard NDVI products.

## Files

| File | What |
|---|---|
| `pipeline.py` | Loading, proxy task, NDVI, RGB, band selection (MI / F-score / PCA / uniform / decorrelated) |
| `qnn.py` | Rybotycki architecture: perceptron → amplitude embedding → 3× SimplifiedTwoDesign → ⟨Z⟩ → sigmoid |
| `run_experiment.py` | Band sweep over k ∈ {4,8,16,32,64} × 3 seeds → `results.json` |
| `make_figures.py` | `fig_pipeline.png` (the brief's 3 subplots), `fig_scaling.png` |
| `POSTER_DESIGN.md` | Poster brief — **cost figures predate the correction above** |

## Reproduce

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy scipy matplotlib scikit-learn h5py spectral torch
./.venv/bin/pip install pennylane "autoray==0.6.12"   # pin: py3.9 gets PennyLane 0.38
unzip -o -j ../HybridSN-master.zip "HybridSN-master/data/PaviaU*.mat" -d data/
./.venv/bin/python run_experiment.py     # ~2 min
./.venv/bin/python make_figures.py
```

## Not done

- **Qiskit alongside PennyLane** (project objective 2 — why simulators disagree).
  The analytic/shot distinction is the likely answer and is a short experiment:
  same weights, `shots=None` vs `shots=1024`.
- **Real bi-temporal change detection.** Needs an image pair; OSCD/ONERA or a
  Sentinel-2 pair over a named data center site would do it. Both STAC endpoints
  (Planetary Computer, Earth Search) are reachable.
- **Hardware run.** Everything here is noiseless simulation.

Sources are open: Pavia University via the HybridSN repo, arXiv 2503.08962v3,
Zenodo 14784888.
