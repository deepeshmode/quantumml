# Final poster copy

A0 portrait, three columns, 50 mm margins. ~640 body words (Carter's ceiling is
800). Prose throughout — no bullet lists, no tables, no boxed sections. Figures
`fig_pipeline.png` and `fig_scaling.png`; `fig_shot_noise.png` runs on the
monitor, not the poster.

Rebalanced for an NGA audience: column 1 is a sensing problem statement, not a
policy brief. The compute-governance framing is cited, not argued — that room
knows imagery better than any paper does.

---

## TITLE BAR

**Quantum machine learning on satellite imagery: band selection, not qubit
count, sets the cost**

*What a variational quantum classifier costs to run on multispectral data, and
which choice actually drives it*

Deepesh V. Chaudhari
National Geospatial-Intelligence Agency · 23 July 2026

---

## COLUMN 1

### Which construction stages leave a spectral signature

Hyperscale data centre construction is one of the few AI-infrastructure signals
observable from orbit. Identifying the compute itself is not realistic — AI
hardware shares halls with general-purpose servers, carries no distinguishing
visual signature, and accounts for only 10–20% of data centre power. What is
tractable is detecting *change* across repeat imagery.

Construction proceeds through eight stages. Six alter surface material in ways a
spectrometer registers: clearing strips vegetation, foundations expose concrete,
framing and roofing introduce metal, cladding changes it again, and cooling
plant adds structure. Planning and commissioning leave nothing to see.

That reduces to per-pixel classification, repeated over large scenes and many
revisits. The cost of that repetition is what this work measures.

**[8-stage flowchart — vertical, numbered, stages 1/3/4/5/6/7 in blue, 2 and 8
in grey]**

*Six of eight stages leave a detectable spectral signature (blue); planning and
commissioning (grey) do not. Roofing and cladding are precisely the
painted-metal and bitumen classes this model separates.*

---

## COLUMN 2

### Hyperspectral to multispectral to quantum classifier

**[fig_pipeline.png — large, dominating the column]**

A 103-band hyperspectral cube is reduced by uniform band selection to a 16-band
multispectral stack, which feeds a hybrid model: a classical perceptron,
amplitude embedding, three SimplifiedTwoDesign layers, and a single-qubit
measurement. This is an independent implementation of the architecture described
in Rybotycki et al., not a run of their published code. Amplitude embedding
packs *k* bands into log₂*k* qubits, so band selection sets circuit width
directly — a preprocessing choice that propagates into hardware cost.

Pavia University stands in as a proxy. No public hyperspectral scene exists over
a hyperscale data centre, and this scene's material classes — painted metal
sheets, asphalt, bitumen, gravel, bare soil — are what such a campus is built
from. The map shows a single run at 94.7%; the sweep at right reports
95.2 ± 0.3% across three seeds.

---

## COLUMN 3

### Accuracy plateaus while circuit cost keeps doubling

**[fig_scaling.png]**

Accuracy peaks at 32 bands (95.6%) and falls by 64. Moving from 16 to 64 bands
loses 0.9 percentage points while costing five times the state-preparation
CNOTs, because exact amplitude embedding scales as O(2ⁿ). Uniformly spaced bands
beat supervised mutual-information selection at every *k* ≥ 16 — useful, since a
fixed multispectral sensor supplies even spacing for free. The quantum model's
margin over logistic regression is 2.1 points at *k* = 16, and gone by 64.

### Why inference is the thing that costs

Estimating a quantum expectation requires repeated measurement. The same pixel
through the same circuit returns a different answer each time, converging on the
exact value as 1/√N — four times the shots buys twice the precision. Built
natively in both frameworks, PennyLane and Qiskit agree analytically to
5×10⁻¹⁶; what differs between runs is sampling, which both share. Certainty is
purchasable, and that is what makes inference, not training, the recurring cost.
**See the monitor.**

> **Band selection, not qubit count, is the first-order cost variable for
> quantum machine learning on satellite imagery.**

### What this does not show

This is single-date per-pixel classification on a proxy scene — not bi-temporal
change detection over a real data centre. All results are noiseless simulation;
nothing ran on quantum hardware. CNOT counts are modelled for exact amplitude
embedding rather than measured, because the simulator applies state preparation
in a single operation and hides the term entirely. Three seeds per point.

The classical baseline here is logistic regression. A geospatial foundation
model — Scale-MAE, arXiv:2212.14532 — is the stronger comparison, and is the
next benchmark rather than a demonstrated result.

One arithmetic note: the source paper's per-sample cost inverts its own
spreadsheet's units, which compute samples per minute — implying 28 s and
roughly $45 per sample rather than the stated 2 min 13 s.

*Sources: Rybotycki, Gupta & Gawron, arXiv:2503.08962v3 · Zenodo 14784888 ·
Pavia University (ROSIS) · Krawec, FAS · EPRI. Code:
github.com/deepeshmode/quantumml. With thanks to Siwei Qiu.*

**[QR code — qr_repo.png, small, bottom right]**

---

## Changes from the previous draft

**Subtitle.** "Satellite imagery cannot identify AI hardware" told a room of
GEOINT professionals what imagery can't do. Replaced with a scope statement.

**Column 1 reframed** from policy motivation to sensing problem. EPRI survives
as a single clause; TrendForce and the GPT-4/Iowa example are cut — they argue a
point this audience doesn't need argued, and the words are better spent on
detectability.

**Objective 2 added** as the "why inference costs" section. It supplies the
mechanism behind the cost claim, which was previously asserted.

**Accuracy corrected** to 94.7% for the pipeline figure, with the three-seed
mean alongside so the two figures no longer appear to disagree.

## Before printing

Pull the current `fig_pipeline.png` — the poster tool's copy is stale (it shows
92.1% from the superseded decorrelated-MI run). Scan the printed QR once; the
repo is public and resolving.
