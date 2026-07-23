# Claude Design prompt

Everything below the line is self-contained — paste it as one message and attach
`fig_pipeline.png` and `fig_scaling.png`. Attach `qr_repo.png` too, or let the
tool generate a QR to the URL given.

---

Design a print-ready academic conference poster. Follow every specification
below exactly; where your default instinct conflicts with a rule here, the rule
wins. All body copy is given verbatim — set it as written, do not paraphrase,
summarise, or expand it.

## Canvas

A0 portrait, 841 × 1189 mm. Three equal columns. Margins of 64 mm on all four
sides and between every column. Every element aligns to the column grid; nothing
intrudes into a margin. Export a print-ready PDF at 300 dpi with fonts embedded,
plus a web-resolution PNG.

## Typography

| Element | Spec |
|---|---|
| Title | 100 pt bold, maximum two lines |
| Subtitle | 44 pt regular |
| Author line | 40 pt |
| Section headings | 48 pt bold, in `#2a78d6` |
| Body text | 32 pt |
| Figure captions | 26 pt |

Two typefaces maximum, both sans-serif — one for headings, one for body. No
serif faces anywhere.

## Colour

Exactly three colours plus neutrals. Do not introduce any others.

- `#2a78d6` blue — quantum, built surface, section headings
- `#eb6834` orange — classical, vegetation
- `#1baf7a` green — third series
- Text: `#0b0b0b` primary, `#52514e` secondary
- Background: `#fcfcfb` flat, no gradient, no texture, no photograph

## Composition rules

- Roughly 30% white space, 40% title and text, 30% graphics. If it feels
  crowded, cut text — never shrink the margins.
- Flow must be unambiguous: entry at top-left, down column 1, then column 2,
  then column 3. A reader should never wonder where to look next.
- **No bullet lists.** Every section is prose. Bullets destroy narrative.
- **No tables.**
- **No boxes, borders, or rules around sections.** Separate content with white
  space and alignment only. Boxed sections read as desperate.
- **No abstract.** A poster is already a summary.
- Institutional logo, if any: one only, small, right-aligned in the title bar,
  no taller than the affiliation text.

---

# COPY

## Title bar

**Quantum machine learning on satellite imagery: band selection, not qubit
count, sets the cost**

*What a variational quantum classifier costs to run on multispectral data, and
which choice actually drives it*

Deepesh V. Chaudhari
National Geospatial-Intelligence Agency · 23 July 2026

## Column 1

### Which construction stages leave a spectral signature

Hyperscale data centre construction is one of the few AI-infrastructure signals
observable from orbit. Identifying the compute itself is not realistic — AI
hardware shares halls with general-purpose servers, carries no distinguishing
visual signature, and accounts for only 10–20% of data centre power. What is
tractable is detecting change across repeat imagery.

Construction proceeds through eight stages. Six alter surface material in ways a
spectrometer registers: clearing strips vegetation, foundations expose concrete,
framing and roofing introduce metal, cladding changes it again, and cooling
plant adds structure. Planning and commissioning leave nothing to see.

That reduces to per-pixel classification, repeated over large scenes and many
revisits. The cost of that repetition is what this work measures.

**FIGURE — you generate this one.** A vertical flowchart, single column, eight
numbered stages top to bottom, connected by thin arrows. Fill only, no outlined
boxes. Stages 1, 3, 4, 5, 6 and 7 filled `#2a78d6`; stages 2 and 8 filled
neutral grey `#8a8a85`.

1. Site clearing
2. Data hall planning
3. Foundation
4. Steel framing
5. Roofing
6. Cladding
7. HVAC, substations and cooling
8. Commissioning

Caption: *Six of eight stages leave a detectable spectral signature (blue);
planning and commissioning (grey) do not. Roofing and cladding are precisely the
painted-metal and bitumen classes this model separates.*

## Column 2

### Hyperspectral to multispectral to quantum classifier

**FIGURE — place `fig_pipeline.png` here, large enough to dominate the column.**

A 103-band hyperspectral cube is reduced by uniform band selection to a 16-band
multispectral stack, which feeds a hybrid model: a classical perceptron,
amplitude embedding, three SimplifiedTwoDesign layers, and a single-qubit
measurement. This is an independent implementation of the architecture described
in Rybotycki et al., not a run of their published code. Amplitude embedding
packs k bands into log₂k qubits, so band selection sets circuit width directly —
a preprocessing choice that propagates into hardware cost.

Pavia University stands in as a proxy. No public hyperspectral scene exists over
a hyperscale data centre, and this scene's material classes — painted metal
sheets, asphalt, bitumen, gravel, bare soil — are what such a campus is built
from. The map shows a single run at 94.7%; the sweep at right reports
95.2 ± 0.3% across three seeds.

## Column 3

### Accuracy plateaus while circuit cost keeps doubling

**FIGURE — place `fig_scaling.png` here.**

Accuracy peaks at 32 bands (95.6%) and falls by 64. Moving from 16 to 64 bands
loses 0.9 percentage points while costing five times the state-preparation
CNOTs, because exact amplitude embedding scales as O(2ⁿ). Uniformly spaced bands
beat supervised mutual-information selection at every k ≥ 16 — useful, since a
fixed multispectral sensor supplies even spacing for free. The quantum model's
margin over logistic regression is 2.1 points at k = 16, and gone by 64.

### Why inference is the thing that costs

Estimating a quantum expectation requires repeated measurement. The same pixel
through the same circuit returns a different answer each time, converging on the
exact value as 1/√N — four times the shots buys twice the precision. Built
natively in both frameworks, PennyLane and Qiskit agree analytically to
5×10⁻¹⁶; what differs between runs is sampling, which both share. Certainty is
purchasable, and that is what makes inference, not training, the recurring cost.
See the monitor.

**SET THIS APART — larger than body text, the single most prominent element
after the title. No box; use white space and weight.**

> Band selection, not qubit count, is the first-order cost variable for quantum
> machine learning on satellite imagery.

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

## Footer of column 3, small type

*Sources: Rybotycki, Gupta & Gawron, arXiv:2503.08962v3 · Zenodo 14784888 ·
Pavia University (ROSIS) · Krawec, FAS · EPRI. Code:
github.com/deepeshmode/quantumml. With thanks to Siwei Qiu.*

**QR code**, small, bottom-right corner, pointing to
`https://github.com/deepeshmode/quantumml`.

---

## Before you return the design, verify

- Title fits in two lines and is legible from across a room
- Body copy is set verbatim, with no bullets, tables, or boxed sections
- Exactly three colours are used; section headings are `#2a78d6`
- Margins and column gaps are a true 64 mm
- Roughly 30% of the area is white space
- The bottom-line statement is the most prominent text after the title
- Both supplied figures are placed at full resolution, not downsampled
- Nothing overflows a column or crosses a margin
