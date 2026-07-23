# Poster brief — paste into Claude Design or Gamma

Everything below the line is the prompt. Attach `fig_pipeline.png` and
`fig_scaling.png` when you paste it.

---

## PROMPT

Design an academic conference poster. Follow these specifications exactly.

### Format

A0 portrait (841 × 1189 mm). Three equal columns with 50 mm margins on all
sides and between columns. Horizontal title bar across the full width at the
top. Every element aligns to the column grid — nothing intrudes into a margin.

### Design language — follow strictly

These rules come from Faulkes, *Better Posters* and Carter, *Designing Science
Presentations*. Where a default instinct conflicts with a rule below, the rule
wins.

- **Total body text must not exceed 800 words.** There is an inverse
  correlation between text volume and whether anyone reads it. Cut ruthlessly.
- **Write in prose, not bullet points.** Bullets destroy narrative. Each
  section is two to five short sentences that connect causally.
- **No abstract.** A poster is already a summary; an abstract summarizes a
  summary and wastes space.
- **No tables.** Convert any tabular content to a figure or a sentence.
- **No boxes or borders around sections.** Boxes read as desperate. Use white
  space and alignment to separate content instead.
- **Plain background.** No photographs, gradients, or textures behind text.
- **Sans-serif throughout.** Title letters at least 25 mm high. Body text large
  enough to read at 1.5 m — nothing below 24 pt.
- **One entry point.** The eye should land in the top-left of the left column
  and move down, then right.
- **Institutional logos small and in the title bar only.**

### Palette

Two colors plus neutrals, applied consistently and never decoratively:

- Quantum / built surface: `#2a78d6` (blue)
- Classical / vegetation: `#eb6834` (orange)
- Third series where needed: `#1baf7a` (green)
- Text: `#0b0b0b` primary, `#52514e` secondary
- Background: `#fcfcfb`

This palette is colorblind-safe and validated. Do not add hues. Identity must
never rest on color alone — label series directly.

### Title bar

Main title, set as a conclusion rather than a topic:

**Satellite imagery cannot identify AI hardware — so the analytic value lies in
change over time**

Subtitle, smaller:

*Automating that change detection with quantum machine learning: band selection,
not qubit count, sets the cost*

Below that, in smaller type: author name, NGA, and the date. Nothing else in
the title bar.

### Narrative spine

The poster follows an "and / but / therefore" structure. Preserve the causal
chain across sections — each one should follow from the last.

- **AND** — satellite imagery reliably locates and sizes data centers, and
  policy demand for visibility into AI infrastructure is rising sharply.
- **BUT** — it cannot answer the AI-specific questions.
- **THEREFORE** — the value must come from monitoring change over time, which
  means automating change detection — so the operative question is what that
  automation costs.

### Column 1 (left) — the problem

**Section heading: "Locating data centers is not the same as finding AI"**

Write four to six sentences making these points in prose. Do not bulletize.

Satellite imagery of data centers offers limited immediate value for
AI-specific insight. AI hardware is a small share of what is inside — EPRI
estimates AI workloads account for roughly 10–20% of data center power, and
TrendForce put AI servers at about 12% of 2024 server shipments. That hardware
usually shares space with general-purpose compute rather than occupying
dedicated facilities; GPT-4 was reportedly trained in a Microsoft data center
in Iowa far larger than the ~20,000 A100s involved would require. There are no
reliable visual identifiers: higher power density means more cooling
infrastructure, but that is a coarse proxy, and AI facilities are not otherwise
visually distinguishable from conventional ones. Even where AI hardware is
known to be present, inferring chip counts and computational performance from
external structures or energy data remains imprecise.

**Section heading: "The way forward is temporal, not spatial"**

Two to three sentences: tracking development over time is what would raise the
value proposition, because as AI takes a growing share of compute, correlating
new construction with AI-specific demand becomes more feasible. Findings should
be cross-correlated with other open sources — investment records, permitting,
construction filings. Continuous monitoring at scale means automated change
detection, which is where this work begins.

Place a simple flowchart here showing the eight construction stages a facility
passes through: site clearing → data hall planning → foundation → steel framing
→ roofing → cladding → HVAC and substations → commissioning. Flowcharts are
underused on posters and work well. Keep it one color plus neutrals.

### Column 2 (middle) — what was built and what it found

**Section heading: "Hyperspectral to multispectral to quantum classifier"**

Place `fig_pipeline.png` large — it should dominate this column. Caption in two
or three sentences: a 103-band hyperspectral cube is reduced by band selection
to a k-band multispectral stack, which feeds a hybrid model — classical
perceptron, amplitude embedding, three SimplifiedTwoDesign layers, single-qubit
measurement. Amplitude embedding packs k bands into log₂k qubits, so band
selection sets circuit width directly.

State plainly, in the caption, that this uses Pavia University as a proxy: no
hyperspectral scene over a hyperscale data center is publicly available, and
this scene's material classes — painted metal sheets, asphalt, bitumen, gravel,
bare soil — are what a data center campus is built from.

### Column 3 (right, upper) — the result

**Section heading: "Accuracy plateaus while hardware cost keeps doubling"**

Place `fig_scaling.png`. Caption in three or four sentences: accuracy peaks at
32 bands (95.6%) and falls by 64; going from 16 to 64 bands loses 0.9
percentage points while costing five times the state-preparation CNOTs.
Uniformly spaced bands beat supervised mutual-information selection at every
k ≥ 16 — which is good news, because a fixed multispectral sensor supplies
evenly spaced bands for free. The quantum model's margin over logistic
regression is +2.1 points at k=16 and gone by k=64.

**Bottom line — set this larger than body text, as the single take-home:**

Band selection, not qubit count, is the first-order cost variable for quantum
machine learning on satellite imagery.

### Column 3 (right, lower) — limits and sources

**Section heading: "What this does not show"**

Three or four sentences, stated plainly. This is single-date per-pixel
classification on a proxy scene, not bi-temporal change detection over a real
data center. All results are noiseless simulation; nothing ran on quantum
hardware. CNOT counts are modeled for exact amplitude embedding rather than
measured, because the simulator applies state preparation in a single operation
and hides the term entirely.

Add one short note: the source paper's reported per-sample cost appears to
invert its own units — its spreadsheet computes samples per minute, giving
28 s and roughly $45 per sample rather than the 2 min 13 s stated.

Then, in small type: sources — Rybotycki, Gupta & Gawron, arXiv:2503.08962v3;
Zenodo 14784888; Pavia University (ROSIS); EPRI; TrendForce. Acknowledgment to
Siwei Qiu. A QR code to the code repository, small, bottom right.

### Final checks

Confirm before returning: under 800 body words, no bullet lists, no tables, no
boxes, every element aligned to the grid, title readable from across a room,
and the bottom line visually the most prominent text after the title.
