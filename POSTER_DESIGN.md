# Poster Design Brief
**Working title:** Quantum Change Detection for Hyperscale AI Data Center Monitoring — A Cost-Reality Assessment

**Assumed format:** A0 portrait, 3 columns, print-to-PDF from HTML. Flag if it's A1/landscape/PowerPoint and I'll re-cut the layout.

---

## 1. The angle

Krawec (FAS, 05.12.26) establishes the analytic problem: EO satellite imagery is **best suited to tracking data center construction over time** — an 8-stage taxonomy (site clearing → data hall planning → foundation → steel framing → roofing → cladding → HVAC/substations/cooling → commissioning). Her two case studies (Khazna Ajman, xAI Colossus) are done by hand, one site at a time. That doesn't scale to a global buildout.

Automated pixel-level change detection on multispectral imagery is the obvious fix. The project's resource set points at a specific route: the QNN change-detection work of Rybotycki, Gupta & Gawron (arXiv 2503.08962v3), its Zenodo artifacts (record 14784888), the ONERA/OSCD dataset, Qiskit/PennyLane, and OLCF Frontier for simulation.

**The poster asks: does the quantum route actually help this mission today?** The defensible answer from the resources is *no, and here is the quantified reason* — which is a stronger poster than a hand-wavy "quantum promises speedup." It is falsifiable, policy-relevant, and buildable in an hour because the underlying numbers already exist and are real.

**One-line takeaway (top of poster, big):** *The binding constraint on quantum change detection is software and QPU cost — not the algorithm. Monitoring one data center site to Krawec's 8-stage standard would cost ~$6M in QPU time at current market rates.*

---

## 2. Panel layout (A0 portrait, 3 columns)

**Header band** — title, your name/affiliation, the one-line takeaway in 48pt.

### Column 1 — The problem
- **P1. Why data center monitoring matters.** EPRI: data centers → 9–17% of US electricity by 2030 (from 4–5%). Announcements vs. ground truth diverge; Krawec shows satellite imagery as independent verification. *Pull the Ajman/Colossus contrast.*
- **P2. What EO imagery can and cannot do.** Krawec's findings, condensed: good for construction, weak for ongoing operations, cannot see AI-specific hardware inside a finished hall. The 8-stage construction taxonomy as a **graphic** — this is your best visual and it's free.
- **P3. The scaling gap.** Manual analyst workflow → N sites × M revisits. State the arithmetic explicitly. This is the hinge into the rest of the poster.

### Column 2 — The quantum approach, tested
- **P4. The model.** Hybrid quantum-classical: classical perceptron → amplitude embedding → 3× `SimplifiedTwoDesign` → single-qubit measurement → sigmoid. 8 qubits, depth 44, <200 trainable parameters. **Figure: circuit schematic.**
- **P5. Does it work in simulation?** Best noiseless accuracy **0.74** (amplitude + STD×3), from a 12-configuration sweep. Classical FCN baseline is ~**15 percentage points better** — with ~1M parameters vs. <200. **Figure: training curves** (real data, Zenodo `training_curve_data.txt`: train acc 0.709→0.760, val acc 0.632→0.713 over 20 epochs).
- **P6. Does it survive real hardware?** `ibm_brisbane`: **67.4% (sim) → 56.7% (device)**, a 10 pp drop. Device-oriented noisy training **never completed** — out-of-memory even on compute clusters. **Figure: sim-vs-device bar chart** with the three explainability metrics (Sureness 0.118→0.298; Confidence 0.523→0.510; Imbalance −110→−647). The imbalance blow-up is the relaxation-decoherence signature — good talking point, it's the one place the paper does genuine new science.

### Column 3 — The cost wall, and what would change it
- **P7. What it cost.** 1,815 samples classified in **14h 11m** of QPU time = **2m 13s per sample** = **$81,696** at $96/min pay-as-you-go. That's **5.8%** of the 31,256-sample test set. Full test set: **~$6.4M**.
- **P8. Cost-to-mission — THE MONEY FIGURE.** *This is the original contribution and the thing to build.* Take Krawec's operational scenario — monitor a hyperscale site across the 8 construction stages — and project QPU cost as a function of scene size and revisit count. Overlay the classical-GPU cost line. Log-scale y-axis. The gap will be ~6 orders of magnitude. **Nobody has connected the QPU cost model to a real remote-sensing mission; this figure is why the poster is worth standing in front of.**
- **P9. Break-even conditions.** Invert the figure: what has to be true for quantum to compete? Rough thresholds to state as an explicit list — cost/minute, samples/minute, and the fact that noisy simulation must become tractable before device-oriented training is even possible. Be honest that these are extrapolations.
- **P10. Verdict + next steps.** Not viable for operational monitoring today. Near-term value of the resource set is as a **benchmark harness**, not a deployed capability. Next: run the classical FCN baseline on OSCD for a same-machine comparison; test whether Frontier-scale simulation lifts the OOM ceiling.

**Footer** — sources (FAS, arXiv 2503.08962v3, Zenodo 14784888, OSCD/ONERA, OLCF training series), QR code to the repo.

---

## 3. What to build in the next hour

In `quantumml/`, four scripts → four figures. Ordered by value-per-minute; if we run out of time we stop after F3 and P8 uses hand-computed numbers.

| # | Script | Output | Data source | Est. |
|---|--------|--------|-------------|------|
| F1 | `fig_training_curves.py` | Train/val accuracy + loss, 20 epochs | `14784888/training_curve_data.txt` — **real, already on disk** | 10 min |
| F2 | `fig_sim_vs_device.py` | Grouped bars: accuracy + Sureness/Confidence/Imbalance | Paper Table 3 — **real** | 10 min |
| F3 | `fig_cost_to_mission.py` | **Money figure.** QPU $ vs. scene size × revisits, log y, classical line overlaid | Derived from Zenodo `computations_cost_estimation.xlsx` (2m13s/sample, $96/min) | 20 min |
| F4 | `run_qnn_demo.py` | Small noiseless PennyLane QNN, our own accuracy number on a handful of OSCD patches | PennyLane + OSCD patch | 20 min, **optional** |

F4 is the one at risk — PennyLane install and OSCD patch extraction can eat the hour. It only adds "we ran it ourselves" credibility; F1–F3 carry the argument alone. My call: build F1–F3 first, attempt F4 with whatever is left.

**Assets already local, no download needed:** Zenodo record unpacked at `14784888/`, `2503.08962v3.pdf`, `spectral-master.zip` (SPy), `quantum-training-series-main.zip` (OLCF/PennyLane-on-Frontier examples), `fully-convolutional-change-detection.ipynb` (the classical baseline), FAS PDF. `Tomev/qnn_change_detection` is **not** local — clone it if F4 goes ahead.

---

## 4. Design notes

- **Two colors plus grey.** One for classical, one for quantum. Every figure obeys it. Nothing else gets color.
- **Body text ≥ 24pt, figure captions ≥ 20pt.** A0 is read from 1.5m.
- **Each panel gets a one-sentence bolded finding as its heading**, not a topic label. "Hardware costs $96/min" beats "Cost Analysis."
- **Lead with the takeaway, not the method.** People read the header band and one figure. Make that figure P8.
- **Do not oversell.** The honest framing — a rigorous negative result with a number attached — is the strength here. The failure modes in the source paper (the untrained-weights bug, the OOM wall, the IQM incompatibility) are the interesting finding about the state of QML software, and they're worth a line in P6.

---

## 5. Open question for you

The FAS paper is public and cites public case studies; the Zenodo/arXiv material is open. Everything above draws only on those. **Confirm the poster is cleared as fully open-source-derived before printing.**
