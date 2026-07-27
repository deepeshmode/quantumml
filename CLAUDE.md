# quantumml — working notes for Claude

Hyperspectral → multispectral → QNN change detection. Read `README.md` first for
what the project found; this file is about how to work in it without breaking
things.

## Hard rule: source boundary

Only open sources go in this repo. Pavia University via the HybridSN repo,
arXiv:2503.08962v3, Zenodo 14784888. **Nothing from the NGA internal share** —
no data, no figures, no numbers, no paraphrase. This repo is public. If a task
seems to need internal material, stop and ask rather than working around it.

## Environment

Use the repo venv explicitly — system `python3` has none of the deps:

```bash
./.venv/bin/python run_experiment.py   # ~2 min
./.venv/bin/python sim_comparison.py   # ~2 min
./.venv/bin/python make_figures.py
./.venv/bin/python verify_claims.py    # seconds — no training, just checks
```

Python 3.9, which caps PennyLane at 0.38. `autoray` must stay pinned at 0.6.12
(0.8.x breaks the PennyLane 0.38 import with `no attribute 'NumpyMimic'`).
Don't "upgrade to fix" an import error here — the pins are the fix.

`data/` is gitignored and not shipped: `PaviaU.mat` / `PaviaU_gt.mat` come from
the HybridSN repo zip, see README. Also never commit `.venv/`, `*.mat`,
`demo.gif`, `demo_data.npz`.

## The README's numbers are claims, not decoration

Every table and finding in `README.md` is derived from `results.json` and
`sim_comparison.json`. `verify_claims.py` re-checks them. So:

- After changing `pipeline.py`, `qnn.py`, or `run_experiment.py`, rerun
  `run_experiment.py`, then `verify_claims.py`, and update the README tables in
  the same change. A silently drifted number is worse than a failing script.
- After changing `sim_comparison.py`, rerun it and then `verify_claims.py`.
- If a claim legitimately changes, update the expected value in
  `verify_claims.py` *and* the README prose together — the script is the record
  of what the repo asserts.

## Accuracy noise floor

`split()` uses 3000 train / 1500 test pixels over `SEEDS = [0, 1, 2]`. Run-to-run
s.d. is ~0.4–0.5 pp. Differences below ~1 pp are not real. Don't chase them, and
don't report them as findings.

## Known trap: PennyLane/Qiskit endianness

PennyLane wire *i* is Qiskit qubit *n−1−i*. Under that remap an amplitude vector
passes through **unchanged** — do not also reverse the vector. Reversing it is
the intuitive move and it silently applies an X to every qubit, returning
−0.0333 instead of −0.1670. No error is raised, and the wrong value looks
plausible. `sim_comparison.py` gets this right; copy its convention.

Related: the two frameworks agree analytically to ~5e-16. If they appear to
disagree, the circuits differ — check gate-by-gate before blaming the simulator.
That is exactly the bug in the upstream repo (see finding 8).

## Sibling directories (outside this repo)

- `../qnn_change_detection/` — clone of `Tomev/qnn_change_detection`, the paper's
  code. Separate Python 3.11 venv with the authors' Zenodo pins (PennyLane
  0.38.1, *yanked* on PyPI; qiskit 1.1.0; qiskit-iqm 13.13). `REPRODUCTION.md`
  there is the reproduction write-up: 72.8% val acc / F1 0.691 on OSCD.
  Its MLflow tracking URI is hardcoded to `127.0.0.1:8080`, so a local
  `mlflow server` must be running or training stalls on retries.
- `../oscd_root/` — OSCD dataset, 11 train / 3 test cities (test labels weren't
  in the archive, so the split deviates from the paper's 14/10).
- `../fully_convolutional_change_detection-master/` — Daudt's Siam-U-Net, the
  classical baseline for the paper's "~15 points worse" claim.

## Stale by design

`POSTER_DESIGN.md` and the other `POSTER_*.md` files hold **pre-correction** QPU
cost figures ($6.4M / "2 min 13 s" per sample). The corrected numbers are
28.1 s/sample, $45.01/sample, ≈$1.4M — in the README and `run_experiment.py`.
Don't propagate the poster figures into new work.
