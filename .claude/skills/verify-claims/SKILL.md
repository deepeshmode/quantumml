---
name: verify-claims
description: Rerun the affected experiment and reconcile README.md's published numbers with the data after changing code in this repo. Use after editing pipeline.py, qnn.py, run_experiment.py, or sim_comparison.py; when verify_claims.py fails; or when asked to check whether the README is still accurate. Triggers on "verify claims", "check the README numbers", "did that change the results", "sync the tables".
---

# Verify and sync published claims

Every table and finding in `README.md` is derived from `results.json` and
`sim_comparison.json`. `verify_claims.py` re-derives them and fails on drift.
This skill closes that loop: rerun what the change affected, check, and — only
when a number legitimately moved — update the README and the expected value
*together*.

All commands use the repo venv. System `python3` has none of the deps.

## 1. Work out what to rerun

Check what actually changed (`git status --short`, `git diff --stat`), then:

| Changed | Rerun | Time |
|---|---|---|
| `pipeline.py`, `qnn.py`, `run_experiment.py` | `./.venv/bin/python run_experiment.py` → `results.json` | ~2 min |
| `sim_comparison.py` | `./.venv/bin/python sim_comparison.py` → `sim_comparison.json` | ~2 min |
| `make_figures.py` | `./.venv/bin/python make_figures.py` | figures only, no claims |
| README prose only | nothing — skip to step 2 | |

If both experiment scripts are affected, rerun both before verifying.

`run_experiment.py` needs `data/PaviaU.mat` and `data/PaviaU_gt.mat`, which are
gitignored and not shipped. If `data/` is empty, stop and say so — get the files
from the HybridSN repo zip per `README.md`. Do not synthesize substitute data or
report numbers from the existing `results.json` as if they were a fresh run.

## 2. Verify

```bash
./.venv/bin/python verify_claims.py
```

Exit 0 means the README is accurate — you are done. Report which checks ran and
stop.

Exit 1 means at least one claim moved. Go to step 3. The script prints each
failure as `published X, actual Y`.

## 3. Classify each failure before touching anything

For accuracy claims, apply the noise floor: `split()` uses 3000 train / 1500
test pixels over `SEEDS = [0, 1, 2]`, and run-to-run s.d. is ~0.4–0.5 pp.

- **Moved by less than ~1 pp**, and the code change was not supposed to affect
  accuracy → this is noise, not a finding. Suspect an unintended change. Do not
  report it as a result and do not update the README to chase it.
- **Moved by more than ~1 pp**, or a structural claim changed (qubit counts,
  CNOT counts, depth, which k is the peak, orderings like "uniform > decorr") →
  real. Continue.

Structural claims have no noise floor. `stateprep_cnots_hw` should equal
`2^n - n - 1` exactly; if that identity breaks, it is a bug in the state-prep
accounting, not a number to re-publish.

For the PennyLane/Qiskit checks, remember the endianness trap: PennyLane wire
*i* is Qiskit qubit *n−1−i*, and under that remap the amplitude vector passes
through **unchanged**. Reversing it as well silently applies an X to every qubit
and returns −0.0333 instead of −0.1670 with no error raised. If analytic
agreement degrades from ~5e-16, compare the circuits gate by gate before
concluding the simulators disagree.

## 4. Update the README and the expected values together

Only for changes classified real in step 3. For each:

1. Update the number in `README.md` — **the table cell and any prose that quotes
   it**. Findings are written as sentences ("k=32 is the peak", "16 → 64 loses
   ~0.9 pp for 5× the CNOTs", "~15 points worse"); a table edit that leaves the
   sentence stale is the failure this whole loop exists to prevent.
2. Update the matching constant in `verify_claims.py` — `SWEEP_TABLE`,
   `CLASSICAL_ALL_BANDS_PP`, `DECORR_BEST_PP`, `ANALYTIC_AGREEMENT`, or the cost
   block.
3. If a *finding* is now false rather than merely shifted (an ordering flipped,
   the peak moved), rewrite the finding. Do not preserve a conclusion the data
   no longer supports by widening a tolerance.

**Never edit an expected value alone to make the script pass.** The script is
the record of what the repo asserts; changing it without changing the README
launders drift into the published claims and defeats the check. If you cannot
explain *why* a number moved, say so and stop rather than making it green.

Do not widen `ACC_TOL_PP` or any tolerance to accommodate drift.

## 5. Re-verify and report

Rerun `./.venv/bin/python verify_claims.py` and confirm exit 0.

Report: what was rerun, which claims moved and by how much, which were noise,
what was edited in `README.md` and `verify_claims.py`, and the final check count.

## Do not touch

`POSTER_DESIGN.md`, `POSTER_BRIEF.md`, `POSTER_FINAL.md`, `POSTER_PITCH.md`, and
`POSTER_PROMPT.md` hold **pre-correction** QPU cost figures ($6.4M, "2 min 13 s"
per sample) and are stale by design. The corrected figures — 28.1 s/sample,
$45.01/sample, ≈$1.4M for the full 31,256-sample test set — live in `README.md`
and `run_experiment.py`. Never sync poster figures into the README, and never
"fix" the posters to match.
