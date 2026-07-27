"""
Check that README.md's published claims still match results.json / sim_comparison.json.

Every table and finding in the README is derived from those two files. This
re-derives them and fails loudly on drift, so a change to pipeline.py, qnn.py or
sim_comparison.py can't silently invalidate a published number.

Runs in seconds — it reads the JSON, it does not retrain. After rerunning
run_experiment.py or sim_comparison.py:

    ./.venv/bin/python verify_claims.py

Exit 0 = README is accurate. Exit 1 = a claim moved; fix the code or update the
README and the expected value here together.
"""

import json
import sys

# Accuracies are printed to 2 dp in the README, so anything past rounding is drift.
# Seeds are fixed, so unchanged code reproduces these exactly.
ACC_TOL_PP = 0.05

# README finding 2: the band sweep table (uniform selection).
# k -> (qubits, qnn acc %, logreg acc %, state-prep CNOTs)
SWEEP_TABLE = {
    4:  (2, 91.96, 91.62, 1),
    8:  (3, 91.84, 90.84, 4),
    16: (4, 95.16, 93.09, 11),
    32: (5, 95.64, 93.87, 26),
    64: (6, 94.27, 94.31, 57),
}

CLASSICAL_ALL_BANDS_PP = 94.33   # README finding 5
DECORR_BEST_PP = 93.78           # README finding 3
ANALYTIC_AGREEMENT = 5e-16       # README finding 6

# README's cost correction to the source paper.
SEC_PER_SAMPLE = 28.1
USD_PER_SAMPLE = 45.01
QPU_RATE = 96.0
FULL_TEST_SET = 31256
FULL_TEST_USD_M = 1.4

failures = []
checks = 0


def check(ok, label, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}: {detail}")
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))


def close(actual, published, tol, label, unit=""):
    check(abs(actual - published) <= tol, label,
          f"published {published}{unit}, actual {actual:.4f}{unit}")


def main():
    with open("results.json") as f:
        res = json.load(f)
    with open("sim_comparison.json") as f:
        sim = json.load(f)

    runs = {(r["method"], r["k"]): r for r in res["runs"]}
    meta = res["meta"]

    print("\nExperiment metadata")
    check(meta["seeds"] == [0, 1, 2], "seeds are [0, 1, 2]", str(meta["seeds"]))
    check(meta["epochs"] == 15, "15 epochs", str(meta["epochs"]))
    check(meta["n_labeled"] == 41829, "41,829 labelled pixels", str(meta["n_labeled"]))

    print("\nFinding 2 — band sweep table (uniform)")
    for k, (qubits, qnn_pp, clf_pp, cnots) in SWEEP_TABLE.items():
        r = runs[("uniform", k)]
        check(r["n_qubits"] == qubits, f"k={k}: {qubits} qubits", str(r["n_qubits"]))
        close(100 * r["acc_mean"], qnn_pp, ACC_TOL_PP, f"k={k}: QNN accuracy", "%")
        close(100 * r["classical_acc_mean"], clf_pp, ACC_TOL_PP, f"k={k}: LogReg accuracy", "%")
        check(r["stateprep_cnots_hw"] == cnots, f"k={k}: {cnots} state-prep CNOTs",
              str(r["stateprep_cnots_hw"]))

    print("\nFinding 1 — state-prep cost is 2^n - n - 1, hidden by the simulator")
    for k in SWEEP_TABLE:
        r = runs[("uniform", k)]
        n = r["n_qubits"]
        check(r["stateprep_cnots_hw"] == 2 ** n - n - 1,
              f"k={k}: CNOTs = 2^{n} - {n} - 1", str(r["stateprep_cnots_hw"]))
    flat = {runs[("uniform", k)]["depth"] for k in (8, 16, 32, 64)}
    check(flat == {14}, "simulator depth stays flat at 14 for k >= 8", str(sorted(flat)))

    print("\nFinding 2 — k=32 is the peak, 16 -> 64 loses ~0.9 pp for 5x the CNOTs")
    acc16 = 100 * runs[("uniform", 16)]["acc_mean"]
    acc32 = 100 * runs[("uniform", 32)]["acc_mean"]
    acc64 = 100 * runs[("uniform", 64)]["acc_mean"]
    check(acc32 == max(acc16, acc32, acc64), "k=32 is the peak",
          f"16={acc16:.2f} 32={acc32:.2f} 64={acc64:.2f}")
    close(acc16 - acc64, 0.9, 0.1, "16 -> 64 accuracy loss", " pp")
    ratio = runs[("uniform", 64)]["stateprep_cnots_hw"] / runs[("uniform", 16)]["stateprep_cnots_hw"]
    check(4.5 <= ratio <= 5.5, "16 -> 64 costs ~5x the CNOTs", f"{ratio:.2f}x")

    print("\nFinding 3 — uniform beats decorrelated MI at every k >= 16")
    best_decorr = max(100 * runs[("decorr", k)]["acc_mean"] for k in SWEEP_TABLE)
    close(best_decorr, DECORR_BEST_PP, ACC_TOL_PP, "decorrelated MI tops out", "%")
    for k in (16, 32, 64):
        u = 100 * runs[("uniform", k)]["acc_mean"]
        d = 100 * runs[("decorr", k)]["acc_mean"]
        check(u > d, f"k={k}: uniform > decorr", f"uniform {u:.2f} vs decorr {d:.2f}")

    print("\nFinding 5 — the QNN's margin over logistic regression")
    for k, published in ((16, 2.1), (32, 1.8), (64, -0.04)):
        r = runs[("uniform", k)]
        margin = 100 * (r["acc_mean"] - r["classical_acc_mean"])
        close(margin, published, 0.1, f"k={k}: margin over LogReg", " pp")
    close(100 * meta["classical_acc_all_bands"], CLASSICAL_ALL_BANDS_PP, ACC_TOL_PP,
          "LogReg on all 103 bands", "%")

    print("\nCost correction to the source paper")
    close(meta["measured_sec_per_sample"], SEC_PER_SAMPLE, 0.05, "28.1 s/sample", " s")
    close(meta["measured_usd_per_sample"], USD_PER_SAMPLE, 0.01, "$45.01/sample", "")
    check(meta["qpu_rate_usd_per_min"] == QPU_RATE, "$96/min QPU rate",
          str(meta["qpu_rate_usd_per_min"]))
    full_m = FULL_TEST_SET * meta["measured_usd_per_sample"] / 1e6
    close(full_m, FULL_TEST_USD_M, 0.05, "full 31,256-sample test set ~= $1.4M", "M")

    print("\nFinding 6 — PennyLane and Qiskit agree analytically")
    a = sim["analytic"]
    check(a["abs_diff"] <= 2 * ANALYTIC_AGREEMENT, "analytic agreement ~5e-16",
          f"{a['abs_diff']:.2e}")
    close(a["pennylane"], -0.16704, 1e-4, "analytic <Z_0> = -0.16704", "")

    print("\nFinding 6 — sampled s.d. tracks the binomial 1/sqrt(N)")
    by_shots = {s["shots"]: s for s in sim["sweep"]}
    for shots in (64, 1024, 16384):
        s = by_shots[shots]
        # Both frameworks should sit within 30% of the binomial standard error.
        for fw in ("pl", "qk"):
            rel = abs(s[f"{fw}_std"] - s["theory_std"]) / s["theory_std"]
            check(rel < 0.30, f"{shots} shots: {fw} s.d. tracks theory",
                  f"{s[f'{fw}_std']:.4f} vs {s['theory_std']:.4f} ({rel:.0%} off)")
    # 4x the shots buys 2x the precision.
    scale = by_shots[1024]["theory_std"] / by_shots[16384]["theory_std"]
    check(3.5 <= scale <= 4.5, "16x shots -> 4x precision", f"{scale:.2f}x")

    print()
    if failures:
        print(f"{len(failures)} of {checks} checks FAILED:\n")
        for f in failures:
            print(f"  - {f}")
        print("\nEither the code regressed, or a claim legitimately moved — in which")
        print("case update README.md and the expected value in this file together.")
        print("Note: accuracy differences below ~1 pp are inside the run-to-run noise")
        print("floor (3000 train / 1500 test pixels, 3 seeds) and are not findings.")
        return 1

    print(f"All {checks} checks passed — README.md matches the data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
