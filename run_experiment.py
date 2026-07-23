"""
Band count sweep: how much of the hyperspectral cube does the QNN actually need?

Sweeps k over powers of two, so each k maps to log2(k) qubits. Records test
accuracy, measured simulator circuit resources, and a modeled hardware
state-preparation cost. Results land in results.json for the figure script.
"""

import json
import time
import numpy as np

from pipeline import (
    load_pavia, binary_task, split, wavelengths,
    select_bands, select_bands_decorrelated,
)
from qnn import train_eval, circuit_stats

K_VALUES = [4, 8, 16, 32, 64]
SEEDS = [0, 1, 2]
EPOCHS = 15

# Measured on ibm_brisbane by Rybotycki et al. (Zenodo 14784888):
# 1815 samples in 851 min of QPU time at $96/min pay-as-you-go.
# NOTE: the paper reports "2 min 13 s" per sample, but its own spreadsheet
# computes samples-per-minute (1815/851 = 2.13) and the label is the
# reciprocal. 851/1815 = 28.1 s/sample is what their totals actually support.
QPU_RATE_USD_PER_MIN = 96.0
MEASURED_SEC_PER_SAMPLE = 851 * 60 / 1815
MEASURED_USD_PER_SAMPLE = 851 * QPU_RATE_USD_PER_MIN / 1815


def hardware_stateprep_cnots(n_qubits):
    """
    Modeled CNOT count for exact amplitude embedding on hardware.

    default.qubit applies StatePrep directly, so the simulator depth below
    hides this entirely. On a real device an arbitrary n-qubit state costs
    O(2^n) CNOTs (Mottonen et al. 2005; lower bound 2^n - n - 1). This is the
    term band selection actually controls.
    """
    return 2 ** n_qubits - n_qubits - 1


def classical_baseline(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def main():
    cube, gt = load_pavia()
    X, y = binary_task(cube, gt)
    wl = wavelengths(X.shape[1])
    print(f"labeled pixels {X.shape}, built fraction {y.mean():.3f}")

    results = {
        "meta": {
            "dataset": "Pavia University (ROSIS, 103 bands, 430-860 nm)",
            "task": "built/disturbed surface vs undisturbed vegetation",
            "n_labeled": int(len(y)),
            "epochs": EPOCHS,
            "seeds": SEEDS,
            "measured_sec_per_sample": MEASURED_SEC_PER_SAMPLE,
            "measured_usd_per_sample": MEASURED_USD_PER_SAMPLE,
            "qpu_rate_usd_per_min": QPU_RATE_USD_PER_MIN,
        },
        "runs": [],
        "bands": {},
    }

    for method in ["decorr", "uniform"]:
        for k in K_VALUES:
            if method == "uniform":
                idx = select_bands(X, y, k, "uniform")
            else:
                idx = select_bands_decorrelated(X, y, k, "mi")
            results["bands"][f"{method}_{k}"] = {
                "idx": idx.tolist(), "nm": wl[idx].round(1).tolist(),
            }

            accs, clf_accs = [], []
            t0 = time.time()
            for seed in SEEDS:
                Xtr, ytr, Xte, yte = split(X[:, idx], y, seed=seed)
                acc, _, _ = train_eval(Xtr, ytr, Xte, yte, k, epochs=EPOCHS, seed=seed)
                accs.append(acc)
                clf_accs.append(classical_baseline(Xtr, ytr, Xte, yte))

            stats = circuit_stats(k)
            stats["stateprep_cnots_hw"] = hardware_stateprep_cnots(stats["n_qubits"])
            results["runs"].append({
                "method": method, "k": k,
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "accs": accs,
                "classical_acc_mean": float(np.mean(clf_accs)),
                **stats,
            })
            print(f"{method:8s} k={k:3d} n={stats['n_qubits']} "
                  f"qnn={np.mean(accs):.4f}+/-{np.std(accs):.4f} "
                  f"logreg={np.mean(clf_accs):.4f} "
                  f"gates={stats['n_gates']} hw_cnots={stats['stateprep_cnots_hw']} "
                  f"({time.time()-t0:.0f}s)")

    # Full 103-band reference point: what you give up by reducing at all.
    Xtr, ytr, Xte, yte = split(X, y, seed=0)
    results["meta"]["classical_acc_all_bands"] = classical_baseline(Xtr, ytr, Xte, yte)
    print(f"logreg on all 103 bands: {results['meta']['classical_acc_all_bands']:.4f}")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json")


if __name__ == "__main__":
    main()
