"""
Precompute everything the monitor demo animates, so the loop itself does no
quantum work and cannot stall in front of an audience.

Writes demo_data.npz:
  maps      - prediction map per band count
  accs      - test accuracy per band count
  cnots     - modeled hardware state-prep CNOTs per band count
  spectrum  - the single pixel used in the shot-noise scene
  samples   - repeated shot-based estimates of <Z_0> for that pixel
  exact     - the analytic value they converge on
"""

import json
import numpy as np
import torch

from pipeline import load_pavia, binary_task, split, select_bands, wavelengths, BUILT_CLASSES, VEG_CLASSES
from qnn import train_eval
from run_experiment import hardware_stateprep_cnots

K_VALUES = [4, 8, 16, 32, 64]
SEED = 0


def main():
    cube, gt = load_pavia()
    X, y = binary_task(cube, gt)
    labeled = np.isin(gt, BUILT_CLASSES + VEG_CLASSES)

    maps, accs, cnots, qubits, bands = [], [], [], [], []
    for k in K_VALUES:
        idx = select_bands(X, y, k, "uniform")
        Xtr, ytr, Xte, yte = split(X[:, idx], y, seed=SEED)
        acc, _, model = train_eval(Xtr, ytr, Xte, yte, k, epochs=15, seed=SEED)

        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xall = (cube[labeled][:, idx] - mu) / sd
        with torch.no_grad():
            p = model(torch.tensor(Xall, dtype=torch.float32)).numpy()
        pred = np.full(gt.shape, -1, dtype=np.int8)
        pred[labeled] = (p > 0.5).astype(np.int8)

        n = int(np.log2(k))
        maps.append(pred); accs.append(acc)
        cnots.append(hardware_stateprep_cnots(n)); qubits.append(n)
        bands.append(wavelengths(X.shape[1])[idx])
        print(f"k={k:3d}  n={n}  acc={acc:.4f}  cnots={cnots[-1]}")

    sim = json.load(open("sim_comparison.json"))
    row = next(s for s in sim["sweep"] if s["shots"] == 1024)

    # The pixel the shot-noise scene uses, in full spectral resolution.
    idx16 = select_bands(X, y, 16, "uniform")
    Xtr16, _, _, _ = split(X[:, idx16], y, seed=SEED)

    np.savez_compressed(
        "demo_data.npz",
        maps=np.stack(maps), accs=np.array(accs), cnots=np.array(cnots),
        qubits=np.array(qubits), ks=np.array(K_VALUES),
        bands16=bands[2], spectrum=Xtr16[0],
        samples=np.array(row["pl_samples"]), exact=float(sim["exact"]),
        shots=int(row["shots"]),
        rgb_wl=wavelengths(X.shape[1]),
    )
    print("wrote demo_data.npz")


if __name__ == "__main__":
    main()
