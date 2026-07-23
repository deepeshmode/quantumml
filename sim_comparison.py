"""
Objective 2: why do PennyLane and Qiskit give different results?

Runs the *same* circuit - amplitude embedding of a real Pavia pixel, then three
SimplifiedTwoDesign layers - in both frameworks and compares <Z_0>.

The circuit is built natively in each rather than converted, so any difference
is a simulator difference and not a translation artifact. Qiskit is little-endian
(qubit 0 is the rightmost bit) where PennyLane is big-endian, so the state vector
and the measured qubit are reindexed accordingly.

Three estimators are compared:
  analytic  - exact <Z> from the state vector; no sampling
  sampled   - <Z> estimated from a finite number of measurement shots
  both      - PennyLane and Qiskit Aer, sampled identically

If the two frameworks agree within sampling error, the "difference" people
observe is shot noise, not a difference in underlying theory.
"""

import json
import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

from pipeline import load_pavia, binary_task, split, select_bands

N_QUBITS = 4
N_LAYERS = 3
SEED = 0


def std_pairs(n):
    """Wire pairs in the order SimplifiedTwoDesign applies them."""
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    return even + odd


def qiskit_circuit(state, init_w, weights, n=N_QUBITS):
    """SimplifiedTwoDesign in Qiskit, matching PennyLane's decomposition."""
    qc = QuantumCircuit(n)
    # PennyLane wire i == Qiskit qubit (n-1-i). Under that remap the two
    # frameworks index amplitudes identically, so the vector passes through
    # unchanged - reversing it here silently applies an X to every qubit and
    # produces a plausible but wrong expectation value.
    qc.initialize(np.asarray(state, dtype=complex).reshape(2 ** n), range(n))

    def q(i):
        return n - 1 - i

    for i in range(n):
        qc.ry(float(init_w[i]), q(i))
    for layer in weights:
        for p, (a, b) in enumerate(std_pairs(n)):
            qc.cz(q(a), q(b))
            qc.ry(float(layer[p][0]), q(a))
            qc.ry(float(layer[p][1]), q(b))
    return qc


def pennylane_expval(state, init_w, weights, shots=None, seed=None):
    dev = qml.device("default.qubit", wires=N_QUBITS, shots=shots, seed=seed)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(state, wires=range(N_QUBITS), normalize=True)
        qml.SimplifiedTwoDesign(initial_layer_weights=init_w, weights=weights,
                                wires=range(N_QUBITS))
        return qml.expval(qml.PauliZ(0))

    return float(circuit())


def qiskit_expval_analytic(state, init_w, weights):
    qc = qiskit_circuit(state, init_w, weights)
    sv = Statevector(qc)
    # PennyLane wire 0 -> Qiskit qubit n-1
    probs = sv.probabilities([N_QUBITS - 1])
    return float(probs[0] - probs[1])


def qiskit_expval_sampled(state, init_w, weights, shots, seed):
    qc = qiskit_circuit(state, init_w, weights)
    qc.measure_all()
    sim = AerSimulator(seed_simulator=seed)
    counts = sim.run(qc, shots=shots).result().get_counts()
    # Bitstring is little-endian: PennyLane wire 0 is the leftmost character.
    p0 = sum(c for b, c in counts.items() if b.replace(" ", "")[0] == "0")
    return (2 * p0 - shots) / shots


def main():
    rng = np.random.default_rng(SEED)

    # A real pixel from the scene, on the same 16 uniform bands the poster uses.
    cube, gt = load_pavia()
    X, y = binary_task(cube, gt)
    idx = select_bands(X, y, 2 ** N_QUBITS, "uniform")
    Xtr, ytr, _, _ = split(X[:, idx], y, seed=SEED)
    state = Xtr[0].astype(float)
    state = state / np.linalg.norm(state)

    shp = qml.SimplifiedTwoDesign.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    init_w = rng.normal(size=shp[0])
    weights = rng.normal(size=shp[1])

    pl_exact = pennylane_expval(state, init_w, weights)
    qk_exact = qiskit_expval_analytic(state, init_w, weights)

    print("=" * 66)
    print("ANALYTIC  (exact <Z_0>, no sampling)")
    print(f"  PennyLane {pl_exact:+.12f}")
    print(f"  Qiskit    {qk_exact:+.12f}")
    print(f"  |diff|     {abs(pl_exact - qk_exact):.2e}   "
          f"{'-> identical circuit' if abs(pl_exact-qk_exact) < 1e-9 else '-> MISMATCH'}")

    results = {"analytic": {"pennylane": pl_exact, "qiskit": qk_exact,
                            "abs_diff": abs(pl_exact - qk_exact)},
               "sweep": []}

    R = 40
    print()
    print("=" * 66)
    print(f"SAMPLED   ({R} independent repetitions per shot count)")
    print(f"{'shots':>7} {'PL mean':>10} {'PL sd':>8} {'Qk mean':>10} "
          f"{'Qk sd':>8} {'theory sd':>10}")

    for shots in [64, 256, 1024, 4096, 16384]:
        pl = [pennylane_expval(state, init_w, weights, shots=shots, seed=1000 + r)
              for r in range(R)]
        qk = [qiskit_expval_sampled(state, init_w, weights, shots, seed=2000 + r)
              for r in range(R)]
        # Binomial standard error on an expectation bounded in [-1, 1].
        theory = np.sqrt((1 - pl_exact ** 2) / shots)
        print(f"{shots:>7} {np.mean(pl):>+10.5f} {np.std(pl):>8.5f} "
              f"{np.mean(qk):>+10.5f} {np.std(qk):>8.5f} {theory:>10.5f}")
        results["sweep"].append({
            "shots": shots,
            "pl_mean": float(np.mean(pl)), "pl_std": float(np.std(pl)),
            "qk_mean": float(np.mean(qk)), "qk_std": float(np.std(qk)),
            "theory_std": float(theory),
            "pl_samples": [float(v) for v in pl],
            "qk_samples": [float(v) for v in qk],
        })

    results["exact"] = pl_exact
    with open("sim_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote sim_comparison.json")


if __name__ == "__main__":
    main()
