"""
The SimplifiedTwoDesign discrepancy in Tomev/qnn_change_detection.

That repo's README documents an unresolved issue: its PennyLane and Qiskit
implementations of the same model give different outputs, and the authors could
not explain why. This reproduces the cause standalone (no dependency on their
repo): their hand-written Qiskit circuit is not the same circuit as
pennylane.SimplifiedTwoDesign.

Two defects in their SimplifiedTwoDesignQiskit, for N qubits, L layers:

  1. Missing initial rotation layer. PennyLane opens with RY on every wire from
     initial_layer_weights; their version omits it (N gates gone).
  2. Wrong even-block RY targets. PennyLane applies RY to both qubits of each
     even pair; their loop over range(N-1) skips the last wire in the even block.

Net for N=4, L=2: 18 gates vs PennyLane's 22. Either alone changes the computed
function. On genuinely identical circuits the two frameworks agree to ~5e-16
(see sim_comparison.py) - so the discrepancy is a circuit bug, not a simulator
or endianness difference.
"""

import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

N, L = 4, 2


def their_simplified_two_design_qiskit(n_qubits, n_layers):
    """Verbatim reconstruction of SimplifiedTwoDesignQiskit from the repo."""
    weights = []
    for i in range(n_layers):
        weights.append((ParameterVector(f"w_{i}_even", n_qubits - 1),
                        ParameterVector(f"w_{i}_odd", n_qubits - 1)))
    wires = range(n_qubits)
    c = QuantumCircuit(n_qubits)
    even = [wires[i:i + 2] for i in range(0, len(wires) - 1, 2)]
    odd = [wires[i:i + 2] for i in range(1, len(wires) - 1, 2)]
    for lw in weights:
        for p in even:
            c.cz(p[0], p[1])
        for q in range(n_qubits - 1):          # <-- misses the last even-block wire
            c.ry(lw[0][q], q)
        for p in odd:
            c.cz(p[0], p[1])
        for q in range(n_qubits - 1):
            c.ry(lw[1][q], q + 1)
    return c                                    # <-- no initial RY layer at all


def pennylane_ops():
    shp = qml.SimplifiedTwoDesign.shape(n_layers=L, n_wires=N)
    init = np.round(np.arange(1, N + 1) * 0.1, 3)
    w = np.round(np.arange(1, np.prod(shp[1]) + 1).reshape(shp[1]) * 0.01, 3)
    return qml.SimplifiedTwoDesign(
        initial_layer_weights=init, weights=w, wires=range(N)).decomposition()


def main():
    pl = pennylane_ops()
    qc = their_simplified_two_design_qiskit(N, L)

    print("--- PennyLane SimplifiedTwoDesign ---")
    for o in pl:
        p = f" {float(o.parameters[0]):.3f}" if o.parameters else ""
        print(f"    {o.name:3s} {list(o.wires)}{p}")

    print("\n--- their SimplifiedTwoDesignQiskit ---")
    for inst in qc.data:
        q = [qc.find_bit(b).index for b in inst.qubits]
        p = f" {inst.operation.params[0]}" if inst.operation.params else ""
        print(f"    {inst.operation.name:3s} {q}{p}")

    pl_initial_ry = sum(1 for o in pl[:next(i for i, o in enumerate(pl)
                                            if o.name == "CZ")] if o.name == "RY")
    qk_initial_ry = 0
    for inst in qc.data:
        if inst.operation.name == "cz":
            break
        if inst.operation.name == "ry":
            qk_initial_ry += 1

    print("\n=== findings ===")
    print(f"initial RY layer   pennylane: {pl_initial_ry}   qiskit: {qk_initial_ry}"
          f"   {'MISMATCH' if pl_initial_ry != qk_initial_ry else 'ok'}")
    print(f"total gates        pennylane: {len(pl)}   qiskit: {len(qc.data)}"
          f"   {'MISMATCH' if len(pl) != len(qc.data) else 'ok'}")


if __name__ == "__main__":
    main()
