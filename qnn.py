"""
Hybrid quantum-classical classifier, following the architecture in
Rybotycki, Gupta & Gawron (arXiv:2503.08962v3):

    classical perceptron -> amplitude embedding -> 3x SimplifiedTwoDesign
    -> <Z> on wire 0 -> sigmoid

The band count k fixes the qubit count: amplitude embedding packs k values into
log2(k) qubits, so band selection is what sets circuit width and state-prep
depth. That is the link between the preprocessing stage and what the circuit
costs to run.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def n_qubits_for(k):
    """Amplitude embedding packs k amplitudes into log2(k) qubits."""
    n = int(np.ceil(np.log2(k)))
    if 2 ** n != k:
        raise ValueError(f"k must be a power of 2, got {k}")
    return n


def build_qnode(n_qubits, n_layers=3, shots=None):
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, interface="torch", diff_method="backprop" if shots is None else "parameter-shift")
    def circuit(inputs, initial_layer_weights, weights):
        qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
        qml.SimplifiedTwoDesign(
            initial_layer_weights=initial_layer_weights,
            weights=weights,
            wires=range(n_qubits),
        )
        return qml.expval(qml.PauliZ(0))

    shapes = {
        "initial_layer_weights": (n_qubits,),
        "weights": qml.SimplifiedTwoDesign.shape(n_layers=n_layers, n_wires=n_qubits)[1],
    }
    return circuit, shapes


class HybridModel(nn.Module):
    def __init__(self, k, n_layers=3, shots=None):
        super().__init__()
        self.n_qubits = n_qubits_for(k)
        self.pre = nn.Linear(k, 2 ** self.n_qubits)
        circuit, shapes = build_qnode(self.n_qubits, n_layers, shots)
        self.q = qml.qnn.TorchLayer(circuit, shapes)

    def forward(self, x):
        z = self.q(self.pre(x))
        return torch.sigmoid(z.reshape(-1))


def circuit_stats(k, n_layers=3):
    """Measured gate count and depth for the circuit at this band count."""
    n = n_qubits_for(k)
    circuit, shapes = build_qnode(n, n_layers)
    rng = np.random.default_rng(0)
    args = (
        torch.tensor(rng.normal(size=2 ** n), dtype=torch.float64),
        torch.tensor(rng.normal(size=shapes["initial_layer_weights"]), dtype=torch.float64),
        torch.tensor(rng.normal(size=shapes["weights"]), dtype=torch.float64),
    )
    # Templates count as one op each unless we expand to the device gate set.
    spec = qml.specs(circuit, level="device")(*args)
    res = spec["resources"]
    n_params = int(np.prod(shapes["initial_layer_weights"]) + np.prod(shapes["weights"]))
    return {
        "k": k, "n_qubits": n, "depth": int(res.depth),
        "n_gates": int(res.num_gates),
        "quantum_params": n_params,
        "classical_params": k * (2 ** n) + 2 ** n,
    }


def train_eval(Xtr, ytr, Xte, yte, k, epochs=15, batch=64, lr=0.05, seed=0, n_layers=3):
    """Train on the k selected bands, return test accuracy."""
    torch.manual_seed(seed)
    model = HybridModel(k, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)

    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = ((model(Xte_t) > 0.5).numpy().astype(int) == yte).mean()
        history.append(float(acc))

    return float(history[-1]), history, model
