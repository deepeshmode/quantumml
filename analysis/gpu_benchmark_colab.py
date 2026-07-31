"""GPU vs CPU simulator benchmark — run on Colab with a GPU runtime.

Produces the numbers behind the "does GPU acceleration help?" question for this
project: per-circuit wall-clock across qubit counts on every available backend,
plus adjoint-differentiation timing, which is what training actually costs.

HOW TO RUN
----------
1. Colab -> Runtime -> Change runtime type -> T4 GPU (or A100).
2. Paste the INSTALL cell below into a cell and run it.
3. Upload this file (or paste it into a cell) and run it.
4. Download `gpu_benchmark.json` and hand it back — it merges with the CPU run.

INSTALL CELL (run this first, then restart the runtime):

    !nvidia-smi
    !pip install -q pennylane==0.38.1 pennylane-lightning==0.38.0 \
                    pennylane-lightning-gpu==0.38.0 custatevec-cu12

Notes
-----
* `lightning.gpu` needs CUDA and cuQuantum; it will simply not appear in the
  backend list on a CPU runtime, and the script still runs.
* Memory is the binding constraint at high wire counts: a statevector is
  2**n complex128 = 16 * 2**n bytes. A T4 (16 GB) tops out near 29 wires,
  an A100 80 GB near 32. The script stops a backend once it OOMs rather
  than crashing.
* Batches are capped so the broadcast state stays bounded — per-circuit
  microseconds is the comparable number, not wall-clock per call.
"""
import json
import platform
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

# Sweep well past this project's 8 wires so the crossover is visible.
WIRE_COUNTS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
N_LAYERS = 3
REPEATS = 3
MAX_BROADCAST_AMPS = 2 ** 22          # cap broadcast state at ~67 MB
# Retire a backend once a single measurement exceeds this. Without it the pure
# Python simulator would spend hours past ~22 wires while the GPU is the point.
GIVE_UP_SECONDS = 25.0
CANDIDATES = ["default.qubit", "lightning.qubit", "lightning.gpu",
              "lightning.kokkos", "qiskit.aer"]

# This project's operating points, marked on the output for context.
PROJECT_POINTS = {
    8:  "pixel-level QNN (2 x 4 PCA components) and 2x2-patch QCNN",
    18: "3x3-patch QCNN",
    32: "4x4-patch QCNN — 68 GB, out of reach on any single GPU",
}


def batch_for(n_wires):
    return max(1, min(256, MAX_BROADCAST_AMPS // (2 ** n_wires)))


def statevector_bytes(n_wires):
    return 16 * 2 ** n_wires


def available_backends():
    found, missing = [], {}
    for name in CANDIDATES:
        try:
            qml.device(name, wires=2)
            found.append(name)
        except Exception as e:
            missing[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    return found, missing


def make_qnode(backend, n_wires, diff_method=None):
    """The project's actual circuit: AngleEmbedding + SimplifiedTwoDesign."""
    dev = qml.device(backend, wires=n_wires)
    shape = qml.SimplifiedTwoDesign.shape(n_layers=N_LAYERS, n_wires=n_wires)
    kwargs = {"diff_method": diff_method} if diff_method else {}

    @qml.qnode(dev, **kwargs)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_wires), rotation="Y")
        qml.SimplifiedTwoDesign(
            initial_layer_weights=pnp.zeros(n_wires),
            weights=weights,
            wires=range(n_wires),
        )
        return qml.expval(qml.PauliZ(0))

    return circuit, shape


def bench_forward(backend, n_wires):
    batch = batch_for(n_wires)
    circuit, shape = make_qnode(backend, n_wires)
    x = pnp.array(np.random.uniform(0, np.pi, (batch, n_wires)))
    w = pnp.array(np.random.uniform(0, np.pi, shape[1]))
    circuit(x[0], w)                                       # warm up
    best = min(_timed(circuit, x, w) for _ in range(REPEATS))
    return best, batch


def bench_gradient(backend, n_wires):
    """Adjoint differentiation — this is what a training step actually costs."""
    circuit, shape = make_qnode(backend, n_wires, diff_method="adjoint")
    x = pnp.array(np.random.uniform(0, np.pi, n_wires))
    w = pnp.array(np.random.uniform(0, np.pi, shape[1]), requires_grad=True)
    grad = qml.grad(circuit, argnum=1)
    grad(x, w)                                             # warm up
    return min(_timed(grad, x, w) for _ in range(REPEATS))


def _timed(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def main():
    found, missing = available_backends()
    gpu_present = "lightning.gpu" in found

    print(f"platform     : {platform.platform()}  ({platform.machine()})")
    print(f"pennylane    : {qml.__version__}")
    print(f"backends     : {found}")
    for k, v in missing.items():
        print(f"  unavailable  {k:<18} {v}")
    if not gpu_present:
        print("\n!! lightning.gpu NOT available — this is a CPU-only run.")
        print("   Colab: Runtime > Change runtime type > T4 GPU, then rerun the install cell.\n")

    rows = []
    print(f"\nForward pass — per-circuit microseconds "
          f"(SimplifiedTwoDesign, {N_LAYERS} layers, best of {REPEATS})")
    head = "  wires    state  " + "".join(f"{b:>18}" for b in found)
    print(head + "\n  " + "-" * (len(head) - 2))

    dead = set()
    for n in WIRE_COUNTS:
        cells = []
        for b in found:
            if b in dead:
                cells.append(f"{'—':>18}")
                continue
            try:
                t, batch = bench_forward(b, n)
                us = 1e6 * t / batch
                rows.append({"kind": "forward", "backend": b, "wires": n,
                             "batch": batch, "seconds": t, "per_circuit_us": us})
                cells.append(f"{us:>15.1f}us")
                if t > GIVE_UP_SECONDS:
                    dead.add(b)                            # too slow to continue
            except Exception as e:
                dead.add(b)                                # OOM / unsupported
                rows.append({"kind": "forward", "backend": b, "wires": n,
                             "error": f"{type(e).__name__}"})
                cells.append(f"{type(e).__name__[:16]:>18}")
        sv = statevector_bytes(n)
        sv_s = f"{sv/1024**2:.0f}MB" if sv >= 1024**2 else f"{sv/1024:.0f}KB"
        print(f"  {n:>5}  {sv_s:>7}  " + "".join(cells))

    print(f"\nGradient (adjoint) — milliseconds per parameter-shift-free gradient")
    head = "  wires  " + "".join(f"{b:>18}" for b in found)
    print(head + "\n  " + "-" * (len(head) - 2))
    dead = set()
    for n in WIRE_COUNTS:
        cells = []
        for b in found:
            if b in dead:
                cells.append(f"{'—':>18}")
                continue
            try:
                t = bench_gradient(b, n)
                rows.append({"kind": "gradient", "backend": b, "wires": n,
                             "seconds": t})
                cells.append(f"{1e3*t:>15.2f}ms")
                if t > GIVE_UP_SECONDS:
                    dead.add(b)
            except Exception as e:
                dead.add(b)
                rows.append({"kind": "gradient", "backend": b, "wires": n,
                             "error": f"{type(e).__name__}"})
                cells.append(f"{type(e).__name__[:16]:>18}")
        print(f"  {n:>5}  " + "".join(cells))

    # Crossover: smallest wire count where lightning.gpu beats the best CPU backend.
    crossover = None
    if gpu_present:
        cpu = [b for b in found if b != "lightning.gpu"]
        for n in WIRE_COUNTS:
            g = next((r["per_circuit_us"] for r in rows
                      if r.get("kind") == "forward" and r["backend"] == "lightning.gpu"
                      and r["wires"] == n and "per_circuit_us" in r), None)
            c = [r["per_circuit_us"] for r in rows
                 if r.get("kind") == "forward" and r["backend"] in cpu
                 and r["wires"] == n and "per_circuit_us" in r]
            if g is not None and c and g < min(c):
                crossover = n
                break
        print(f"\nGPU overtakes the fastest CPU backend at: "
              f"{crossover if crossover else 'no crossover in the swept range'} wires")

    print("\nThis project's operating points:")
    for n, label in PROJECT_POINTS.items():
        sv = statevector_bytes(n)
        sv_s = (f"{sv/1024**3:.1f}GB" if sv >= 1024**3
                else f"{sv/1024**2:.0f}MB" if sv >= 1024**2 else f"{sv/1024:.0f}KB")
        print(f"  {n:>2} wires  {sv_s:>7}   {label}")

    out = {
        "platform": platform.platform(), "machine": platform.machine(),
        "pennylane": qml.__version__, "backends": found, "unavailable": missing,
        "n_layers": N_LAYERS, "crossover_wires": crossover,
        "gpu_present": gpu_present, "results": rows,
    }
    try:                                                   # record the GPU model
        import subprocess
        out["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        out["nvidia_smi"] = None

    with open("gpu_benchmark.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote gpu_benchmark.json — download this and hand it back")


if __name__ == "__main__":
    main()
