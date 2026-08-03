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

INSTALL CELL (run this, then RESTART THE RUNTIME before anything else):

    !nvidia-smi
    !pip install -q "numpy==1.26.4" "scipy==1.13.1" "autoray==0.6.12" \
                    pennylane==0.38.1 pennylane-lightning==0.38.0 \
                    pennylane-lightning-gpu==0.38.0 custatevec-cu12

None of these three pins is optional, and each fails differently:

* numpy — PennyLane 0.38.1 predates numpy 2.x while Colab ships numpy 2:

      ValueError: numpy.dtype size changed, may indicate binary
      incompatibility. Expected 96 from C header, got 88 from PyObject

* scipy — Colab's stock build targets numpy 2 and breaks if numpy is
  downgraded on its own, so it must move together with numpy.

* autoray — 0.8.x removed the symbol PennyLane 0.38 imports:

      AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'

  This is the same pin the training environment carries; see CLAUDE.md.

Restarting the runtime after installing is equally mandatory: pip cannot swap
a numpy the kernel has already imported.

If the pinned stack cannot be made to work, running the newest PennyLane
instead is acceptable — record the version in the output and the CPU baseline
will be re-run to match, since cross-version timings are not comparable.

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

# Operating points marked on the output. See wire_budget.py for the arithmetic:
# angle embedding costs T*V*F wires, amplitude embedding ceil(log2(T*V*F)).
PROJECT_POINTS = {
    8:  "current pixel-level QNN — 2 dates x 4 PCA components (angle)",
    14: "4D: 8^3 voxels x 10 bands x 2 dates (amplitude)",
    18: "3x3-patch QCNN (angle)  |  4D: 16^3 voxels x 13 bands x 4 dates (amplitude)",
    20: "1x1 px x 10 bands, PCA removed (angle) — fixes the per-date basis mismatch",
    21: "4D: 32^3 voxels x 13 bands x 4 dates (amplitude)",
    24: "4D: 64^3 voxels x 13 bands x 4 dates (amplitude)",
    26: "1x1 px x all 13 bands (angle)",
    28: "4D: 128^3 voxels x 13 bands x 8 dates (amplitude)",
    32: "2x2 patch x 4 PCA (angle) — 68 GB, beyond any single GPU",
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
    """Time a forward pass, broadcasting a batch where the backend supports it.

    PennyLane 0.45's default.qubit raises `ValueError: shape-mismatch for sum`
    for any broadcast batch above 11 wires, so fall back to single-circuit
    timing rather than losing the whole column. Per-circuit microseconds stays
    the comparable quantity either way; `mode` records which path was taken.
    """
    circuit, shape = make_qnode(backend, n_wires)
    w = pnp.array(np.random.uniform(0, np.pi, shape[1]))
    batch = batch_for(n_wires)

    if batch > 1:
        x = pnp.array(np.random.uniform(0, np.pi, (batch, n_wires)))
        try:
            circuit(x[0], w)                               # warm up
            best = min(_timed(circuit, x, w) for _ in range(REPEATS))
            return best, batch, "broadcast"
        except Exception:
            pass                                           # fall through

    x1 = pnp.array(np.random.uniform(0, np.pi, n_wires))
    circuit(x1, w)                                         # warm up
    best = min(_timed(circuit, x1, w) for _ in range(REPEATS))
    return best, 1, "single"


def bench_gradient(backend, n_wires):
    """Adjoint differentiation — this is what a training step actually costs.

    The qnode takes weights only, with the input angles baked in as
    constants: qml.grad's argnum kwarg was removed in PennyLane 0.4x, and a
    single-argument qnode sidesteps the difference entirely, so this runs
    unchanged on 0.38 and 0.45+."""
    dev = qml.device(backend, wires=n_wires)
    shape = qml.SimplifiedTwoDesign.shape(n_layers=N_LAYERS, n_wires=n_wires)
    x = np.random.uniform(0, np.pi, n_wires)

    @qml.qnode(dev, diff_method="adjoint")
    def circuit(weights):
        qml.AngleEmbedding(x, wires=range(n_wires), rotation="Y")
        qml.SimplifiedTwoDesign(
            initial_layer_weights=pnp.zeros(n_wires),
            weights=weights,
            wires=range(n_wires),
        )
        return qml.expval(qml.PauliZ(0))

    w = pnp.array(np.random.uniform(0, np.pi, shape[1]), requires_grad=True)
    grad = qml.grad(circuit)
    grad(w)                                                # warm up
    return min(_timed(grad, w) for _ in range(REPEATS))


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
    print("  * = backend rejected a broadcast batch; timed one circuit at a time")
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
                t, batch, mode = bench_forward(b, n)
                us = 1e6 * t / batch
                rows.append({"kind": "forward", "backend": b, "wires": n,
                             "batch": batch, "mode": mode, "seconds": t,
                             "per_circuit_us": us})
                cells.append(f"{us:>14.1f}us{'*' if mode == 'single' else ' '}")
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

    tag = "gpu" if gpu_present else platform.machine()
    name = f"backend_benchmark_{tag}.json"
    with open(name, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {name} — download this and hand it back")


if __name__ == "__main__":
    main()
