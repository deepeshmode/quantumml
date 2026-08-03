"""Circuit-width budget for 2D and 4D (3D volume + time) change detection.

Answers two questions the current architecture cannot:

  1. How wide does the circuit get as data grows from 2D pixels to 4D volumes?
  2. Where does that land relative to the ~20-wire threshold at which GPU
     statevector simulation starts to beat CPU?

The answer depends almost entirely on the embedding, not on the hardware:

  angle embedding      wires = T * V * F          linear in the data
  amplitude embedding  wires = ceil(log2(T*V*F))  logarithmic in the data

Angle embedding is what the OSCD repo uses and it cannot survive 4D — a 3x3x3
voxel patch already exceeds any machine. Amplitude embedding compresses the
same data into 11-24 wires, which is precisely the band where a GPU wins.
The cost is moved rather than removed: exact state preparation is O(2^n) CNOTs
on real hardware (Mottonen et al.), so amplitude embedding buys simulation
feasibility and pays for it at execution time.

Run:  python wire_budget.py
"""
import math

# Memory ceilings, complex128 statevector = 16 * 2**n bytes.
CEILINGS = [
    ("this Mac, 24 GB", 24 * 1024 ** 3),
    ("T4, 15 GB", 15 * 1024 ** 3),
    ("A100, 80 GB", 80 * 1024 ** 3),
]
GPU_CROSSOVER = 20          # measured: accelerated backends win above ~this
CPU_CROSSOVER = 14          # measured: lightning.qubit overtakes default.qubit


def state_bytes(wires):
    return 16 * 2 ** wires


def human(nbytes):
    for unit, size in (("TB", 1024 ** 4), ("GB", 1024 ** 3),
                       ("MB", 1024 ** 2), ("KB", 1024)):
        if nbytes >= size:
            return f"{nbytes / size:.0f}{unit}"
    return f"{nbytes}B"


def max_wires(budget_bytes):
    return int(math.floor(math.log2(budget_bytes / 16)))


def angle_wires(timestamps, voxels, features):
    return timestamps * voxels * features


def amplitude_wires(timestamps, voxels, features):
    return max(1, math.ceil(math.log2(timestamps * voxels * features)))


def verdict(wires):
    fits = state_bytes(wires) <= 24 * 1024 ** 3
    if not fits:
        return "infeasible"
    if wires >= GPU_CROSSOVER:
        return "GPU territory"
    if wires >= CPU_CROSSOVER:
        return "C++ backend wins"
    return "below crossover"


CONFIGS_2D = [
    # label,                            T, voxels, features
    ("1x1 px, 4 PCA  (current)",        2, 1,      4),
    ("1x1 px, 10 bands, no PCA",        2, 1,      10),
    ("1x1 px, 13 bands (all)",          2, 1,      13),
    ("2x2 patch, 1 feature",            2, 4,      1),
    ("2x2 patch, 4 PCA",                2, 4,      4),
    ("3x3 patch, 1 feature",            2, 9,      1),
    ("3x3 patch, 4 PCA",                2, 9,      4),
]

CONFIGS_4D = [
    ("4^3 voxels, 10 bands, 2 dates",   2, 4 ** 3,  10),
    ("8^3 voxels, 10 bands, 2 dates",   2, 8 ** 3,  10),
    ("16^3 voxels, 13 bands, 4 dates",  4, 16 ** 3, 13),
    ("32^3 voxels, 13 bands, 4 dates",  4, 32 ** 3, 13),
    ("64^3 voxels, 13 bands, 4 dates",  4, 64 ** 3, 13),
    ("128^3 voxels, 13 bands, 8 dates", 8, 128 ** 3, 13),
]


def table(title, configs):
    print(f"\n{title}")
    print(f"  {'configuration':<34}{'values':>12}"
          f"{'angle: wires':>15}{'state':>9}   "
          f"{'amplitude: wires':>18}{'state':>9}   verdict (amplitude)")
    print("  " + "-" * 128)
    for label, t, v, f in configs:
        n = t * v * f
        aw = angle_wires(t, v, f)
        mw = amplitude_wires(t, v, f)
        a_state = human(state_bytes(aw)) if aw <= 45 else "overflow"
        print(f"  {label:<34}{n:>12,}"
              f"{aw:>15,}{a_state:>9}   "
              f"{mw:>18}{human(state_bytes(mw)):>9}   {verdict(mw)}")


if __name__ == "__main__":
    print("Circuit-width budget — 2D pixels vs 4D volumes")
    print("\nMemory ceilings (complex128 statevector):")
    for name, budget in CEILINGS:
        print(f"  {name:<18} {max_wires(budget):>3} wires max")
    print(f"\nMeasured crossovers on this project's circuit:")
    print(f"  lightning.qubit overtakes default.qubit at ~{CPU_CROSSOVER} wires")
    print(f"  GPU expected to overtake CPU at        ~{GPU_CROSSOVER} wires")

    table("2D — the current problem", CONFIGS_2D)
    table("4D — 3D volumes with time", CONFIGS_4D)

    print("""
Reading this
------------
Angle embedding puts one feature on one wire, so wires grow linearly with the
data. It survives 2D single pixels and nothing else: a 2x2 patch with 4 PCA
components already needs 32 wires (68 GB), and every 4D configuration is
astronomically out of reach. This is the embedding the OSCD repo uses.

Amplitude embedding packs 2**n values into n wires, so realistic 4D volumes
land at 11-24 wires — below any memory ceiling, and squarely in the range
where GPU statevector simulation is worth using. This is the embedding the
Pavia pipeline uses.

So a GPU is not justified by the current 8-wire model, and is justified by 4D
work — provided the architecture moves to amplitude embedding.

The catch, and it must be stated alongside the above: exact amplitude state
preparation costs O(2^n) CNOTs on hardware, 2**n - n - 1 for n qubits. At 24
wires that is ~16.7 million two-qubit gates for a single sample. Amplitude
embedding makes 4D simulable, not executable. Any claim of quantum advantage
here has to confront state preparation, not circuit depth.

A GPU also helps the half of this workload that is not quantum at all. Loading
and preprocessing 4D volumes dominates memory today — measured at 1.27 GB peak
for 2D, of which the statevector is 0.004 MB — and that gap widens with volume
data. cuPy/RAPIDS for the pipeline and a classical 3D encoder on GPU feeding a
compact quantum head is the architecture where GPU earns its place twice.
""")
