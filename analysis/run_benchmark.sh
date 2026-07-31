#!/usr/bin/env bash
# Set up the pinned PennyLane stack and run the simulator backend benchmark.
#
# This is the path that actually produces GPU numbers: pennylane-lightning-gpu
# publishes Linux wheels only, so native Windows cannot run it. Under WSL2,
# CUDA works through the Windows host driver — you do not install a driver in
# the guest.
#
#   wsl --install -d Ubuntu       # once, on Windows, then reboot
#   wsl
#   cd /mnt/c/path/to/quantumml/analysis
#   bash run_benchmark.sh
#
# Needs Python 3.11 — the pinned stack has no wheels for 3.12+.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo
echo "=== PennyLane backend benchmark — setup ==="

# --- CUDA visible? ----------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader |
        sed 's/^/  GPU: /'
else
    echo "  nvidia-smi not found — this will be a CPU-only run."
    echo "  Under WSL2 the Windows driver provides it; if it is missing,"
    echo "  update the NVIDIA driver on the Windows host (not inside WSL)."
fi

# --- locate Python 3.11 -----------------------------------------------------
PY=""
for c in python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" --version 2>&1 | grep -q "3\.11"; then
        PY="$c"; break
    fi
done
if [ -z "$PY" ]; then
    echo
    echo "  Python 3.11 not found. The pinned stack has no wheels for 3.12+."
    echo "  On Ubuntu/WSL2:"
    echo "      sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update"
    echo "      sudo apt install -y python3.11 python3.11-venv"
    exit 1
fi
echo "  python: $($PY --version)"

# --- virtual environment ----------------------------------------------------
VENV=".venv-gpu"
[ -d "$VENV" ] || { echo "  creating venv at $VENV"; "$PY" -m venv "$VENV"; }
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip

# --- pinned CPU stack -------------------------------------------------------
# Every pin is load-bearing. autoray especially: >=0.6.0 permits 0.8.x, which
# removed the NumpyMimic symbol PennyLane 0.38 imports at module load, so the
# lower bound alone reproduces the exact failure it was meant to fix.
echo
echo "  installing pinned stack..."
if ! "$VPY" -m pip install --quiet \
        "numpy==1.26.4" "scipy==1.13.1" "autoray==0.6.12" \
        "pennylane==0.38.1" "pennylane-lightning==0.38.0"; then
    echo "  pinned install FAILED"; exit 1
fi
echo "  pinned stack installed"

# --- CUDA backend -----------------------------------------------------------
echo
echo "  installing CUDA backend..."
if "$VPY" -m pip install --quiet "pennylane-lightning-gpu==0.38.0" "custatevec-cu12"; then
    echo "  CUDA backend installed"
else
    echo "  CUDA backend install failed — continuing CPU-only."
    echo "  Common causes: no CUDA toolkit, or a CUDA 11 machine (use custatevec-cu11)."
fi

# --- verify -----------------------------------------------------------------
echo
echo "=== backend check ==="
"$VPY" - <<'PYEOF'
import warnings; warnings.filterwarnings("ignore")
import pennylane as qml, numpy
print("  pennylane", qml.__version__, "| numpy", numpy.__version__)
import autoray; print("  autoray", autoray.__version__, "(must be 0.6.12)")
for n in ["default.qubit", "lightning.qubit", "lightning.gpu"]:
    try:
        qml.device(n, wires=2); print("   OK  ", n)
    except Exception as e:
        print("   n/a ", n, "|", type(e).__name__)
PYEOF
[ $? -eq 0 ] || { echo "  import failed — the pins did not take effect."; exit 1; }

# --- run --------------------------------------------------------------------
if [ ! -f gpu_benchmark_colab.py ]; then
    echo "  fetching benchmark script..."
    curl -fsSL -o gpu_benchmark_colab.py \
      https://raw.githubusercontent.com/deepeshmode/quantumml/main/analysis/gpu_benchmark_colab.py
fi

echo
echo "=== running benchmark ==="
echo "  expect several minutes; slow backends retire automatically"
echo
"$VPY" gpu_benchmark_colab.py

echo
echo "=== done ==="
ls -1 backend_benchmark_*.json 2>/dev/null | sed 's/^/  wrote /'
echo "  send that JSON back to have it merged into fig6."
echo
