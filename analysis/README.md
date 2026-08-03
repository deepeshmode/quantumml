# analysis/

Figures and measurements behind [`../ANSWERS.md`](../ANSWERS.md). Every number
the answers assert is in `metrics.json` or `backend_benchmark*.json`.

| File | What it shows |
|---|---|
| `fig1_training_dynamics.png` | Train/held-out accuracy and loss over 20 epochs; the 9-epoch plateau and the epoch-18 peak |
| `fig2_change_maps.png` | Full-resolution predicted probability, ground truth, and TP/FP/FN error map per held-out city |
| `fig3_threshold.png` | Precision/recall/F1 against decision threshold, and the precision–recall curve |
| `fig4_per_city.png` | Per-city metric breakdown |
| `fig5_separability.png` | Whether ‖PCA(t₂) − PCA(t₁)‖ separates change from no-change *before* the model sees it |
| `fig6_backend_crossover.png` | Simulator cost against circuit width, one line per backend |
| `metrics.json` | Full-resolution confusion matrices, per city and aggregate |
| `backend_benchmark.json` | Backend timings, PennyLane 0.38.1, this project's pinned stack |
| `backend_benchmark_pl045.json` | Same sweep on PennyLane 0.45.1, for reference |
| `fig7_xai.png` | Exact SHAP / IG attributions and Fourier structure of the trained model (`xai_oscd.py`) |
| `fig8_gpu_t4.png` | The measured GPU crossover on a Colab T4 (`plot_gpu_t4.py`) |
| `xai_results.json` | Attribution numbers, symmetry decomposition, 4D xAI cost table |
| `experiment_4d_results.json` | 4D experiment: both arms, count baseline, T4 benchmark |
| `REPRODUCTION.md` | Corrected reproduction write-up (mirrored from the sibling checkout) |

## Regenerating the OSCD analysis

Runs the trained model over every pixel of the three held-out cities. Needs the
sibling `qnn_change_detection` checkout, its Python 3.11 venv, the OSCD dataset,
and the MLflow run artifact.

```bash
cd ~/Downloads/qnn_change_detection
./.venv/bin/python analyse_oscd.py
```

Inference is cached to `.inference_cache.npz` after the first run, so figure
tweaks are instant. Delete that file to force re-inference. Expect ~4–5 minutes
cold: 2.5 M pixels at ~9,700 px/s.

**Batch size matters more than anything else here.** Throughput is ~9,700 px/s
at batch 1000 and collapses to ~650 px/s at batch 20,000 — a 15× penalty purely
from broadcasting too wide.

## Running the backend benchmark

`gpu_benchmark_colab.py` sweeps circuit width across every simulator available
and reports per-circuit microseconds plus adjoint-gradient timing. It runs
unchanged everywhere; backends that are missing simply do not appear, and slow
or out-of-memory backends retire mid-sweep instead of crashing.

Two wrappers set up the environment for you:

```powershell
# Windows PowerShell
powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1
```

```bash
# Linux, WSL2, macOS
bash run_benchmark.sh
```

### The CUDA backend needs Linux

`pennylane-lightning-gpu` publishes Linux (manylinux) wheels only. **Native
Windows cannot run it regardless of the GPU present** — `nvidia-smi` working
does not change this. `run_benchmark.ps1` detects the case, says so, and
completes a CPU-only run rather than failing obscurely.

For real GPU numbers, use WSL2. CUDA works there through the Windows host
driver; you do not install a driver inside the guest.

```powershell
wsl --install -d Ubuntu     # once, then reboot
wsl
```

### The pins are load-bearing

PennyLane 0.38.1 needs all three, and each fails differently:

| pin | failure without it |
|---|---|
| `numpy==1.26.4` | `ValueError: numpy.dtype size changed … Expected 96 from C header, got 88` |
| `scipy==1.13.1` | breaks when numpy is downgraded on its own |
| `autoray==0.6.12` | `AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'` |

**`autoray>=0.6.0` does not work.** It is a lower bound only, so pip installs
0.8.x — exactly the releases that removed the symbol PennyLane 0.38 imports at
module load. Use `==0.6.12` or `<0.7`. After installing, restart the
interpreter: pip cannot swap a numpy that has already been imported.

Python 3.11 is required; the pinned stack has no wheels for 3.12+.

If the pinned stack proves impossible on a given machine, current PennyLane
(0.45+) is numpy-2 native and needs no pins. Record the version — timings are
not comparable across PennyLane versions, so the CPU baseline has to be re-run
to match.

## Claude Code on WSL

Run inside the WSL terminal, not PowerShell:

```bash
curl -fsSL https://claude.ai/install.sh | bash
claude --version
claude doctor          # read-only diagnostics
```

Authenticate by running `claude` and following the browser prompt. Requires a
Pro, Max, Team, Enterprise, or Console account.

If the installer fails, the apt repository is the fallback:

```bash
sudo apt install curl gnupg
sudo install -d -m 0755 /etc/apt/keyrings
sudo curl -fsSL https://downloads.claude.ai/keys/claude-code.asc \
  -o /etc/apt/keyrings/claude-code.asc
echo "deb [signed-by=/etc/apt/keyrings/claude-code.asc] https://downloads.claude.ai/claude-code/apt/stable stable main" \
  | sudo tee /etc/apt/sources.list.d/claude-code.list
sudo apt update && sudo apt install claude-code
```

**Clone into the WSL filesystem, not `/mnt/c`.** Cross-filesystem access from
WSL2 to the Windows drive is slow enough to hurt both Claude Code and the
benchmark.

```bash
cd ~ && git clone https://github.com/deepeshmode/quantumml.git
cd quantumml/analysis && bash run_benchmark.sh
```

WSL2 is also the only Windows option that supports Claude Code's sandboxing.

With Claude Code installed you can skip the wrapper scripts and ask it directly
to set up the stack and run the benchmark — `../CLAUDE.md` already documents the
autoray trap, so it has the context. The scripts remain as a record of what
worked.
