<#
.SYNOPSIS
    Set up the pinned PennyLane stack and run the simulator backend benchmark.

.DESCRIPTION
    Creates a virtual environment, installs the pinned stack that PennyLane
    0.38.1 requires, attempts the CUDA backend, and runs the benchmark.

    IMPORTANT: pennylane-lightning-gpu publishes Linux wheels only. On native
    Windows the CUDA backend cannot be installed no matter what hardware is
    present. This script detects that and tells you, rather than failing
    obscurely. For real GPU numbers use WSL2 and run_benchmark.sh instead:

        wsl --install -d Ubuntu          # once, then reboot
        wsl
        cd /mnt/c/path/to/quantumml/analysis
        bash run_benchmark.sh

    NVIDIA supports CUDA inside WSL2 with the standard Windows driver — you do
    not install a separate driver in the WSL guest.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1

.NOTES
    Needs Python 3.11. 3.12+ has no wheels for this pinned stack.
#>

$ErrorActionPreference = "Stop"
$here = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location $here

Write-Host "`n=== PennyLane backend benchmark — setup ===" -ForegroundColor Cyan

# --- locate Python 3.11 -----------------------------------------------------
$py = $null
foreach ($candidate in @(@("py", "-3.11"), @("python3.11"), @("python"))) {
    try {
        $exe = $candidate[0]
        $argv = if ($candidate.Count -gt 1) { $candidate[1..($candidate.Count - 1)] } else { @() }
        $ver = & $exe @argv --version 2>&1
        if ($ver -match "3\.11") { $py = $candidate; Write-Host "  python: $ver"; break }
    } catch { }
}
if (-not $py) {
    Write-Host "`nPython 3.11 not found." -ForegroundColor Red
    Write-Host "The pinned stack (PennyLane 0.38.1 / numpy 1.26.4) has no wheels for 3.12+."
    Write-Host "Install 3.11 from https://www.python.org/downloads/release/python-3119/"
    Write-Host "and re-run. Tick 'Add python.exe to PATH' in the installer."
    exit 1
}

# --- virtual environment ----------------------------------------------------
$venv = Join-Path $here ".venv-gpu"
if (-not (Test-Path $venv)) {
    Write-Host "  creating venv at $venv"
    $exe = $py[0]
    $argv = if ($py.Count -gt 1) { $py[1..($py.Count - 1)] } else { @() }
    & $exe @argv -m venv $venv
}
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "venv creation failed" -ForegroundColor Red; exit 1 }

& $vpy -m pip install --quiet --upgrade pip

# --- pinned CPU stack -------------------------------------------------------
# Each pin is load-bearing; see gpu_benchmark_colab.py for the failure each one
# prevents. autoray in particular MUST be capped: >=0.6.0 permits 0.8.x, which
# removes the NumpyMimic symbol PennyLane 0.38 imports at module load.
Write-Host "`n  installing pinned stack (this takes a minute)..."
& $vpy -m pip install --quiet `
    "numpy==1.26.4" "scipy==1.13.1" "autoray==0.6.12" `
    "pennylane==0.38.1" "pennylane-lightning==0.38.0"
if ($LASTEXITCODE -ne 0) { Write-Host "  pinned install FAILED" -ForegroundColor Red; exit 1 }
Write-Host "  pinned stack installed" -ForegroundColor Green

# --- CUDA backend (expected to fail on native Windows) ----------------------
Write-Host "`n  attempting CUDA backend..."
$ErrorActionPreference = "Continue"
& $vpy -m pip install --quiet "pennylane-lightning-gpu==0.38.0" "custatevec-cu12" 2>&1 |
    Out-String | Write-Host
$gpuInstalled = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"

if (-not $gpuInstalled) {
    Write-Host @"

  CUDA backend unavailable on native Windows.

  pennylane-lightning-gpu publishes Linux (manylinux) wheels only. This is a
  packaging limitation, not a driver or hardware problem — nvidia-smi working
  on this machine does not change it.

  To get GPU numbers, run the Linux script under WSL2:

      wsl --install -d Ubuntu     (once, then reboot)
      wsl
      cd /mnt/c/path/to/quantumml/analysis
      bash run_benchmark.sh

  Continuing with a CPU-only run, which is still worth having — it gives the
  default.qubit vs lightning.qubit crossover on this machine.

"@ -ForegroundColor Yellow
}

# --- verify -----------------------------------------------------------------
Write-Host "`n=== backend check ===" -ForegroundColor Cyan
& $vpy -c @"
import warnings; warnings.filterwarnings('ignore')
import pennylane as qml, numpy
print('  pennylane', qml.__version__, '| numpy', numpy.__version__)
for n in ['default.qubit', 'lightning.qubit', 'lightning.gpu']:
    try:
        qml.device(n, wires=2); print('   OK  ', n)
    except Exception as e:
        print('   n/a ', n, '|', type(e).__name__)
"@
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n  import failed — the pins did not take effect." -ForegroundColor Red
    Write-Host "  Check: $vpy -c `"import autoray; print(autoray.__version__)`""
    Write-Host "  It must be 0.6.12. If it is not, something upgraded it."
    exit 1
}

# --- run --------------------------------------------------------------------
$script = Join-Path $here "gpu_benchmark_colab.py"
if (-not (Test-Path $script)) {
    Write-Host "`n  fetching benchmark script..."
    Invoke-WebRequest -UseBasicParsing `
        -Uri "https://raw.githubusercontent.com/deepeshmode/quantumml/main/analysis/gpu_benchmark_colab.py" `
        -OutFile $script
}

Write-Host "`n=== running benchmark ===" -ForegroundColor Cyan
Write-Host "  expect several minutes; slow backends retire automatically`n"
& $vpy $script

Write-Host "`n=== done ===" -ForegroundColor Cyan
Get-ChildItem -Path $here -Filter "backend_benchmark_*.json" |
    ForEach-Object { Write-Host "  wrote $($_.Name)" -ForegroundColor Green }
Write-Host "  send that JSON back to have it merged into fig6.`n"
