$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"

if (-not (Test-Path -LiteralPath $venv)) {
    python -m venv --system-site-packages $venv
}

$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install setuptools==80.9.0
& $python -m pip install --no-deps --no-build-isolation -r (Join-Path $root "requirements-moge.lock.txt")
& $python -c "from moge.model.v2 import MoGeModel; import torch, utils3d; print('MoGe inference imports OK; torch=' + torch.__version__)"
