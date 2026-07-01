$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"

if (-not (Test-Path -LiteralPath $venv)) {
    python -m venv --system-site-packages $venv
}

$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --no-deps -r (Join-Path $root "requirements-moge.lock.txt")
& $python -m pip check
