$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CodexExe = "C:\Users\andre\AppData\Local\OpenAI\Codex\bin\070117a2efe12b41\codex.exe"
$CodexHome = "C:\Users\andre\.codex"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Project virtual environment not found at $PythonExe"
}

if (-not (Test-Path -LiteralPath $CodexExe)) {
    throw "Codex executable not found at $CodexExe"
}

$env:CODEX_HOME = $CodexHome

& $CodexExe mcp remove local_sam 2>$null
& $CodexExe mcp add local_sam --env "PYTHONPATH=$ProjectRoot" -- $PythonExe -m backend.mcp_server
& $CodexExe mcp get local_sam
