# Start SmarterFencing locally on Windows (same /demo UI + pipeline as production v262).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:WORKSPACE_ROOT = if ($env:WORKSPACE_ROOT) { $env:WORKSPACE_ROOT } else { Join-Path $Root "local_workspace" }
$env:LOCAL_WEBAPP_PORT = if ($env:LOCAL_WEBAPP_PORT) { $env:LOCAL_WEBAPP_PORT } else { "5000" }
Set-Location $Root

$Python = if ($env:PYTHON_BIN) {
    $env:PYTHON_BIN
} elseif ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    "python"
}

& $Python run_local_webapp.py
