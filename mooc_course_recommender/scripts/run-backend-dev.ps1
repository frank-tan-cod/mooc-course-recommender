$ErrorActionPreference = "Stop"

Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)

if (-not $env:CORS_ALLOW_ORIGINS) {
    $env:CORS_ALLOW_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
}

$python = Join-Path (Split-Path -Parent (Get-Location)) ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

& $python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
