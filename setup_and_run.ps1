$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "FarmSignal AI - setup and launch" -ForegroundColor Green

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) { $pythonCmd = "py" } else { $pythonCmd = "python" }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $pythonCmd -m venv .venv
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r requirements.txt

Write-Host "Launching FarmSignal AI dashboard..." -ForegroundColor Cyan
& $venvPython -m streamlit run src/dashboard.py
