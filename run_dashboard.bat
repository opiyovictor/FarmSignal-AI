@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo       FarmSignal AI - Kenya Dashboard
echo ==============================================

where py >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON=py"
) else (
    set "PYTHON=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :error
)

set "VENV_PY=.venv\Scripts\python.exe"

echo [2/3] Installing dashboard dependencies...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/3] Launching FarmSignal AI...
"%VENV_PY%" -m streamlit run src\dashboard.py
if errorlevel 1 goto :error
goto :end

:error
echo.
echo FarmSignal AI could not start. Review the error above.
pause
exit /b 1

:end
endlocal
