@echo off
chcp 65001 >nul
setlocal

set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"

set "TARGET_ENV=%FEM_CONDA_ENV%"
if not defined TARGET_ENV (
    if defined CONDA_DEFAULT_ENV (
        if /i not "%CONDA_DEFAULT_ENV%"=="base" (
            set "TARGET_ENV=%CONDA_DEFAULT_ENV%"
        )
    )
)
if not defined TARGET_ENV set "TARGET_ENV=fem"

set "CONDA_BAT="
for /f "delims=" %%i in ('where conda.bat 2^>nul') do (
    set "CONDA_BAT=%%i"
    goto :conda_found
)
for /f "delims=" %%i in ('where conda 2^>nul') do (
    set "CONDA_BAT=%%i"
    goto :conda_found
)

:conda_found
if not defined CONDA_BAT (
    echo [ERROR] Conda not found. Install Conda and ensure conda is on PATH.
    pause
    exit /b 1
)

call "%CONDA_BAT%" run -n base python -m tools.cli bootstrap --env-name "%TARGET_ENV%"
if errorlevel 1 (
    pause
    exit /b %errorlevel%
)

call "%CONDA_BAT%" activate "%TARGET_ENV%"
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment %TARGET_ENV%.
    pause
    exit /b %errorlevel%
)

python -m tools.cli %*
exit /b %errorlevel%
