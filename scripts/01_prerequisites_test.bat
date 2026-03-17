@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   FEM Project - Prerequisites Test
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

cd /d "%PROJECT_ROOT%"

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python version: %PYVER%

echo.
echo [2/5] Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] pip not found
    pause
    exit /b 1
)
echo [OK] pip available

echo.
echo [3/5] Checking required packages...
set MISSING=0

python -c "import numpy" 2>nul
if errorlevel 1 (
    echo [MISSING] numpy
    set MISSING=1
) else (
    echo [OK] numpy
)

python -c "import torch" 2>nul
if errorlevel 1 (
    echo [MISSING] torch
    set MISSING=1
) else (
    echo [OK] torch
)

python -c "import yaml" 2>nul
if errorlevel 1 (
    echo [MISSING] pyyaml
    set MISSING=1
) else (
    echo [OK] pyyaml
)

python -c "import rich" 2>nul
if errorlevel 1 (
    echo [MISSING] rich
    set MISSING=1
) else (
    echo [OK] rich
)

python -c "import tqdm" 2>nul
if errorlevel 1 (
    echo [MISSING] tqdm
    set MISSING=1
) else (
    echo [OK] tqdm
)

if %MISSING%==1 (
    echo.
    echo [INFO] Installing missing packages...
    pip install numpy torch pyyaml rich tqdm
)

echo.
echo [4/5] Checking configuration files...
if exist "structure.yaml" (
    echo [OK] structure.yaml
) else (
    echo [MISSING] structure.yaml
    set MISSING=1
)

if exist "dataset_config.yaml" (
    echo [OK] dataset_config.yaml
) else (
    echo [MISSING] dataset_config.yaml
    set MISSING=1
)

if exist "condition_case.yaml" (
    echo [OK] condition_case.yaml
) else (
    echo [MISSING] condition_case.yaml
    set MISSING=1
)

echo.
echo [5/5] Running smoke test (FEM static analysis)...
python PyFEM_Dynamics/main.py >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Smoke test failed
    echo Run manually: python PyFEM_Dynamics/main.py
    echo.
    echo ============================================================
    echo   Press any key to exit...
    echo ============================================================
    pause >nul
    exit /b 1
)
echo [OK] Smoke test passed

echo.
echo ============================================================
echo   Prerequisites check completed successfully!
echo ============================================================
echo.
echo Press any key to exit...
pause >nul