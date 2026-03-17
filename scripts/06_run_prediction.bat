@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   FEM Project - Run Condition Prediction
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CONFIG_FILE=%1
set OUTPUT_DIR=%2

if "%CONFIG_FILE%"=="" set CONFIG_FILE=condition_case.yaml

cd /d "%PROJECT_ROOT%"

echo Configuration: %CONFIG_FILE%
if not "%OUTPUT_DIR%"=="" echo Output directory: %OUTPUT_DIR%
echo.

if not exist "%CONFIG_FILE%" (
    echo [ERROR] Configuration file not found: %CONFIG_FILE%
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

if not exist "Deep_learning\checkpoints\gt_best.pth" (
    echo [WARNING] GT checkpoint not found. Run 03_train_gt first.
)

echo Starting condition prediction...
echo.

if "%OUTPUT_DIR%"=="" (
    python cli.py predict --config "%CONFIG_FILE%"
) else (
    python cli.py predict --config "%CONFIG_FILE%" --output-dir "%OUTPUT_DIR%"
)

if errorlevel 1 (
    echo.
    echo [FAIL] Condition prediction failed
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo   Condition prediction completed!
echo ============================================================
echo.
echo Press any key to exit...
pause >nul