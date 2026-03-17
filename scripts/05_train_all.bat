@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   FEM Project - Train All Models (GT + PINN)
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CONFIG_FILE=%1
set EPOCHS=%2

if "%CONFIG_FILE%"=="" set CONFIG_FILE=dataset_config.yaml
if "%EPOCHS%"=="" set EPOCHS=100

cd /d "%PROJECT_ROOT%"

echo Configuration: %CONFIG_FILE%
echo Epochs: %EPOCHS%
echo Models: GT + PINN
echo.

if not exist "dataset\train.npz" (
    echo [ERROR] Dataset not found. Please run 02_generate_dataset first.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo Starting combined model training...
echo.

python cli.py train --config "%CONFIG_FILE%" --model both --epochs %EPOCHS%

if errorlevel 1 (
    echo.
    echo [FAIL] Model training failed
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo   All models training completed!
echo ============================================================

if exist "Deep_learning\checkpoints\gt_best.pth" (
    echo GT Checkpoint: Deep_learning\checkpoints\gt_best.pth
)
if exist "Deep_learning\checkpoints\pinn_best.pth" (
    echo PINN Checkpoint: Deep_learning\checkpoints\pinn_best.pth
)

echo.
echo Press any key to exit...
pause >nul