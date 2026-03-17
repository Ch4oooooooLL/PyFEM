@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   FEM Project - Train Graph Transformer (GT) Model
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CONFIG_FILE=%1
set EPOCHS=%2
set BATCH_SIZE=%3
set LR=%4
set DEVICE=%5

if "%CONFIG_FILE%"=="" set CONFIG_FILE=dataset_config.yaml
if "%EPOCHS%"=="" set EPOCHS=100
if "%BATCH_SIZE%"=="" set BATCH_SIZE=32
if "%LR%"=="" set LR=0.001
if "%DEVICE%"=="" set DEVICE=auto

cd /d "%PROJECT_ROOT%"

echo Configuration: %CONFIG_FILE%
echo Epochs: %EPOCHS%
echo Batch size: %BATCH_SIZE%
echo Learning rate: %LR%
echo Device: %DEVICE%
echo Model: GT (Graph Transformer)
echo.

if not exist "dataset\train.npz" (
    echo [ERROR] Dataset not found. Please run 02_generate_dataset first.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo Starting GT model training...
echo.

python cli.py train --config "%CONFIG_FILE%" --model gt --epochs %EPOCHS% --batch_size %BATCH_SIZE% --lr %LR% --device %DEVICE%

if errorlevel 1 (
    echo.
    echo [FAIL] GT model training failed
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo   GT model training completed!
echo ============================================================

if exist "Deep_learning\checkpoints\gt_best.pth" (
    echo Checkpoint: Deep_learning\checkpoints\gt_best.pth
)

echo.
echo Press any key to exit...
pause >nul