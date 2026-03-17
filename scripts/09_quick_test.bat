@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   FEM Project - Quick Test (1 Epoch Training)
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set MODEL=%1

if "%MODEL%"=="" set MODEL=gt

cd /d "%PROJECT_ROOT%"

echo Model: %MODEL%
echo Epochs: 1 (quick test)
echo.

if not exist "dataset\train.npz" (
    echo [ERROR] Dataset not found. Please run 02_generate_dataset first.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo Running quick training test (1 epoch)...
echo.

python cli.py train --model %MODEL% --epochs 1 --batch_size 8

if errorlevel 1 (
    echo.
    echo [FAIL] Quick test failed
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo   Quick test completed!
echo ============================================================
echo.
echo Press any key to exit...
pause >nul