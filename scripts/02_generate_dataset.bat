@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   FEM Project - Generate Training Dataset
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set CONFIG_FILE=%1
set N_JOBS=%2

if "%CONFIG_FILE%"=="" set CONFIG_FILE=dataset_config.yaml
if "%N_JOBS%"=="" set N_JOBS=-1

cd /d "%PROJECT_ROOT%"

echo Configuration: %CONFIG_FILE%
echo Parallel jobs: %N_JOBS%
echo.

if not exist "%CONFIG_FILE%" (
    echo [ERROR] Configuration file not found: %CONFIG_FILE%
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo Starting dataset generation...
echo.

python cli.py dataset --config "%CONFIG_FILE%" -j %N_JOBS%

if errorlevel 1 (
    echo.
    echo [FAIL] Dataset generation failed
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo   Dataset generation completed successfully!
echo ============================================================

if exist "dataset\train.npz" (
    echo Output: dataset\train.npz
)

echo.
echo Press any key to exit...
pause >nul