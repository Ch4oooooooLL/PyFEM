@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ============================================================
echo   FEM Project - Full Pipeline Execution
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

cd /d "%PROJECT_ROOT%"

set SKIP_STATIC=
set SKIP_DATASET=
set SKIP_TRAIN=
set SKIP_PREDICT=

:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--skip-static" set SKIP_STATIC=--skip-static
if /i "%~1"=="--skip-dataset" set SKIP_DATASET=--skip-dataset
if /i "%~1"=="--skip-train" set SKIP_TRAIN=--skip-train
if /i "%~1"=="--skip-predict" set SKIP_PREDICT=--skip-predict
shift
goto :parse_args
:end_parse

echo Pipeline Options:
if "%SKIP_STATIC%"=="" (echo   [x] Static Analysis) else (echo   [ ] Static Analysis ^(skipped^))
if "%SKIP_DATASET%"=="" (echo   [x] Dataset Generation) else (echo   [ ] Dataset Generation ^(skipped^))
if "%SKIP_TRAIN%"=="" (echo   [x] Model Training) else (echo   [ ] Model Training ^(skipped^))
if "%SKIP_PREDICT%"=="" (echo   [x] Condition Prediction) else (echo   [ ] Condition Prediction ^(skipped^))
echo.

echo Starting full pipeline execution...
echo.

python cli.py pipeline %SKIP_STATIC% %SKIP_DATASET% %SKIP_TRAIN% %SKIP_PREDICT%

if errorlevel 1 (
    echo.
    echo [FAIL] Pipeline execution failed
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo.
echo ============================================================
echo   Full pipeline completed successfully!
echo ============================================================
echo.
echo Press any key to exit...
pause >nul