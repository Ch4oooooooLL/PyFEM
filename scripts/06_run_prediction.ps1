#!/usr/bin/env pwsh
# FEM Project - Run Condition Prediction

param(
    [string]$ConfigFile = "condition_case.yaml",
    [string]$OutputDir = ""
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FEM Project - Run Condition Prediction" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

Write-Host "Configuration: $ConfigFile" -ForegroundColor Yellow
if ($OutputDir -ne "") {
    Write-Host "Output directory: $OutputDir" -ForegroundColor Yellow
}
Write-Host ""

if (-not (Test-Path $ConfigFile)) {
    Write-Host "[ERROR] Configuration file not found: $ConfigFile" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "Deep_learning\checkpoints\gt_best.pth")) {
    Write-Host "[WARNING] GT checkpoint not found. Run 03_train_gt first." -ForegroundColor Yellow
}

Write-Host "Starting condition prediction..." -ForegroundColor Yellow
Write-Host ""

if ($OutputDir -eq "") {
    python cli.py predict --config $ConfigFile
} else {
    python cli.py predict --config $ConfigFile --output-dir $OutputDir
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Condition prediction completed!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[FAIL] Condition prediction failed" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"