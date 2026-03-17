#!/usr/bin/env pwsh
# FEM Project - Train All Models (GT + PINN)

param(
    [string]$ConfigFile = "dataset_config.yaml",
    [int]$Epochs = 100
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FEM Project - Train All Models (GT + PINN)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

Write-Host "Configuration: $ConfigFile" -ForegroundColor Yellow
Write-Host "Epochs: $Epochs" -ForegroundColor Yellow
Write-Host "Models: GT + PINN" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path "dataset\train.npz")) {
    Write-Host "[ERROR] Dataset not found. Please run 02_generate_dataset first." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting combined model training..." -ForegroundColor Yellow
Write-Host ""

python cli.py train --config $ConfigFile --model both --epochs $Epochs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  All models training completed!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    
    if (Test-Path "Deep_learning\checkpoints\gt_best.pth") {
        Write-Host "GT Checkpoint: Deep_learning\checkpoints\gt_best.pth" -ForegroundColor Yellow
    }
    if (Test-Path "Deep_learning\checkpoints\pinn_best.pth") {
        Write-Host "PINN Checkpoint: Deep_learning\checkpoints\pinn_best.pth" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "[FAIL] Model training failed" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"