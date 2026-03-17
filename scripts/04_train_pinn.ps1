#!/usr/bin/env pwsh
# FEM Project - Train PINN Model

param(
    [string]$ConfigFile = "dataset_config.yaml",
    [int]$Epochs = 100,
    [int]$BatchSize = 32,
    [double]$Lr = 0.001,
    [string]$Device = "auto"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FEM Project - Train PINN Model" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

Write-Host "Configuration: $ConfigFile" -ForegroundColor Yellow
Write-Host "Epochs: $Epochs" -ForegroundColor Yellow
Write-Host "Batch size: $BatchSize" -ForegroundColor Yellow
Write-Host "Learning rate: $Lr" -ForegroundColor Yellow
Write-Host "Device: $Device" -ForegroundColor Yellow
Write-Host "Model: PINN (Physics-Informed Neural Network)" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path "dataset\train.npz")) {
    Write-Host "[ERROR] Dataset not found. Please run 02_generate_dataset first." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting PINN model training..." -ForegroundColor Yellow
Write-Host ""

python cli.py train --config $ConfigFile --model pinn --epochs $Epochs --batch_size $BatchSize --lr $Lr --device $Device

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  PINN model training completed!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    
    if (Test-Path "Deep_learning\checkpoints\pinn_best.pth") {
        Write-Host "Checkpoint: Deep_learning\checkpoints\pinn_best.pth" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "[FAIL] PINN model training failed" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"