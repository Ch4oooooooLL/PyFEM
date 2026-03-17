#!/usr/bin/env pwsh
# FEM Project - Quick Test (1 Epoch Training)

param(
    [string]$Model = "gt"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FEM Project - Quick Test (1 Epoch Training)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

Write-Host "Model: $Model" -ForegroundColor Yellow
Write-Host "Epochs: 1 (quick test)" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path "dataset\train.npz")) {
    Write-Host "[ERROR] Dataset not found. Please run 02_generate_dataset first." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Running quick training test (1 epoch)..." -ForegroundColor Yellow
Write-Host ""

python cli.py train --model $Model --epochs 1 --batch_size 8

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Quick test completed!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[FAIL] Quick test failed" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"