#!/usr/bin/env pwsh
# FEM Project - Generate Training Dataset

param(
    [string]$ConfigFile = "dataset_config.yaml",
    [string]$NJobs = "-1"
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FEM Project - Generate Training Dataset" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

Write-Host "Configuration: $ConfigFile" -ForegroundColor Yellow
Write-Host "Parallel jobs: $NJobs" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $ConfigFile)) {
    Write-Host "[ERROR] Configuration file not found: $ConfigFile" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Starting dataset generation..." -ForegroundColor Yellow
Write-Host ""

python cli.py dataset --config $ConfigFile -j $NJobs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Dataset generation completed successfully!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    
    if (Test-Path "dataset\train.npz") {
        Write-Host "Output: dataset\train.npz" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "[FAIL] Dataset generation failed" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"