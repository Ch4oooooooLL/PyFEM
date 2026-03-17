#!/usr/bin/env pwsh
# FEM Project - Full Pipeline Execution

param(
    [switch]$SkipStatic,
    [switch]$SkipDataset,
    [switch]$SkipTrain,
    [switch]$SkipPredict
)

Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "  FEM Project - Full Pipeline Execution" -ForegroundColor Magenta
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

Write-Host "Pipeline Options:" -ForegroundColor Yellow
if (-not $SkipStatic) {
    Write-Host "  [x] Static Analysis" -ForegroundColor Green
} else {
    Write-Host "  [ ] Static Analysis (skipped)" -ForegroundColor Gray
}

if (-not $SkipDataset) {
    Write-Host "  [x] Dataset Generation" -ForegroundColor Green
} else {
    Write-Host "  [ ] Dataset Generation (skipped)" -ForegroundColor Gray
}

if (-not $SkipTrain) {
    Write-Host "  [x] Model Training" -ForegroundColor Green
} else {
    Write-Host "  [ ] Model Training (skipped)" -ForegroundColor Gray
}

if (-not $SkipPredict) {
    Write-Host "  [x] Condition Prediction" -ForegroundColor Green
} else {
    Write-Host "  [ ] Condition Prediction (skipped)" -ForegroundColor Gray
}
Write-Host ""

Write-Host "Starting full pipeline execution..." -ForegroundColor Yellow
Write-Host ""

$arguments = @("pipeline")

if ($SkipStatic) { $arguments += "--skip-static" }
if ($SkipDataset) { $arguments += "--skip-dataset" }
if ($SkipTrain) { $arguments += "--skip-train" }
if ($SkipPredict) { $arguments += "--skip-predict" }

python cli.py @arguments

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Full pipeline completed successfully!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[FAIL] Pipeline execution failed" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to exit"