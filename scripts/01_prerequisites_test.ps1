#!/usr/bin/env pwsh
# FEM Project - Prerequisites Test

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FEM Project - Prerequisites Test" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

# [1/5] Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Python not found. Please install Python 3.9+" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# [2/5] Check pip
Write-Host ""
Write-Host "[2/5] Checking pip..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    Write-Host "[OK] pip available" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] pip not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# [3/5] Check required packages
Write-Host ""
Write-Host "[3/5] Checking required packages..." -ForegroundColor Yellow
$packages = @("numpy", "torch", "yaml", "rich", "tqdm")
$missing = @()

foreach ($pkg in $packages) {
    try {
        python -c "import $pkg" 2>$null
        Write-Host "[OK] $pkg" -ForegroundColor Green
    } catch {
        Write-Host "[MISSING] $pkg" -ForegroundColor Yellow
        $missing += $pkg
    }
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host "[INFO] Installing missing packages..." -ForegroundColor Yellow
    pip install numpy torch pyyaml rich tqdm
}

# [4/5] Check configuration files
Write-Host ""
Write-Host "[4/5] Checking configuration files..." -ForegroundColor Yellow
$configFiles = @("structure.yaml", "dataset_config.yaml", "condition_case.yaml")
foreach ($cfg in $configFiles) {
    if (Test-Path $cfg) {
        Write-Host "[OK] $cfg" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $cfg" -ForegroundColor Yellow
    }
}

# [5/5] Run smoke test
Write-Host ""
Write-Host "[5/5] Running smoke test (FEM static analysis)..." -ForegroundColor Yellow
$result = python PyFEM_Dynamics/main.py 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Smoke test passed" -ForegroundColor Green
} else {
    Write-Host "[FAIL] Smoke test failed" -ForegroundColor Red
    Write-Host "Run manually: python PyFEM_Dynamics/main.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Prerequisites check completed successfully!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"