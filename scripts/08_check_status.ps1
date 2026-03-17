#!/usr/bin/env pwsh
# FEM Project - Status Check

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  FEM Project - Status Check" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_ROOT = Split-Path -Parent $SCRIPT_DIR

Set-Location $PROJECT_ROOT

python cli.py status

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"