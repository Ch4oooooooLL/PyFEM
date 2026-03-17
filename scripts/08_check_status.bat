@echo off
chcp 65001 >nul

echo ============================================================
echo   FEM Project - Status Check
echo ============================================================
echo.

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

cd /d "%PROJECT_ROOT%"

python cli.py status

echo.
echo ============================================================
echo.
echo Press any key to exit...
pause >nul