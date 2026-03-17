#!/bin/bash
set -e

echo "============================================================"
echo "  FEM Project - Prerequisites Test"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[FAIL] Python not found. Please install Python 3.9+"
        read -p "Press Enter to exit..."
        exit 1
    fi
    PYTHON_CMD=python
else
    PYTHON_CMD=python3
fi

PYVER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo "[OK] Python version: $PYVER"

echo ""
echo "[2/5] Checking pip..."
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "[FAIL] pip not found"
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] pip available"

echo ""
echo "[3/5] Checking required packages..."
MISSING=0

for pkg in numpy torch yaml rich tqdm; do
    if $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        echo "[OK] $pkg"
    else
        echo "[MISSING] $pkg"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "[INFO] Installing missing packages..."
    $PYTHON_CMD -m pip install numpy torch pyyaml rich tqdm
fi

echo ""
echo "[4/5] Checking configuration files..."
for cfg in structure.yaml dataset_config.yaml condition_case.yaml; do
    if [ -f "$cfg" ]; then
        echo "[OK] $cfg"
    else
        echo "[MISSING] $cfg"
        MISSING=1
    fi
done

echo ""
echo "[5/5] Running smoke test (FEM static analysis)..."
if ! $PYTHON_CMD PyFEM_Dynamics/main.py &> /dev/null; then
    echo "[FAIL] Smoke test failed"
    echo "Run manually: python PyFEM_Dynamics/main.py"
    echo ""
    echo "============================================================"
    read -p "Press Enter to exit..."
    exit 1
fi
echo "[OK] Smoke test passed"

echo ""
echo "============================================================"
echo "  Prerequisites check completed successfully!"
echo "============================================================"
echo ""
read -p "Press Enter to exit..."