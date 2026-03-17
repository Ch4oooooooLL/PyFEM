#!/bin/bash
set -e

echo "============================================================"
echo "  FEM Project - Quick Test (1 Epoch Training)"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL="${1:-gt}"

cd "$PROJECT_ROOT"

echo "Model: $MODEL"
echo "Epochs: 1 (quick test)"
echo ""

if [ ! -f "dataset/train.npz" ]; then
    echo "[ERROR] Dataset not found. Please run 02_generate_dataset first."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Running quick training test (1 epoch)..."
echo ""

python cli.py train --model "$MODEL" --epochs 1 --batch_size 8

echo ""
echo "============================================================"
echo "  Quick test completed!"
echo "============================================================"
echo ""
read -p "Press Enter to exit..."