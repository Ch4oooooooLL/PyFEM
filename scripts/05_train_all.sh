#!/bin/bash
set -e

echo "============================================================"
echo "  FEM Project - Train All Models (GT + PINN)"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${1:-dataset_config.yaml}"
EPOCHS="${2:-100}"

cd "$PROJECT_ROOT"

echo "Configuration: $CONFIG_FILE"
echo "Epochs: $EPOCHS"
echo "Models: GT + PINN"
echo ""

if [ ! -f "dataset/train.npz" ]; then
    echo "[ERROR] Dataset not found. Please run 02_generate_dataset first."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Starting combined model training..."
echo ""

python cli.py train --config "$CONFIG_FILE" --model both --epochs "$EPOCHS"

echo ""
echo "============================================================"
echo "  All models training completed!"
echo "============================================================"

if [ -f "Deep_learning/checkpoints/gt_best.pth" ]; then
    echo "GT Checkpoint: Deep_learning/checkpoints/gt_best.pth"
fi
if [ -f "Deep_learning/checkpoints/pinn_best.pth" ]; then
    echo "PINN Checkpoint: Deep_learning/checkpoints/pinn_best.pth"
fi

echo ""
read -p "Press Enter to exit..."