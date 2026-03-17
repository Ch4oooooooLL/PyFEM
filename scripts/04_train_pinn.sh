#!/bin/bash
set -e

echo "============================================================"
echo "  FEM Project - Train PINN Model"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${1:-dataset_config.yaml}"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-32}"
LR="${4:-0.001}"
DEVICE="${5:-auto}"

cd "$PROJECT_ROOT"

echo "Configuration: $CONFIG_FILE"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Device: $DEVICE"
echo "Model: PINN (Physics-Informed Neural Network)"
echo ""

if [ ! -f "dataset/train.npz" ]; then
    echo "[ERROR] Dataset not found. Please run 02_generate_dataset first."
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Starting PINN model training..."
echo ""

python cli.py train --config "$CONFIG_FILE" --model pinn --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --device "$DEVICE"

echo ""
echo "============================================================"
echo "  PINN model training completed!"
echo "============================================================"

if [ -f "Deep_learning/checkpoints/pinn_best.pth" ]; then
    echo "Checkpoint: Deep_learning/checkpoints/pinn_best.pth"
fi

echo ""
read -p "Press Enter to exit..."