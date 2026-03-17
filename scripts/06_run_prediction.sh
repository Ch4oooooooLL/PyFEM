#!/bin/bash
set -e

echo "============================================================"
echo "  FEM Project - Run Condition Prediction"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${1:-condition_case.yaml}"
OUTPUT_DIR="${2:-}"

cd "$PROJECT_ROOT"

echo "Configuration: $CONFIG_FILE"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
fi
echo ""

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

if [ ! -f "Deep_learning/checkpoints/gt_best.pth" ]; then
    echo "[WARNING] GT checkpoint not found. Run 03_train_gt first."
fi

echo "Starting condition prediction..."
echo ""

if [ -z "$OUTPUT_DIR" ]; then
    python cli.py predict --config "$CONFIG_FILE"
else
    python cli.py predict --config "$CONFIG_FILE" --output-dir "$OUTPUT_DIR"
fi

echo ""
echo "============================================================"
echo "  Condition prediction completed!"
echo "============================================================"
echo ""
read -p "Press Enter to exit..."