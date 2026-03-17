#!/bin/bash
set -e

echo "============================================================"
echo "  FEM Project - Generate Training Dataset"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG_FILE="${1:-dataset_config.yaml}"
N_JOBS="${2:--1}"

cd "$PROJECT_ROOT"

echo "Configuration: $CONFIG_FILE"
echo "Parallel jobs: $N_JOBS"
echo ""

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Configuration file not found: $CONFIG_FILE"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Starting dataset generation..."
echo ""

python cli.py dataset --config "$CONFIG_FILE" -j "$N_JOBS"

echo ""
echo "============================================================"
echo "  Dataset generation completed successfully!"
echo "============================================================"

if [ -f "dataset/train.npz" ]; then
    echo "Output: dataset/train.npz"
fi

echo ""
read -p "Press Enter to exit..."