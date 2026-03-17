#!/bin/bash
set -e

echo "============================================================"
echo "  FEM Project - Full Pipeline Execution"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

SKIP_STATIC=""
SKIP_DATASET=""
SKIP_TRAIN=""
SKIP_PREDICT=""

for arg in "$@"; do
    case "$arg" in
        --skip-static) SKIP_STATIC="--skip-static" ;;
        --skip-dataset) SKIP_DATASET="--skip-dataset" ;;
        --skip-train) SKIP_TRAIN="--skip-train" ;;
        --skip-predict) SKIP_PREDICT="--skip-predict" ;;
    esac
done

echo "Pipeline Options:"
[ -z "$SKIP_STATIC" ] && echo "  [x] Static Analysis" || echo "  [ ] Static Analysis (skipped)"
[ -z "$SKIP_DATASET" ] && echo "  [x] Dataset Generation" || echo "  [ ] Dataset Generation (skipped)"
[ -z "$SKIP_TRAIN" ] && echo "  [x] Model Training" || echo "  [ ] Model Training (skipped)"
[ -z "$SKIP_PREDICT" ] && echo "  [x] Condition Prediction" || echo "  [ ] Condition Prediction (skipped)"
echo ""

echo "Starting full pipeline execution..."
echo ""

python cli.py pipeline $SKIP_STATIC $SKIP_DATASET $SKIP_TRAIN $SKIP_PREDICT

echo ""
echo "============================================================"
echo "  Full pipeline completed successfully!"
echo "============================================================"
echo ""
read -p "Press Enter to exit..."