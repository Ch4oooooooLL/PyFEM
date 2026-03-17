#!/bin/bash

echo "============================================================"
echo "  FEM Project - Status Check"
echo "============================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

python cli.py status

echo ""
echo "============================================================"
echo ""
read -p "Press Enter to exit..."