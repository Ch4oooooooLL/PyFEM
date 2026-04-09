#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

TARGET_ENV="${FEM_CONDA_ENV:-}"
if [[ -z "$TARGET_ENV" ]]; then
    if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
        TARGET_ENV="${CONDA_DEFAULT_ENV}"
    else
        TARGET_ENV="fem"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] Conda not found. Install Conda and ensure \`conda\` is on PATH."
    exit 1
fi

conda run -n base python -m tools.cli bootstrap --env-name "$TARGET_ENV"
eval "$(conda shell.bash hook)"
conda activate "$TARGET_ENV"
exec python -m tools.cli "$@"
