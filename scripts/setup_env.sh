#!/usr/bin/env bash
set -euo pipefail
# Sets up a Python venv and installs required packages.
# Usage: ./scripts/setup_env.sh [.venv]
VENV_DIR="${1:-.venv}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install numpy pyyaml matplotlib
# Optional heavy deps:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
echo "✅ Environment ready. Activate with: source $VENV_DIR/bin/activate"
