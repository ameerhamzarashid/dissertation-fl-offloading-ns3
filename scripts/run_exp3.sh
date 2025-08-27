#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml
python -m python_fl.scripts.aggregate_exp3
python -m python_fl.scripts.plot_exp3
echo "Exp3 complete. See results/exp3 and data/plots/exp3_scaling.png"
