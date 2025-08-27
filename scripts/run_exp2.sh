#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml
python -m python_fl.scripts.aggregate_exp2
python -m python_fl.scripts.plot_exp2
echo "Exp2 complete. See results/exp2 and data/plots/exp2_tradeoff.png"
