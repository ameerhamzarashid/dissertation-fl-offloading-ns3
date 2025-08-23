#!/usr/bin/env bash
set -euo pipefail

echo "▶ Exp1: Communication efficiency"
python -m python_fl.scripts.run_exp1 --config configs/exp1_comm_efficiency.yaml

echo "▶ Exp2: Performance trade-off"
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml

echo "▶ Exp3: Scalability"
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml

echo "✅ All experiments launched (placeholders write python_logs.csv)."
