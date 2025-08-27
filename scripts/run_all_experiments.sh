#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

WINDOW_S=${WINDOW_S:-6}

echo "▶ Exp1: Communication efficiency (ingest baseline + sfea)"
python -m python_fl.scripts.run_exp1 \
	--config configs/exp1_comm_efficiency.yaml \
	--variant baseline \
	--ns3-log-dir results/exp1/baseline/raw_ns3 \
	--window-s "$WINDOW_S" || true
python -m python_fl.scripts.run_exp1 \
	--config configs/exp1_comm_efficiency.yaml \
	--variant sfea \
	--ns3-log-dir results/exp1/sfea/raw_ns3 \
	--window-s "$WINDOW_S" || true

echo "▶ Exp1: aggregate (multi-seed, if present) and plot"
python -m python_fl.scripts.aggregate_exp1 || true
python -m python_fl.scripts.plot_exp1 || true

echo "▶ Exp2: Performance trade-off"
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml
python -m python_fl.scripts.aggregate_exp2 || true
python -m python_fl.scripts.plot_exp2 || true

echo "▶ Exp3: Scalability"
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml
python -m python_fl.scripts.aggregate_exp3 || true
python -m python_fl.scripts.plot_exp3 || true

echo "✅ All experiments launched (placeholders write python_logs.csv)."
