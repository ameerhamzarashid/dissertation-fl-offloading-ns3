#!/usr/bin/env bash
set -euo pipefail

# 1) Python env
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2) Train baseline (dense FedAvg)
python scripts/run_fedavg.py --config configs/run_baseline.yaml

# 3) Train enhanced (sparse top-k)
python scripts/run_fedavg.py --config configs/run_enhanced.yaml

# 4) Export actions for both
python scripts/export_actions.py --config configs/run_baseline.yaml
python scripts/export_actions.py --config configs/run_enhanced.yaml

# 5) Build ns-3 helpers
bash scripts/build_ns3.sh

# 6) Generate traces (artifact)
bash scripts/gen_traces.sh

# 7) Replay KPIs for both runs
./ns3/build/mec_client --actions=experiments/baseline_dense/actions.csv --out=experiments/baseline_dense/kpis_client.csv
./ns3/build/mec_client --actions=experiments/enhanced_sparse/actions.csv --out=experiments/enhanced_sparse/kpis_client.csv

echo "[ALL DONE] Artifacts in experiments/baseline_dense and experiments/enhanced_sparse"
