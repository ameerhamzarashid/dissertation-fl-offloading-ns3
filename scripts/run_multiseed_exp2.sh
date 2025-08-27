#!/usr/bin/env bash
set -euo pipefail

# ===== user knobs =====
SEEDS=("42" "123" "999")
DURATION=300                 # seconds per run
WINDOW_S=6                   # seconds per round for ingestion
NUM_PARAMS=1000000           # model size (float32 params)
# =======================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NS3_BIN="${ROOT}/ns3_module/build/run_mec_scenario"

if [[ ! -x "$NS3_BIN" ]]; then
  echo "NS-3 binary not found at: $NS3_BIN"
  echo "Build it first: ns3_module/scripts/build_ns3_module.sh Release \"$NS3_DIR\""
  exit 1
fi

# activate venv if present
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  source "${ROOT}/.venv/bin/activate"
fi

for seed in "${SEEDS[@]}"; do
  echo "============================================================"
  echo ">>> Running Exp2 seed=${seed}"
  echo "============================================================"

  # start bridge in background
  python -m python_fl.co_sim.bridge_server &
  BRIDGE_PID=$!
  sleep 2

  # run ns-3 (if --seed unsupported by your binary, it's just ignored)
  "${NS3_BIN}" --duration="${DURATION}" --seed="${seed}"

  # stop bridge
  kill "${BRIDGE_PID}" >/dev/null 2>&1 || true
  sleep 1

  # collect logs
  OUT_RAW="${ROOT}/results/exp2/seed_${seed}/raw_ns3"
  mkdir -p "${OUT_RAW}"
  cp -f "${ROOT}/data/raw_logs/"*.csv "${OUT_RAW}/" || true

  # ingest to per-seed results
  python -m python_fl.scripts.run_exp2 \
    --config "${ROOT}/configs/exp2_performance_tradeoff.yaml" \
    --ns3-log-dir "${OUT_RAW}" \
    --window-s "${WINDOW_S}"
done

echo "Multi-seed Exp2 runs complete. Results under results/exp2/seed_*/"
