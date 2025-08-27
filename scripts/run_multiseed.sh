#!/usr/bin/env bash
set -euo pipefail

# ===== user knobs =====
# default variants (can be overridden by setting RUN_VARIANTS env var as comma-separated list)
VARIANTS=("baseline" "sfea")
SEEDS=("42" "123" "999")
DURATION=300                 # seconds per run
WINDOW_S=6                   # seconds per round for ingestion
NUM_PARAMS=1000000           # model size (float32 params)
# Default SFEA Top-k (% kept). The script will fallback to this if K_PERCENT is not set in environment.
K_PERCENT_DEFAULT=10
# =======================

# How long to wait (seconds) for the bridge to bind the TCP port before aborting
# Can be overridden in the environment: export BIND_WAIT_SECONDS=60
BIND_WAIT_SECONDS=${BIND_WAIT_SECONDS:-30}

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

# Allow optional override: export RUN_VARIANTS="baseline" or "sfea" or "baseline,sfea"
if [[ -n "${RUN_VARIANTS:-}" ]]; then
  IFS=',' read -r -a VARIANTS <<< "${RUN_VARIANTS}"
fi
# Allow optional override for seeds: export RUN_SEEDS="42,123,999"
if [[ -n "${RUN_SEEDS:-}" ]]; then
  IFS=',' read -r -a SEEDS <<< "${RUN_SEEDS}"
fi

for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    echo "============================================================"
    echo ">>> Running variant=${variant}  seed=${seed}"
    echo "============================================================"

    export BRIDGE_VARIANT="${variant}"
    export NUM_PARAMS="${NUM_PARAMS}"
    if [[ "${variant}" == "sfea" ]]; then
      # use explicit env K_PERCENT if provided, otherwise fall back to script default
      export K_PERCENT="${K_PERCENT:-${K_PERCENT_DEFAULT}}"
    else
      unset K_PERCENT 2>/dev/null || true
    fi
    # Port pre-check: avoid ambiguous behavior if a manual bridge is already
    # running (the script launches its own bridge per run). Fail fast so the
    # user doesn't accidentally connect NS-3 to a manually-started bridge.
    if ss -ltnp 2>/dev/null | grep -q ':50051'; then
      echo "ERROR: TCP port 50051 appears to be in use. Stop any manual bridge or free the port (fuser -k 50051/tcp) before running this script."
      exit 1
    fi

    # Ensure MODEL_PATH is set for the bridge. Prefer an explicit env var, but
    # detect obvious mismatches and override with the per-variant default
    # unless the user explicitly set EXPLICIT_MODEL_PATH=1 to force their path.
    if [[ -n "${MODEL_PATH:-}" ]]; then
      base_name="$(basename "${MODEL_PATH}")"
      if [[ "${variant}" == "baseline" && "${base_name}" == *sfea* && "${EXPLICIT_MODEL_PATH:-}" != "1" ]]; then
        echo ">>> Warning: exported MODEL_PATH appears to be SFEA (${MODEL_PATH}) while variant=baseline; overriding to per-variant default. Set EXPLICIT_MODEL_PATH=1 to force your path."
        export MODEL_PATH="${ROOT}/data/models/dueling_dqn.pt"
      elif [[ "${variant}" == "sfea" && "${base_name}" == *dueling* && "${EXPLICIT_MODEL_PATH:-}" != "1" ]]; then
        echo ">>> Warning: exported MODEL_PATH appears to be Baseline (${MODEL_PATH}) while variant=sfea; overriding to per-variant default. Set EXPLICIT_MODEL_PATH=1 to force your path."
        export MODEL_PATH="${ROOT}/data/models/sfea.pt"
      fi
    else
      if [[ "${variant}" == "baseline" ]]; then
        export MODEL_PATH="${ROOT}/data/models/dueling_dqn.pt"
      else
        export MODEL_PATH="${ROOT}/data/models/sfea.pt"
      fi
    fi
    echo ">>> Using MODEL_PATH=${MODEL_PATH} (variant=${variant})"

    # warn if the model file is missing (the bridge will also warn, but
    # this makes the multi-seed output clearer)
    if [[ ! -f "${MODEL_PATH}" ]]; then
      echo ">>> WARNING: MODEL_PATH does not exist: ${MODEL_PATH}"
    fi

    # prepare per-run output dir (also used for bridge logs)
    OUT_RAW_DIR="${ROOT}/results/exp1/${variant}/seed_${seed}"
    mkdir -p "${OUT_RAW_DIR}"

    # choose python executable: prefer project venv if present
    if [[ -x "${ROOT}/.venv/bin/python" ]]; then
      PYTHON_EXEC="${ROOT}/.venv/bin/python"
    else
      PYTHON_EXEC="python"
    fi

    # start bridge in background and capture stdout/stderr to per-run log
    BRIDGE_LOG="${OUT_RAW_DIR}/bridge.log"
    # write a small env dump so we can debug per-run failures
    {
      echo "=== bridge env dump ==="
      echo "PYTHON_EXEC=${PYTHON_EXEC}"
      echo "BRIDGE_VARIANT=${BRIDGE_VARIANT:-}"
      echo "MODEL_PATH=${MODEL_PATH:-}"
      echo "NUM_PARAMS=${NUM_PARAMS:-}"
      echo "K_PERCENT=${K_PERCENT:-}"
      echo "EXPLICIT_MODEL_PATH=${EXPLICIT_MODEL_PATH:-}"
      echo "PWD=$(pwd)"
      echo "WHOAMI=$(whoami)"
      echo "PATH=${PATH}"
      echo "=== end env dump ==="
    } >"${BRIDGE_LOG}"
    # run unbuffered and append runtime output so the log contains both env and any tracebacks
    "$PYTHON_EXEC" -u -m python_fl.co_sim.bridge_server >>"${BRIDGE_LOG}" 2>&1 &
    BRIDGE_PID=$!

    # brief wait and check the bridge process didn't immediately die (show log if it did)
    sleep 1
    if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
      echo ">>> ERROR: bridge process (pid=${BRIDGE_PID}) exited unexpectedly; see ${BRIDGE_LOG} for details"
      echo "---- bridge log start ----"
      sed -n '1,200p' "${BRIDGE_LOG}" || true
      echo "---- bridge log end ----"
      exit 1
    fi

    # wait for the bridge to accept connections. The probe below is
    # multi-pronged and fast:
    #  1) try bash /dev/tcp (very cheap, no new process)
    #  2) fall back to the selected Python executable to attempt a TCP
    #     connect (keeps behaviour across environments)
    #  3) final fallback: check 'ss' listening output
    # If BIND_WAIT_SECONDS is set to 0 the script will skip waiting (risky).
    bound=0
    probe_host=127.0.0.1
    probe_port=50051
    probe_log_line() {
      printf "[%s] %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >>"${BRIDGE_LOG}"
    }

    if [[ "${BIND_WAIT_SECONDS}" -eq 0 ]]; then
      echo ">>> WARNING: skipping wait for bridge bind (BIND_WAIT_SECONDS=0)"
      bound=1
    else
      echo ">>> Waiting up to ${BIND_WAIT_SECONDS}s for bridge to accept connections..."
      probe_start=$(date +%s)
      for i in $(seq 1 "${BIND_WAIT_SECONDS}"); do
        # 1) fast bash /dev/tcp probe (no new interpreter)
        if (echo > "/dev/tcp/${probe_host}/${probe_port}") >/dev/null 2>&1; then
          probe_log_line "probe:${i}: /dev/tcp succeeded"
          bound=1
          break
        fi

        # 2) Python fallback: attempt a short connect using the chosen Python
        if command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
          if "${PYTHON_EXEC}" -c 'import socket,sys
try:
    s=socket.socket()
    s.settimeout(0.6)
    s.connect(("127.0.0.1", 50051))
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
' >/dev/null 2>&1; then
            probe_log_line "probe:${i}: python connect succeeded"
            bound=1
            break
          else
            probe_log_line "probe:${i}: python connect failed"
          fi
        fi

        # 3) last-resort: check ss/listening (cheap, not proof of accept)
        if ss -ltnp 2>/dev/null | grep -q ":${probe_port}"; then
          probe_log_line "probe:${i}: ss reports listening on :${probe_port}"
          bound=1
          break
        fi

        sleep 1
      done
      probe_end=$(date +%s)
      probe_elapsed=$((probe_end-probe_start))
      echo ">>> bind probe elapsed: ${probe_elapsed}s (attempts logged to ${BRIDGE_LOG})"
    fi

    if [[ $bound -ne 1 ]]; then
      echo ">>> ERROR: bridge did not bind port ${probe_port} after ${BIND_WAIT_SECONDS}s; aborting"
      echo "---- bridge log start (${BRIDGE_LOG}) ----"
      sed -n '1,300p' "${BRIDGE_LOG}" || true
      echo "---- bridge log end ----"
      echo "---- system diagnostics: ss -ltnp | grep ${probe_port} ----"
      ss -ltnp 2>/dev/null | grep ":${probe_port}" || true
      echo "---- bridge process info (ps -f) pid=${BRIDGE_PID} ----"
      ps -o pid,ppid,uid,user,cmd -p "${BRIDGE_PID}" || true
      # ensure bridge pid is cleaned up
      kill "${BRIDGE_PID}" >/dev/null 2>&1 || true
      exit 1
    fi

    # run ns-3 (if --seed unsupported by your binary, it's just ignored)
    "${NS3_BIN}" --duration="${DURATION}" --seed="${seed}"

    # stop bridge
    kill "${BRIDGE_PID}" >/dev/null 2>&1 || true
    sleep 1

    # collect logs
    OUT_RAW="${ROOT}/results/exp1/${variant}/seed_${seed}/raw_ns3"
    mkdir -p "${OUT_RAW}"
    cp -f "${ROOT}/data/raw_logs/"*.csv "${OUT_RAW}/" || true

    # ingest to per-seed results (write python_logs.csv into the seed folder)
    python -m python_fl.scripts.run_exp1 \
      --config "${ROOT}/configs/exp1_comm_efficiency.yaml" \
      --variant "${variant}" \
      --ns3-log-dir "${OUT_RAW}" \
      --window-s "${WINDOW_S}" \
      --out-dir "${OUT_RAW_DIR}"
    # verify ingestion produced per-seed python_logs.csv
    if [[ ! -f "${OUT_RAW_DIR}/python_logs.csv" ]]; then
      echo ">>> ERROR: ingestion failed to write ${OUT_RAW_DIR}/python_logs.csv"
      echo "---- bridge log head (${BRIDGE_LOG}) ----"
      sed -n '1,200p' "${BRIDGE_LOG}" || true
      echo "---- raw_ns3 listing (${OUT_RAW}) ----"
      ls -la "${OUT_RAW}" || true
      exit 1
    fi
  # mark successful completion for this seed so users can verify progress
  echo ">>> Completed variant=${variant} seed=${seed} -> ${OUT_RAW_DIR}/python_logs.csv"
  done
done

echo "Multi-seed runs complete. Results under results/exp1/{baseline,sfea}/seed_*/"
