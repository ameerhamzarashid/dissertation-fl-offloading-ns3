# KF7029–MEC–FL‑NS‑3: One‑Flow Guide (End‑to‑End, Energy‑First)

Canonical, copy‑pasteable instructions for building, running, and evaluating the MEC + FL offloading system with NS‑3 (not OMNeT++). Phase 2 adds action‑controlled offloading, FL‑update traffic on the radio link, per‑task latency logging, and round‑level ingestion (comm bytes + energy deltas + avg latency).

All commands are for WSL. Project root in WSL: `/mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3`.

---

## 0) Repository layout (expected)

# Note: make sure the bridge "variant" matches the model file you export as
# `MODEL_PATH`. If you start the bridge with `BRIDGE_VARIANT=baseline` while
# `MODEL_PATH` points to an SFEA file (`sfea.pt`) the server will attempt to
# load a DQN and print state-dict mismatch warnings. Always set
# `BRIDGE_VARIANT` before starting the bridge (see SFEA section below).
```
KF7029-MEC-FL-NS-3/
├─ configs/                      # YAML configs (Exp1/2/3, Top-k sweep)
├─ ns3_module/                   # C++ module + example + build/run scripts
│  ├─ include/
│  │  ├─ logging-helper.h        # writes bytes/energy + task_latency.csv
│  │  └─ offload-app.h           # consumes actions; models FL upload bytes
│  └─ src/
│     ├─ logging-helper.cc       # minimal stub (includes header)
│     └─ offload-app.cc          # minimal stub (includes header)
├─ python_fl/
│  ├─ co_sim/bridge_server.py    # returns action + update_bytes
│  └─ utils/ns3_ingest.py        # per-round comm/energy/latency aggregation
├─ data/ (raw_logs, processed, models, plots)
├─ experiments/ (01_comm_efficiency, ...)
├─ results/ (exp1, exp2, exp3)
├─ scripts/ (setup_env, run_all_experiments, sync_ns3_logs)
├─ tests/
└─ docs/ (this file + others)
```

Do not rename folders — paths below assume this layout.

---

## 1) Paths & terminals

- WSL project root: `/mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3`
- Terminal roles:
  - [P] Python bridge / experiments (venv active)
  - [N] NS‑3 build & run

NS‑3 CMake package dir (ns3_DIR) after install step:
`/mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION/build/install/lib/cmake/ns3`

---

## 2) Prereqs (WSL)

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config python3-venv python3-pip git wget \
  libgtk-3-dev libsqlite3-dev libboost-all-dev
```

---



## 3) Python environment (P) [Single-seed & Multi-seed]

```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
./scripts/setup_env.sh .venv
source .venv/bin/activate
python -c "import numpy, yaml, matplotlib; print('python env ok')"
```

Install CPU PyTorch if your configs require it:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---



## 4) Train ML model (P) — REQUIRED BEFORE ANY NS-3 RUNS [Single-seed & Multi-seed]

**[P]** Activate your Python environment and train your model. This must be done before any bridge or NS-3 run so the bridge can load the trained model.

**Baseline (Dueling DQN):**
```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
source .venv/bin/activate
python -m python_fl.training.train_baseline --config configs/exp1_comm_efficiency.yaml
# Model will be saved to data/models/dueling_dqn.pt
```

**SFEA variant:**
```bash
python -m python_fl.training.train_sfea --config configs/exp1_comm_efficiency.yaml
# Model will be saved to data/models/sfea.pt
```

You only need to train once per model/config. If you retrain, rerun this step and update MODEL_PATH below.


## 4.1) Build NS-3 and this module (N) [Single-seed & Multi-seed]

```bash
export NS3_VERSION=3.43
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
wget https://www.nsnam.org/release/ns-allinone-$NS3_VERSION.tar.bz2
tar -xjf ns-allinone-$NS3_VERSION.tar.bz2

cd ns-allinone-$NS3_VERSION/ns-$NS3_VERSION
mkdir -p build && cd build
cmake .. -G "Ninja"
ninja
cmake --install . --prefix "$(pwd)/install"

export NS3_DIR="/mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION/build/install/lib/cmake/ns3"
ls "$NS3_DIR/ns3Config.cmake" && ls "$NS3_DIR/ns3Targets.cmake"

cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns3_module
./scripts/build_ns3_module.sh Release "$NS3_DIR"
```

Outputs: `ns3_module/build/run_mec_scenario`

---

## 5) Baseline flow — start bridge, run NS‑3, save logs (P + N) [Single-seed]

Start the Python bridge as Baseline (P):
```bash
export MODEL_PATH=data/models/dueling_dqn.pt
```
```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
export BRIDGE_VARIANT=baseline NUM_PARAMS=2000000
# or run helper: ./scripts/env_baseline.sh 2000000
python -m python_fl.co_sim.bridge_server
```

Quick server checks (P):
```bash
# Expect to see on startup:
# [bridge] listening on ('0.0.0.0', 50051)
# [bridge] config: variant=baseline num_params=2000000 ... update_bytes=...

# From another terminal, confirm the port is open:
ss -ltnp | grep 50051 || echo "bridge not bound"

# If the port is taken, free it and retry:
fuser -k 50051/tcp
```

Run NS‑3 and save raw logs (N):
```bash
# Note: NS‑3 writes to project‑root data/raw_logs and it is overwritten on every run
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns3_module
./scripts/run_ns3.sh Release --duration=600

# After launch, expect these in the NS-3 console:
#  - [BridgeClient] connected to 127.0.0.1:50051
#  - [OffloadApp] Using bridge update_bytes=... variant=...
# If instead you see:
#  - [OffloadApp] No bridge reply; applied env fallback update_bytes=...
# then NS-3 didn’t reach the bridge. Quick checks:
#  - Run the bridge inside WSL (same environment as NS-3), not Windows PowerShell.
#  - Ensure the port is listening: ss -ltnp | grep 50051 || echo "bridge not bound"
#  - Free the port if stuck and restart bridge: fuser -k 50051/tcp
#  - Start NS-3 a second after the bridge prints "listening".
#  - Use the default host/port (127.0.0.1:50051) unless you changed both sides.
  # Note: BridgeClient uses a host TCP socket in WSL; keep both processes in WSL.

# Immediately copy to the Baseline folder
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
mkdir -p results/exp1/baseline/raw_ns3
cp -f data/raw_logs/*.csv results/exp1/baseline/raw_ns3/
```

The bridge replies with `{action, update_bytes}`. `update_bytes` is large for Baseline (dense).

---

## 6) SFEA flow — start bridge, run NS‑3, save logs (P + N) [Single-seed]

Start the Python bridge as SFEA (Top‑k) (P):
```bash
export MODEL_PATH=data/models/sfea.pt
```
```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
# Stronger signal: smaller k and larger model
export BRIDGE_VARIANT=sfea K_PERCENT=5 NUM_PARAMS=2000000
# or run helper: ./scripts/env_sfea.sh 5 2000000
python -m python_fl.co_sim.bridge_server
```

Important: Keep this bridge process running while you run NS‑3 in another terminal. Do not press Ctrl+C until the NS‑3 run completes and you have copied the logs. If you stop the bridge before/while NS‑3 runs, SFEA won’t be applied and Baseline/SFEA logs will be identical.

Keep this bridge terminal open until you complete the SFEA NS‑3 run and copy logs.

Quick server checks (P):
```bash
# Expect to see on startup:
# [bridge] listening on ('0.0.0.0', 50051)
# [bridge] config: variant=sfea num_params=2000000 k_percent=5 update_bytes=...

# From another terminal, confirm the port is open:

# IMPORTANT: set `BRIDGE_VARIANT=sfea` before starting the bridge when using
# `data/models/sfea.pt`. If you start the bridge with a different variant
# (for example `baseline`) while `MODEL_PATH` points to `sfea.pt`, the server
# will try to load the wrong model type and print state-dict mismatch warnings.
ss -ltnp | grep 50051 || echo "bridge not bound"

# If the port is taken, free it and retry:
fuser -k 50051/tcp
```

Run NS‑3 and save raw logs (N):
```bash
# Note: NS‑3 writes to project‑root data/raw_logs and it is overwritten on every run
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns3_module
./scripts/run_ns3.sh Release --duration=600

# Immediately copy to the SFEA folder
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
mkdir -p results/exp1/sfea/raw_ns3
cp -f data/raw_logs/*.csv results/exp1/sfea/raw_ns3/
```

Note: Only one variant is active per bridge session. To switch variants, stop the bridge (Ctrl+C), set the new env vars, and start it again.

Important: NS‑3 overwrites `data/raw_logs/` on each run. After each run, copy the CSVs into the correct variant folder before starting the next run to avoid overwriting.

---

## 7) Verify raw logs (per‑variant copies) [Single-seed]

Confirm the Baseline copy:
```bash
ls -l results/exp1/baseline/raw_ns3
head -n 5 results/exp1/baseline/raw_ns3/link_bytes.csv
head -n 5 results/exp1/baseline/raw_ns3/radio_energy.csv
head -n 5 results/exp1/baseline/raw_ns3/task_latency.csv
```

Confirm the SFEA copy:
```bash
ls -l results/exp1/sfea/raw_ns3
head -n 5 results/exp1/sfea/raw_ns3/link_bytes.csv
head -n 5 results/exp1/sfea/raw_ns3/radio_energy.csv
head -n 5 results/exp1/sfea/raw_ns3/task_latency.csv
```

Do not open or reuse `data/raw_logs` after both runs; it gets overwritten each time. Always use the per‑variant copies above for ingestion.

---

## 8) Ingest to Exp1 results (P) [Single-seed]

Ingest Baseline from its per‑variant folder (do not copy again from data/raw_logs):
```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
python -m python_fl.scripts.run_exp1 \
  --config configs/exp1_comm_efficiency.yaml \
  --variant baseline \
  --ns3-log-dir results/exp1/baseline/raw_ns3 \
  --window-s 4
head -n 5 results/exp1/baseline/python_logs.csv
```

Ingest SFEA from its per‑variant folder:
```bash
python -m python_fl.scripts.run_exp1 \
  --config configs/exp1_comm_efficiency.yaml \
  --variant sfea \
  --ns3-log-dir results/exp1/sfea/raw_ns3 \
  --window-s 4
head -n 5 results/exp1/sfea/python_logs.csv
```

Each row in python_logs.csv: round, comm_bytes, avg_latency_ms, energy_j.

---

## 9) Analysis and plots [Single-seed]

### 9.1) Exp1: Baseline vs SFEA

Plot per-round curves for comm_bytes and energy_j:
```bash
python -m python_fl.scripts.plot_exp1
```
Outputs:
- data/plots/exp1_comm_bytes.png
- data/plots/exp1_energy_j.png

Energy reduction must hold: SFEA curves should be strictly below Baseline. If not, re-run SFEA with a smaller K_PERCENT (e.g., 2–5) and a large NUM_PARAMS (e.g., 2000000), keeping the bridge running during NS‑3.

### 9.2) Exp2: Energy ↔ Latency trade-off

Run Exp2, aggregate, and plot the trade-off (Energy vs p95 Latency):
```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml
python -m python_fl.scripts.aggregate_exp2
python -m python_fl.scripts.plot_exp2
```
Outputs:
- results/exp2/python_logs.csv
- results/exp2/summary_tradeoff.csv
- data/plots/exp2_tradeoff.png

### 9.3) Exp3: Scalability & heterogeneity

Run Exp3, aggregate, and plot the scaling summary:
```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml
python -m python_fl.scripts.aggregate_exp3
python -m python_fl.scripts.plot_exp3
```
Outputs:
- results/exp3/python_logs.csv
- results/exp3/summary_scaling.csv
- data/plots/exp3_scaling.png


---

## 10) Multi-seed runs and aggregation (required for core results)

This project uses a single canonical multi-seed flow. Follow these exact steps to run all seeds and variants without confusion:

- The automated multi-seed script `./scripts/run_multiseed.sh` is the supported, recommended flow. It starts and stops the bridge per seed/variant, runs NS‑3, copies raw logs, and ingests per-seed results.

Quick, copy/pasteable commands (venv active):
```bash
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
source .venv/bin/activate
# IMPORTANT: do NOT have MODEL_PATH exported unless you intend to force the same model for every run
unset MODEL_PATH || true
./scripts/run_multiseed.sh
```

If you prefer to run each variant/model one-by-one (safer, six runs total for 2 variants × 3 seeds), use one of these options.

- Run all three seeds for a single variant at a time (recommended):
```bash
# Baseline (three seeds)
export RUN_VARIANTS=baseline
export RUN_SEEDS=42,123,999
unset MODEL_PATH || true
export BIND_WAIT_SECONDS=60
./scripts/run_multiseed.sh

# SFEA (three seeds)
export RUN_VARIANTS=sfea
export RUN_SEEDS=42,123,999
unset MODEL_PATH || true
export BIND_WAIT_SECONDS=60
./scripts/run_multiseed.sh
```

- Or run the entire 6 runs sequentially in one command (the script will loop):
```bash
export RUN_VARIANTS=baseline,sfea
export RUN_SEEDS=42,123,999
unset MODEL_PATH || true
export BIND_WAIT_SECONDS=60
./scripts/run_multiseed.sh
```

Notes:
- Running one variant at a time reduces the chance of cross-variant MODEL_PATH mistakes and makes diagnosing bridge startup issues easier.
- `RUN_SEEDS` and `RUN_VARIANTS` are comma-separated lists and override the defaults in the script.

Options:
- Run a subset of variants:
  ```bash
  export RUN_VARIANTS=baseline
  ./scripts/run_multiseed.sh
  ```
- Force a single custom model for every run (not recommended):
  ```bash
  export MODEL_PATH=/full/path/to/custom_model.pt
  export EXPLICIT_MODEL_PATH=1
  ./scripts/run_multiseed.sh
  ```

Why you may have observed Baseline logs with SFEA or vice versa:
- If `MODEL_PATH` is exported globally to point at an SFEA file (for example `data/models/sfea.pt`) and you start a bridge with `BRIDGE_VARIANT=baseline`, the bridge will attempt to load a DQN from that file and print state-dict warnings. The automated script avoids this by using per-variant defaults unless you set `EXPLICIT_MODEL_PATH=1`.

Port conflicts and who starts the bridge:
- The multi-seed script starts/stops the bridge on port 50051 for each run. Do not run any manual bridge on port 50051 while using the automated script — the script will abort if the port is already in use.

If you'd like, I can also add a tiny `scripts/preflight_multiseed.sh` that prints the effective env and port checks before running; say the word and I'll add it.


### 10.1 Exp1: Aggregate across seeds and compute significance

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  # If needed for t-tests:
  python -m pip install -q scipy
  python -m python_fl.scripts.aggregate_exp1
  ```
Outputs:
- results/exp1/summary_round_stats.csv (mean±std±count per round, per variant)
- results/exp1/summary_seed_totals.csv (per‑seed totals for each metric)
- results/exp1/summary_exp1.csv — Significance is established on seed‑matched pairs using a paired t‑test at α = 0.05, with Shapiro–Wilk normality checks applied to pairwise differences and a Wilcoxon signed‑rank test used as a non‑parametric fallback when normality is rejected.

Interpretation:
- Expect SFEA mean total `energy_j` and `comm_bytes` < Baseline with p_value < 0.05.
- If not significant, increase `NUM_PARAMS` or decrease `K_PERCENT`, and/or extend `DURATION` and rerun 10.1.

### 10.3 Exp1: Final plots (with 95% CI) and acceptance

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  python -m python_fl.scripts.plot_exp1
  ```
Outputs:
- data/plots/exp1_comm_bytes.png
- data/plots/exp1_energy_j.png

Acceptance checklist:
- SFEA curves sit below Baseline for both comm_bytes and energy_j.
- summary_exp1.csv shows p_value < 0.05 for energy_j (and ideally comm_bytes).
- All per‑seed folders exist and contain `python_logs.csv`.

### 10.1 Exp2: Run multi-seed Exp2

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  source .venv/bin/activate
  # Run baseline and SFEA seeds (script handles per-variant runs)
  export RUN_VARIANTS=baseline,sfea
  export RUN_SEEDS=42,123,999
  unset MODEL_PATH || true
  export BIND_WAIT_SECONDS=60
  ./scripts/run_multiseed_exp2.sh
  ```
Artifacts created per seed:
- results/exp2/seed_*/raw_ns3/*.csv
- results/exp2/seed_*/python_logs.csv

### 10.2 Exp2: Aggregate Exp2 multi-seed results

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  python -m python_fl.scripts.aggregate_exp2
  ```
Outputs:
- results/exp2/summary_tradeoff.csv

### 10.3 Exp2: Plot Exp2 multi-seed results

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  python -m python_fl.scripts.plot_exp2
  ```
Outputs:
- data/plots/exp2_tradeoff.png

### 10.1 Exp3: Run multi-seed Exp3

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  source .venv/bin/activate
  # Run baseline and SFEA seeds (script handles per-variant runs)
  export RUN_VARIANTS=baseline,sfea
  export RUN_SEEDS=42,123,999
  unset MODEL_PATH || true
  export BIND_WAIT_SECONDS=60
  ./scripts/run_multiseed_exp3.sh
  ```
Artifacts created per seed:
- results/exp3/seed_*/raw_ns3/*.csv
- results/exp3/seed_*/python_logs.csv

### 10.2 Exp3: Aggregate Exp3 multi-seed results

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  python -m python_fl.scripts.aggregate_exp3
  ```
Outputs:
- results/exp3/summary_scaling.csv

### 10.3 Exp3: Plot Exp3 multi-seed results

- [P]
  ```bash
  cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
  python -m python_fl.scripts.plot_exp3
  ```
Outputs:
- data/plots/exp3_scaling.png

---

## 11) Troubleshooting

- No bytes / energy stays zero: increase offloading temporarily (raise `m_offloadSizeThreshold` in `offload-app.h`), verify bridge is running, and set `BRIDGE_VARIANT` appropriately.
- Ingestor few rows: extend `--duration` or decrease `--window-s`.
- CMake can’t find ns‑3: pass the correct `ns3_DIR` to the build script (see step 4).
- Port already in use: change `DEFAULT_PORT` in `python_fl/co_sim/bridge_server.py` and NS‑3 bridge client to match.

- Flat/zero plots in step 9:
  - Make sure you pulled latest and rebuilt `ns3_module` (energy rows are logged on task completion).
    - Rebuild: `cd ns3_module && ./scripts/build_ns3_module.sh Release "$NS3_DIR"`
  - Start the Python bridge first (step 5). If the bridge isn’t connected, no FL `update_bytes` are added and bytes stay 0.
  - For a quick sanity check, force radio traffic: either lower `m_offloadSizeThreshold` in `ns3_module/include/offload-app.h` (e.g., `1*1024*1024`) or set `TaskGenerator` `m_sizeLow/m_sizeHigh` to 1–2 MB; then rebuild and rerun step 6.

---


## 12) Final acceptance checklist (must all be true)

- [ ] Python venv is active in [P] terminal
- [ ] Model is trained and saved to `data/models/dueling_dqn.pt` or `data/models/sfea.pt`
- [ ] `MODEL_PATH` is set before running the bridge or multi-seed script
  - NOTE: do not export `MODEL_PATH` globally before running `./scripts/run_multiseed.sh` unless you intend to force the same model for all variants. If you must force a custom path, set `EXPLICIT_MODEL_PATH=1` in the shell so the script will not override it.
- [ ] `ns3_module/build/run_mec_scenario` exists
- [ ] Multi-seed script runs without errors and bridge prints `[bridge] Loaded model from ...`
- [ ] Aggregation and plotting scripts run without errors
- [ ] Plots in `data/plots/` show SFEA below Baseline for comm_bytes and energy_j
- [ ] All per-seed folders exist and contain `python_logs.csv`
- [ ] summary_exp1.csv shows p_value < 0.05 for energy_j (and ideally comm_bytes)

---

## 13) Exactly what changed (Phase 2)

- `ns3_module/include/logging-helper.h` → writes `task_latency.csv` in addition to bytes/energy
- `ns3_module/include/offload-app.h` → consumes actions; models FL upload bytes using bridge `update_bytes`
- `python_fl/co_sim/bridge_server.py` → returns `{action, update_bytes}` from `BRIDGE_VARIANT`, `NUM_PARAMS`, `K_PERCENT`
- `python_fl/utils/ns3_ingest.py` → aggregates per‑round `comm_bytes`, `energy_j` (via deltas), and `avg_latency_ms`

Keep this guide in `docs/FULL_GUIDE.md`.
