
# KF7029–MEC–FL–NS-3: One‑Flow Guide (Energy‑First)

Purpose: This guide provides the exact, copy-paste runnable steps and canonical decisions for this repository. All steps are for WSL and the main path C:\Users\ameer\Downloads\KF7029-MEC-FL-NS3. NS-3 LTE/EPC is used (not OMNeT++). Energy is the top priority.

---

## 0) Repository Layout (must stay exactly like this)


KF7029-MEC-FL-NS3/
├─ configs/
├─ ns3_module/
├─ python_fl/
├─ data/
├─ experiments/
├─ results/
├─ scripts/
├─ tests/
├─ docs/
│  ├─ FULL_GUIDE.md
│  ├─ ARCHITECTURE.md
│  ├─ ENERGY_METHOD.md
│  └─ REPRODUCIBILITY.md
└─ README.md

Do not rename or move these folders. All paths below assume this layout and the main project directory.

---


## 1) Paths, Terminals, and Step-by-Step Flow



# KF7029–MEC–FL–NS-3: One‑Flow Guide (Energy‑First)

Purpose: This guide provides the exact, copy-paste runnable steps and canonical decisions for this repository. All steps are for WSL and the main path C:\Users\ameer\Downloads\KF7029-MEC-FL-NS3. NS-3 LTE/EPC is used (not OMNeT++). Energy is the top priority.

## 0) Repository Layout (must stay exactly like this)

KF7029-MEC-FL-NS3/
├─ configs/
├─ ns3_module/
├─ python_fl/
├─ data/
├─ experiments/
├─ results/
├─ scripts/
├─ tests/
├─ docs/
│  ├─ FULL_GUIDE.md
│  ├─ ARCHITECTURE.md
│  ├─ ENERGY_METHOD.md
│  └─ REPRODUCIBILITY.md
└─ README.md

Do not rename or move these folders. All paths below assume this layout and the main project directory.

## 1) Paths, Terminals, and Step-by-Step Flow

Main Project Path:
- Windows: C:\Users\ameer\Downloads\KF7029-MEC-FL-NS3
- WSL: /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3

All commands must be run in WSL. Use the WSL terminal for all Python, NS-3, and experiment scripts. Do not use any other terminal or shell.

Terminal Roles:
- [P] Python/Experiment Terminal (venv active)
- [N] NS-3 Build/Run Terminal

NS-3 CMake package directory (ns3_DIR):
- Always use: /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION/build/install/lib/cmake/ns3

## 2) Python Environment Setup ([P], WSL, ~2 min)

cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
./scripts/setup_env.sh .venv
source .venv/bin/activate
python -c "import numpy, yaml, matplotlib; print('python env ok')"

## 3) Build the NS‑3 Module ([N], WSL, ~2-5 min)

sudo apt update
sudo apt install -y build-essential cmake ninja-build wget python3 g++ pkg-config libgtk-3-dev libsqlite3-dev libboost-all-dev

export NS3_VERSION=3.43
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
wget https://www.nsnam.org/release/ns-allinone-$NS3_VERSION.tar.bz2
tar -xjf ns-allinone-$NS3_VERSION.tar.bz2

cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION
mkdir -p build
cd build
cmake .. -G "Ninja"
ninja
cmake --install . --prefix "$(pwd)/install"

export NS3_DIR="/mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION/build/install/lib/cmake/ns3"
ls "$NS3_DIR/ns3Config.cmake"
ls "$NS3_DIR/ns3Targets.cmake"

cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns3_module
./scripts/build_ns3_module.sh Release "$NS3_DIR"

# If you need to clean and rebuild NS-3:
cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION/build
rm -rf *
cmake .. -G "Ninja"
ninja
cmake --install . --prefix "$(pwd)/install"

## 4) Start the Python Co‑Sim Bridge ([P], WSL, ~10 sec)

cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3
python -m python_fl.co_sim.bridge_server

## 5) Run the NS‑3 Scenario ([N], WSL, ~1-5 min)

cd /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns3_module
./scripts/run_ns3.sh Release --nUe=20 --nEnb=5 --duration=300

## 5.1: # NS-3 logs
ls -l /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/data/raw_logs
head -n 5 /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/data/raw_logs/link_bytes.csv
head -n 5 /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/data/raw_logs/radio_energy.csv

# Experiment logs
ls -l /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/results/exp1
ls -l /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/results/exp2
ls -l /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/results/exp3

## 6) Run Experiment Scripts ([P], WSL, ~1-10 min per experiment)

python -m python_fl.scripts.run_exp1 --config configs/exp1_comm_efficiency.yaml
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml
./scripts/run_all_experiments.sh

## 7) SFEA Top‑k Sweep ([P], WSL)

python -m python_fl.scripts.sweep_topk --config configs/topk_sweep.yaml

## 8) Plotting & Quick Analysis ([P], WSL)

python -c "from python_fl.utils.plot_curves import plot_from_csv; plot_from_csv('results/exp1/python_logs.csv','data/plots/exp1_comm.png','round','comm_bytes','Comm Bytes per Round')"
cat data/raw_logs/link_bytes.csv
cat data/raw_logs/radio_energy.csv

## 9) Config Canon (source of truth)

Key defaults (from configs/sim_default.yaml):
- Area: 1000×1000 m, Duration: 600 s, Seed: 42
- Users: 20, eNBs: 5, Radio: LTE, Bandwidth: 20 MHz
- UE Tx power: 23 dBm, Noise PSD: −174 dBm/Hz
- Poisson arrivals: λ=0.5 tasks/s/user
- Task size: 10–20 MB, Complexity: 900–1100 Mcycles
- MU CPU: 1.5 GHz, Edge CPU: 15 GHz
- FL: FedAvg, Rounds: 100, Local epochs: 5
- Agent: Dueling-DQN + PER
- Sparsification: Top-k + Error-Feedback (SFEA)
- Reward: Energy + Latency weighted (0.5, 0.5), online min-max

## 10) What’s Complete vs. Next

Complete
- Full repo skeleton with configs, NS-3 module build/run, Python co-sim server.
- Energy-first CSV logging on NS-3 side + Python metrics CSV writers.
- Top-k sparsifier (with error feedback), FedAvg + DQN/PER stubs, trainers, tests.

Next
1. Closed-loop actions: in NS-3 OffloadApp, replace the threshold stub with actions received from the bridge so that local vs offload is policy-driven every task.
2. Metrics ingestion: in Python, read data/raw_logs/*.csv per round and merge with RL/FL stats → write complete results/exp*/python_logs.csv for Exp1/2/3.
3. Agent training: enable Dueling-DQN+PER updates on real state/next-state with the energy-latency reward (from reward_functions/energy_latency_reward.py); integrate SFEA aggregator during FL rounds.

## 11) Troubleshooting

- CMake cannot find ns-3: Pass the package dir explicitly: ./scripts/build_ns3_module.sh Release /mnt/c/Users/ameer/Downloads/KF7029-MEC-FL-NS3/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION/build/install/lib/cmake/ns3
- Port already in use: Change ns3.tcp_bridge.port in configs/sim_default.yaml and DEFAULT_PORT in python_fl/co_sim/bridge_server.py so both match.
- Empty NS-3 CSVs: Increase --duration, verify log_energy/log_link_bytes: true in configs/sim_default.yaml, and ensure the bridge is running before NS-3 starts.
- Matplotlib errors: Activate the venv and confirm pip show matplotlib succeeds.

## 12) Verification Checklist

- [ ] Venv active (.venv), python env ok printed.
- [ ] NS-3 built (ns3_module/build/run_mec_scenario exists).
- [ ] Bridge running (python_fl.co_sim.bridge_server on :50051).
- [ ] NS-3 run produced data/raw_logs/link_bytes.csv and radio_energy.csv.
- [ ] results/exp*/python_logs.csv created by run_exp scripts.
- [ ] Plots saved under data/plots/.

## 13) File Index (for assistants)

- NS-3: ns3_module/include/*.h, ns3_module/src/*.cc, example examples/run_mec_scenario.cc
- Bridge: python_fl/co_sim/{bridge_server,tcp_protocol,message_schema}.py
- Algorithms: python_fl/algorithms/{fedavg_dueling_dqn_per,sfea_topk_error_feedback,topk_sparsifier}.py
- Agents: python_fl/agents/{dueling_dqn,replay_buffer_per}.py
- Reward: python_fl/reward_functions/energy_latency_reward.py
- Training: python_fl/training/{train_baseline,train_sfea,evaluator}.py
- Drivers: python_fl/scripts/{run_exp1,run_exp2,run_exp3,sweep_topk,analyze_comm_bytes}.py
- Configs: configs/*.yaml
- Docs: docs/*.md

---

End of One-Flow Guide. All steps above are for WSL and the main path C:\Users\ameer\Downloads\KF7029-MEC-FL-NS3.
```

> `sparsification.k_percent` values are defined in the sweep list in YAML.

---

## 8) Plotting & Quick Analysis

Example: plot Comm Bytes per Round for Exp1.

```bash
python -c "from python_fl.utils.plot_curves import plot_from_csv; \
plot_from_csv('results/exp1/python_logs.csv','data/plots/exp1_comm.png','round','comm_bytes','Comm Bytes per Round')"
```

Inspect raw NS‑3 logs:
```
data/raw_logs/link_bytes.csv
data/raw_logs/radio_energy.csv
```

---

## 9) Config Canon (source of truth)

Key defaults (from `configs/sim_default.yaml`):
- Area: **1000×1000 m**, Duration: **600 s**, Seed: **42**
- Users: **20**, eNBs: **5**, Radio: **LTE**, Bandwidth: **20 MHz**
- UE Tx power: **23 dBm**, Noise PSD: **−174 dBm/Hz**
- Poisson arrivals: **λ=0.5 tasks/s/user**
- Task size: **10–20 MB**, Complexity: **900–1100 Mcycles**
- MU CPU: **1.5 GHz**, Edge CPU: **15 GHz**
- FL: **FedAvg**, Rounds: **100**, Local epochs: **5**
- Agent: **Dueling‑DQN + PER** (baseline hooks ready)
- Sparsification: **Top‑k + Error‑Feedback** (SFEA) for proposed
- Reward: **Energy + Latency weighted** (0.5, 0.5), online min‑max

> Change variants and outputs via `configs/exp*.yaml` (e.g., Exp1 baseline vs SFEA k=10%).

---

## 10) What’s Complete vs. Next

**Complete**
- Full repo skeleton with configs, NS‑3 module build/run, Python co‑sim server.
- Energy‑first CSV logging on NS‑3 side + Python metrics CSV writers.
- Top‑k sparsifier (with error feedback), FedAvg + DQN/PER stubs, trainers, tests.

**Next (for assistants to implement precisely)**
1. **Closed‑loop actions**: in NS‑3 `OffloadApp`, replace the threshold stub with actions received from the bridge (already connected) so that local vs offload is policy‑driven every task.
2. **Metrics ingestion**: in Python, read `data/raw_logs/*.csv` per round and merge with RL/FL stats → write complete `results/exp*/python_logs.csv` for Exp1/2/3.
3. **Agent training**: enable Dueling‑DQN+PER updates on real state/next‑state with the **energy‑latency** reward (from `reward_functions/energy_latency_reward.py`); integrate SFEA aggregator during FL rounds.

> Constraint: **Keep NS‑3 (LTE/EPC) — do not switch to OMNeT++**. Any enhancement must retain the above defaults unless a config explicitly overrides them.

---

## 11) Troubleshooting

- **CMake cannot find ns‑3**  
  Pass the package dir explicitly:  
  `./scripts/build_ns3_module.sh Release /absolute/path/to/lib/cmake/ns3`  
  (Windows: `-Ns3Dir "C:\ns3\lib\cmake\ns3"`)

- **Port already in use**  
  Change `ns3.tcp_bridge.port` in `configs/sim_default.yaml` and `DEFAULT_PORT` in `python_fl/co_sim/bridge_server.py` so both match.

- **Empty NS‑3 CSVs**  
  Increase `--duration`, verify `log_energy/log_link_bytes: true` in `configs/sim_default.yaml`, and ensure the bridge is running before NS‑3 starts.

- **Matplotlib errors**  
  Activate the venv and confirm `pip show matplotlib` succeeds.

---

## 12) Verification Checklist

- [ ] Venv active (`.venv`), `python env ok` printed.
- [ ] NS‑3 built (`ns3_module/build/run_mec_scenario[.exe]` exists).
- [ ] Bridge running (`python_fl.co_sim.bridge_server` on `:50051`).
- [ ] NS‑3 run produced `data/raw_logs/link_bytes.csv` and `radio_energy.csv`.
- [ ] `results/exp*/python_logs.csv` created by run_exp scripts.
- [ ] Optional plots saved under `data/plots/`.

---

## 13) File Index (for assistants)

- **NS‑3**: `ns3_module/include/*.h`, `ns3_module/src/*.cc`, example `examples/run_mec_scenario.cc`  
- **Bridge**: `python_fl/co_sim/{bridge_server,tcp_protocol,message_schema}.py`  
- **Algorithms**: `python_fl/algorithms/{fedavg_dueling_dqn_per,sfea_topk_error_feedback,topk_sparsifier}.py`  
- **Agents**: `python_fl/agents/{dueling_dqn,replay_buffer_per}.py`  
- **Reward**: `python_fl/reward_functions/energy_latency_reward.py`  
- **Training**: `python_fl/training/{train_baseline,train_sfea,evaluator}.py`  
- **Drivers**: `python_fl/scripts/{run_exp1,run_exp2,run_exp3,sweep_topk,analyze_comm_bytes}.py`  
- **Configs**: `configs/*.yaml` (sim defaults, reward, exp variants, sweeps)  
- **Docs**: `docs/*.md`

---


---

**End of One‑Flow Guide. All steps above are for WSL and the main path C:\Users\ameer\Downloads\KF7029-MEC-FL-NS3.**
