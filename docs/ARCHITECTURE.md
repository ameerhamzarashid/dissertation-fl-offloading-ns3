# System Architecture

**Goal:** Energy-first Mobile Edge Computing (MEC) with Federated Learning (FL) for dynamic task offloading in NS-3.

```
+------------------------+        TCP (len+JSON)        +---------------------------+
|        NS-3            | <--------------------------> |         Python             |
|  - LTE/EPC stack       |                               |  - Bridge Server          |
|  - UEs + eNBs          |                               |  - Reward (E+L)           |
|  - TaskGenerator       |                               |  - Agents (DQN, SFEA)     |
|  - OffloadApp          |                               |  - Trainers & Scripts     |
|  - EnergyTracker       |                               |  - Metrics/Plots          |
+------------------------+                               +---------------------------+
         |                                                            |
         | CSV (bytes, energy)                                        | CSV (python_logs.csv)
         v                                                            v
       data/raw_logs/                                              results/exp*/

```

**Key components**

- **NS-3 module (`ns3_module/`)**
  - `MecSim`: builds topology, mobility, devices, installs per-UE apps.
  - `TaskGenerator`: Poisson arrivals (size 10–20 MB, 900–1100 Mcycles).
  - `OffloadApp`: local vs offload decisions (action received via bridge).
  - `EnergyTracker`: CPU + radio energy; later replace with ns-3 EnergyModel.
  - `BridgeClient`: TCP JSON to Python; length-prefixed frames.
  - `LoggingHelper`: CSV logs for bytes & energy.

- **Python (`python_fl/`)**
  - `co_sim/bridge_server.py`: accepts state → returns action; aggregates metrics.
  - `reward_functions/energy_latency_reward.py`: r = −(w_l L + w_e E) with min–max.
  - `agents/`: Dueling-DQN + PER.
  - `algorithms/`: FedAvg baseline and SFEA (Top-k + error feedback).
  - `training/`: trainers & evaluator.
  - `scripts/`: exp runners and analysis helpers.

**Data & results**

- `data/raw_logs/`: NS-3 emits `link_bytes.csv`, `radio_energy.csv`.
- `results/exp*/python_logs.csv`: per-round aggregates (comm bytes, latency, energy, reward).
- `data/plots/`: generated figures (curves).

