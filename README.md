# KF7029 — MEC Federated Learning (NS-3 + Python) : Config Pack

This folder contains the **energy-first** configuration set for your dissertation experiments.
It’s adapted from your proposal/methodology and mapped to **NS-3** (not OMNeT++), keeping the same
network, task, FL and RL assumptions. Use these configs as inputs for the upcoming `run_exp*.py`
scripts and the NS-3 launcher we’ll add next.

## Files

- `configs/sim_default.yaml` — canonical base config (network, tasks, FL, agent, sparsification, logging).
- `configs/reward_energy_latency.yaml` — reward/cost definition focusing on energy+latency.
- `configs/exp1_comm_efficiency.yaml` — Experiment 1: communication overhead comparison (Baseline vs SFEA).
- `configs/exp2_performance_tradeoff.yaml` — Experiment 2: energy/latency trade-off & policy quality.
- `configs/exp3_scalability.yaml` — Experiment 3: load (N=10/50) and sparsity (k=20/50) stress tests.
- `configs/topk_sweep.yaml` — parameter sweep for Top-k sparsification.

## How these map to your docs

- Simulation parameters: area 1000×1000 m, bandwidth 20 MHz, noise −174 dBm/Hz, UE Tx ≈ 23 dBm,
  task arrivals Poisson λ=0.5 tasks/s/user, input size 10–20 MB, complexity 900–1100 Mcycles,
  MU CPU 1.5 GHz, Edge CPU 15 GHz, FL rounds 100, local epochs 5. (From your parameter table.)
- Algorithms: **Baseline** = FedAvg + Dueling-DQN + PER; **Proposed** = SFEA (Top-k + error feedback).
- Experiments: 1) Communication efficiency; 2) Latency & **energy** trade-off; 3) Scalability & k-sensitivity.

> Note: These configs say “radio: lte” by default to ensure they run on vanilla NS-3. If you later
> switch to `nr` (5G) with a suitable NS-3 module, keep the physics identical (20 MHz, same Tx power).

## Suggested usage (coming scripts)

```bash
# Example once scripts are added
python -m python_fl.scripts.run_exp1 --config configs/exp1_comm_efficiency.yaml
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml
# Parameter sweep
python -m python_fl.scripts.sweep_topk --config configs/topk_sweep.yaml
```

## Key sections in the YAML

- `simulation`: area, duration, seeds, user/server counts, mobility model (Gaussian–Markov), radio settings.
- `tasks`: arrival process & task distributions.
- `compute`: CPU frequencies for MU and Edge.
- `fl`: rounds, local epochs, client fraction.
- `agent`: Dueling-DQN architecture + PER knobs (kept tunable).
- `sparsification`: `scheme: none|topk`, `k_percent`, and `error_feedback` toggle.
- `reward`: energy/latency weighting and normalization.
- `logging`: which metrics to log from NS-3 and Python (bytes, latency, energy, reward).

## Next steps

1) You create this folder in your repo exactly as provided here.  
2) I’ll wire the NS-3 energy hooks and the Python co-sim scripts to consume these configs.  
3) We run Exp1 first to baseline communication bytes and validate the energy counters.
