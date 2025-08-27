# -*- coding: utf-8 -*-
"""
Train baseline FedAvg + Dueling-DQN + PER (stubbed environment).
Reads YAML config and simulates rounds for testing the pipeline.
"""
from __future__ import annotations
import argparse, yaml, time, os
import numpy as np
from ..agents.dueling_dqn import DuelingDQNAgent, DQNConfig
from ..agents.replay_buffer_per import PERBuffer
from ..algorithms.fedavg_dueling_dqn_per import FedAvgAggregator
from ..utils.seed import set_all_seeds
from ..utils.metrics import MetricsAggregator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_all_seeds(cfg.get("simulation", {}).get("seed", 42))
    out_dir = cfg.get("logging", {}).get("out_dir", "results/exp")
    os.makedirs(out_dir, exist_ok=True)

    # Agents & buffers
    agent = DuelingDQNAgent(DQNConfig())
    buf = PERBuffer(capacity=100000)
    fedavg = FedAvgAggregator()
    metrics = MetricsAggregator(out_dir=out_dir)

    # Simulate interaction (placeholder for real NS-3 co-sim loop)
    # Push random experiences then perform a few updates
    for step in range(1000):
        obs = np.random.rand(3).astype(np.float32)
        act = agent.act(obs)
        rew = np.random.randn() * 0.1
        nxt = np.random.rand(3).astype(np.float32)
        done = np.random.rand() < 0.01
        buf.push(obs, act, rew, nxt, done, td_err=1.0)

        if step % 10 == 0 and step > 0:
            batch, idx = buf.sample(batch_size=64)
            loss, td = agent.update(batch, gamma=0.99)
            buf.update_priorities(idx, td)

    # Dummy FedAvg aggregation
    grads = [np.random.randn(100).astype(np.float32) for _ in range(5)]
    agg = fedavg.aggregate(grads)
    _ = agg  # no-op


    # Write a minimal metrics file to validate pipeline
    metrics.log_round(round_i=0, comm_bytes=123456, latency_ms=123.4, energy_j=0.78, reward=-0.5)
    metrics.flush()

    # Save trained model
    model_dir = os.path.join("data", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dueling_dqn.pt")
    import torch
    torch.save(agent.q.state_dict(), model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
