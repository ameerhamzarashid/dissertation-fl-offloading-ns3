# -*- coding: utf-8 -*-
"""
Experiment 1: Communication efficiency (Baseline vs SFEA).

Reads configs/exp1_comm_efficiency.yaml and runs baseline + proposed stubs,
then compares comm bytes using NS-3 logs (if present) and Python logs.
"""
from __future__ import annotations
import argparse, yaml, os
from ..utils.metrics import read_ns3_bytes, MetricsAggregator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp1_comm_efficiency.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_base = cfg.get("logging", {}).get("persist", {})
    # Example paths if NS-3 emitted them; safe if missing
    ns3_bytes = read_ns3_bytes("data/raw_logs/link_bytes.csv")

    # Minimal write to show pipeline working
    metrics = MetricsAggregator(out_dir=cfg.get("logging", {}).get("out_dir", "results/exp1"))
    metrics.log_round(0, ns3_bytes, 0.0, 0.0, 0.0)
    metrics.flush()
    print(f"[exp1] total_ns3_tx_bytes={ns3_bytes} -> logged to python_logs.csv")

if __name__ == "__main__":
    main()
