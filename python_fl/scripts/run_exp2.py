# -*- coding: utf-8 -*-
"""
Experiment 2: Energy/Latency trade-off.
"""
from __future__ import annotations
import argparse, yaml
from ..utils.metrics import MetricsAggregator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp2_performance_tradeoff.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    metrics = MetricsAggregator(out_dir=cfg.get("logging", {}).get("out_dir", "results/exp2"))
    # Placeholder values until co-sim loop is wired
    metrics.log_round(0, 12345, 87.6, 0.55, -0.42)
    metrics.flush()
    print("[exp2] wrote placeholder metrics; hook co-sim to replace with real values.")

if __name__ == "__main__":
    main()
