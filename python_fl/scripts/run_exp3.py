# -*- coding: utf-8 -*-
"""
Experiment 3: Scalability & robustness.
"""
from __future__ import annotations
import argparse, yaml, itertools
from ..utils.metrics import MetricsAggregator

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp3_scalability.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg.get("logging", {}).get("out_dir", "results/exp3")
    metrics = MetricsAggregator(out_dir=out_dir)

    # Iterate matrix of n_mobile_users and k_percent (placeholders)
    matrix = cfg.get("matrix", {})
    for nmu in matrix.get("n_mobile_users", [10, 50]):
        for k in matrix.get("k_percent", [20, 50]):
            # Placeholder write per combination
            metrics.log_round(round_i=int(nmu+k), comm_bytes=1000*nmu, latency_ms=50.0, energy_j=0.3, reward=-0.2)
    metrics.flush()
    print("[exp3] wrote placeholder metrics for grid; replace via co-sim later.")

if __name__ == "__main__":
    main()
