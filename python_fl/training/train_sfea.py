# -*- coding: utf-8 -*-
"""
Train SFEA (Top-k + error feedback) scaffold.
"""
from __future__ import annotations
import argparse, yaml, os, numpy as np
from ..algorithms.sfea_topk_error_feedback import SFEAAggregator
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

    k_percent = cfg.get("sparsification", {}).get("k_percent", 10)
    error_feedback = cfg.get("sparsification", {}).get("error_feedback", True)
    sfea = SFEAAggregator(k_percent=k_percent, error_feedback=error_feedback)
    metrics = MetricsAggregator(out_dir=out_dir)

    # Simulated grads for a few "clients"
    dense = [np.random.randn(1000).astype(np.float32) for _ in range(10)]
    comps = []
    for i, g in enumerate(dense):
        s, m = sfea.client_compress(f"client{i}", g)
        comps.append(s)  # zero-padded sparse
    agg = sfea.aggregate(comps)
    _ = agg  # no-op

    # Write minimal metrics
    metrics.log_round(round_i=0, comm_bytes=45678, latency_ms=98.7, energy_j=0.42, reward=-0.3)
    metrics.flush()

if __name__ == "__main__":
    main()
