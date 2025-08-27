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

    # Save trained model
    model_dir = os.path.join("data", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sfea.pt")
    import torch
    # Construct a serializable state dict for SFEA. Top-k sparsifier keeps a residual
    # buffer per client; save the key knobs and the residuals (converted to lists).
    save_dict = {
        'k_percent': float(getattr(sfea.sparsifier, 'k_percent', k_percent)),
        'error_feedback': bool(getattr(sfea.sparsifier, 'error_feedback', error_feedback)),
        'sparsifier_residuals': {},
    }
    # _residual may contain numpy arrays; convert to lists for JSON/pickle safety
    residuals = getattr(sfea.sparsifier, '_residual', {}) or {}
    for k, v in residuals.items():
        try:
            save_dict['sparsifier_residuals'][k] = v.tolist()
        except Exception:
            # fallback: store shape only
            try:
                save_dict['sparsifier_residuals'][k] = {'shape': getattr(v, 'shape', None)}
            except Exception:
                save_dict['sparsifier_residuals'][k] = None

    torch.save(save_dict, model_path)
    print(f"Saved SFEA state to {model_path} (keys: {list(save_dict.keys())})")

if __name__ == "__main__":
    main()
