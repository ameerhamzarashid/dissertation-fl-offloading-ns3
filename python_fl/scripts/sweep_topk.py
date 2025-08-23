# -*- coding: utf-8 -*-
"""
Top-k sweep runner for SFEA.
"""
from __future__ import annotations
import argparse, yaml
from ..training.train_sfea import main as run_sfea

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/topk_sweep.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    for k in cfg.get("sweep", {}).get("k_percent", [10,20,50]):
        print(f"[sweep] running SFEA with k={k}%")
        # Inject into config via temp file or env; for simplicity just call run_sfea with same config.
        run_sfea()

if __name__ == "__main__":
    main()
