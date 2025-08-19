#!/usr/bin/env python3
import yaml, os, argparse
from src.orchestrator import run_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_baseline.yaml")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    outdir = args.outdir or f"experiments/{cfg.get('experiment_name','run')}"
    os.makedirs("experiments", exist_ok=True)
    run_experiment(cfg, out_dir=outdir)
