# -*- coding: utf-8 -*-
"""
Experiment 1: Communication efficiency (Baseline vs SFEA).

CLI now ingests NS-3 CSV logs into per-round metrics and writes
results to results/exp1/<variant>/python_logs.csv.
"""
from __future__ import annotations
import argparse, yaml, os, csv
from ..utils.ns3_ingest import aggregate_rounds

def main():
    ap = argparse.ArgumentParser(description="Exp1: ingest NS-3 logs and write per-round metrics")
    ap.add_argument("--config", type=str, default="configs/exp1_comm_efficiency.yaml",
                    help="YAML config (optional; used to resolve defaults)")
    ap.add_argument("--variant", type=str, choices=["baseline","sfea"], required=True,
                    help="Which variant this run corresponds to")
    ap.add_argument("--ns3-log-dir", type=str, required=True,
                    help="Directory containing NS-3 CSVs (link_bytes.csv, radio_energy.csv, task_latency.csv)")
    ap.add_argument("--window-s", type=float, default=6.0,
                    help="Aggregation window in seconds for round binning")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Override output directory (default: results/exp1/<variant>)")

    args = ap.parse_args()

    # Read config (optional)
    cfg = {}
    try:
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    # Resolve output directory
    default_out = os.path.join("results", "exp1", args.variant)
    out_dir = args.out_dir or cfg.get("logging", {}).get("out_dir", default_out)
    os.makedirs(out_dir, exist_ok=True)

    # Ingest NS-3 CSVs into per-round metrics
    rounds = aggregate_rounds(args.ns3_log_dir, window_s=args.window_s)

    # Write CSV matching the guide: round,comm_bytes,avg_latency_ms,energy_j
    out_csv = os.path.join(out_dir, "python_logs.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = ["round", "comm_bytes", "avg_latency_ms", "energy_j"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rounds:
            w.writerow({k: r.get(k, 0) for k in fieldnames})

    print(f"[exp1] wrote {len(rounds)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
