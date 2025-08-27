# -*- coding: utf-8 -*-
"""
Aggregate Exp2 trade-off: compute total energy and latency percentiles.
Input: results/exp2/python_logs.csv with columns round,comm_bytes,latency_ms,energy_j,reward
Output: results/exp2/summary_tradeoff.csv with energy_total_j, latency_p50_ms, latency_p95_ms
"""
import os, argparse, csv
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/exp2/python_logs.csv")
    ap.add_argument("--out", dest="outp", default="results/exp2/summary_tradeoff.csv")
    args = ap.parse_args()

    rows = []
    if os.path.exists(args.inp):
        with open(args.inp, "r") as f:
            rows = list(csv.DictReader(f))
    else:
        raise SystemExit("Exp2 input not found: " + args.inp)

    # support either 'latency_ms' or 'avg_latency_ms' depending on upstream writer
    lat = np.array([float(r.get("latency_ms", r.get("avg_latency_ms", 0.0))) for r in rows], dtype=float)
    eng = np.array([float(r.get("energy_j", 0.0)) for r in rows], dtype=float)
    out_dir = os.path.dirname(args.outp)
    os.makedirs(out_dir, exist_ok=True)
    with open(args.outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["energy_total_j","latency_p50_ms","latency_p95_ms"]) 
        w.writeheader()
        w.writerow({
            "energy_total_j": float(eng.sum()),
            "latency_p50_ms": float(np.percentile(lat, 50)) if lat.size else 0.0,
            "latency_p95_ms": float(np.percentile(lat, 95)) if lat.size else 0.0,
        })
    print("[exp2] wrote", args.outp)

if __name__ == "__main__":
    main()
