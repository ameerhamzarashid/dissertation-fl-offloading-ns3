# -*- coding: utf-8 -*-
"""
Aggregate Exp3 scaling: summarize totals and averages.
Input: results/exp3/python_logs.csv
Output: results/exp3/summary_scaling.csv with total_comm_bytes,total_energy_j,avg_latency_ms
"""
import os, argparse, csv
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/exp3/python_logs.csv")
    ap.add_argument("--out", dest="outp", default="results/exp3/summary_scaling.csv")
    args = ap.parse_args()

    if not os.path.exists(args.inp):
        raise SystemExit("Exp3 input not found: " + args.inp)
    with open(args.inp, "r") as f:
        rows = list(csv.DictReader(f))

    comm = np.array([float(r.get("comm_bytes", 0)) for r in rows], dtype=float)
    energy = np.array([float(r.get("energy_j", 0.0)) for r in rows], dtype=float)
    # support either per-round 'latency_ms' or 'avg_latency_ms' (some writers use avg_latency_ms)
    lat = np.array([float(r.get("latency_ms", r.get("avg_latency_ms", 0.0))) for r in rows], dtype=float)
    os.makedirs(os.path.dirname(args.outp), exist_ok=True)
    with open(args.outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["total_comm_bytes","total_energy_j","avg_latency_ms"]) 
        w.writeheader()
        w.writerow({
            "total_comm_bytes": float(comm.sum()),
            "total_energy_j": float(energy.sum()),
            "avg_latency_ms": float(lat.mean()) if lat.size else 0.0,
        })
    print("[exp3] wrote", args.outp)

if __name__ == "__main__":
    main()
