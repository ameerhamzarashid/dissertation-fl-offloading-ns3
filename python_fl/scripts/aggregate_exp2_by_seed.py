# -*- coding: utf-8 -*-
"""
Aggregate Exp2 per-seed summary.
Scans results/exp2/seed_*/python_logs.csv and writes:
  results/exp2/summary_tradeoff_by_seed.csv
with columns: seed,energy_total_j,latency_p95_ms
"""
import csv, glob, os
import numpy as np

OUT = "results/exp2/summary_tradeoff_by_seed.csv"
seed_files = sorted(glob.glob("results/exp2/seed_*/python_logs.csv"))
if not seed_files:
    raise SystemExit("No per-seed python_logs.csv found under results/exp2/seed_*/")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fout:
    w = csv.writer(fout)
    w.writerow(["seed","energy_total_j","latency_p95_ms"])
    for p in seed_files:
        seed = os.path.basename(os.path.dirname(p)).split("_")[-1]
        rows = list(csv.DictReader(open(p)))
        lat = [float(r.get("avg_latency_ms", r.get("latency_ms", 0.0))) for r in rows]
        eng = [float(r.get("energy_j", 0.0)) for r in rows]
        energy_total = float(np.sum(eng)) if eng else 0.0
        latency_p95 = float(np.percentile(lat, 95)) if lat else 0.0
        w.writerow([seed, energy_total, latency_p95])

print("wrote", OUT)
