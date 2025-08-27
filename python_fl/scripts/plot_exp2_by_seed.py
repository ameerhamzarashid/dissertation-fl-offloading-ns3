# -*- coding: utf-8 -*-
"""
Plot Exp2 per-seed tradeoff.
Reads: results/exp2/summary_tradeoff_by_seed.csv (seed,energy_total_j,latency_p95_ms)
Writes: data/plots/exp2_tradeoff_by_seed.png
"""
from __future__ import annotations
import csv, os, argparse
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/exp2/summary_tradeoff_by_seed.csv")
    ap.add_argument("--out", default="data/plots/exp2_tradeoff_by_seed.png")
    args = ap.parse_args()

    if not os.path.exists(args.summary):
        raise SystemExit("Per-seed summary not found: " + args.summary)

    seeds = []
    energies = []
    p95s = []
    with open(args.summary, "r") as f:
        for r in csv.DictReader(f):
            seeds.append(r.get("seed",""))
            energies.append(float(r.get("energy_total_j", 0.0)))
            p95s.append(float(r.get("latency_p95_ms", 0.0)))

    plt.figure(figsize=(6,4))
    plt.scatter(p95s, energies, s=100, c='C1')
    for s, x, y in zip(seeds, p95s, energies):
        plt.text(x, y, f"seed {s}", fontsize=8, va='bottom', ha='right')

    xmin, xmax = min(p95s), max(p95s)
    ymin, ymax = min(energies), max(energies)
    xpad = max(1.0, (xmax - xmin) * 0.1) if xmax > xmin else 1.0
    ypad = max((ymax - ymin) * 0.1, max(1e-6, ymax*0.05)) if ymax > ymin else max(1e-6, ymax*0.05)
    plt.xlim(xmin - xpad, xmax + xpad)
    plt.ylim(max(0, ymin - ypad), ymax + ypad)

    plt.xlabel('Latency p95 (ms)')
    plt.ylabel('Total energy (J)')
    plt.title('Exp2 per-seed Energy vs Latency (p95)')
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches='tight')
    plt.close()
    print('saved', args.out)


if __name__ == '__main__':
    main()
