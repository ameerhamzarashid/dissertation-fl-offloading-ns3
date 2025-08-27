# -*- coding: utf-8 -*-
"""
Plot normalized Exp3 metrics so different units become comparable.
Reads: results/exp3/summary_scaling.csv (single-row)
Writes: data/plots/exp3_scaling_normalized.png
"""
from __future__ import annotations
import csv, os, argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/exp3/summary_scaling.csv")
    ap.add_argument("--out", default="data/plots/exp3_scaling_normalized.png")
    args = ap.parse_args()

    if not os.path.exists(args.summary):
        raise SystemExit('summary not found: ' + args.summary)
    with open(args.summary, 'r') as f:
        row = next(csv.DictReader(f))

    comm = float(row.get('total_comm_bytes', 0.0))
    energy = float(row.get('total_energy_j', 0.0))
    lat = float(row.get('avg_latency_ms', 0.0))

    vals = np.array([comm, energy, lat], dtype=float)
    # Normalize to 0-1 for relative comparison
    if vals.max() == vals.min():
        norm = np.zeros_like(vals)
    else:
        norm = (vals - vals.min()) / (vals.max() - vals.min())

    labels = ['Comm', 'Energy', 'Latency']
    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, norm, color=['C0','C1','C2'])
    plt.ylim(0,1.05)
    for b, v in zip(bars, norm):
        plt.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha='center', va='bottom')
    plt.title('Exp3 normalized scaling (relative)')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches='tight')
    plt.close()
    print('saved', args.out)


if __name__ == '__main__':
    main()
