# -*- coding: utf-8 -*-
"""
Plot Exp1 Baseline vs SFEA per-round curves.
Inputs:
- results/exp1/baseline/python_logs.csv
- results/exp1/sfea/python_logs.csv

Outputs (PNG):
- data/plots/exp1_comm_bytes.png
- data/plots/exp1_energy_j.png
"""
from __future__ import annotations
import csv, os, argparse
import matplotlib.pyplot as plt
import pandas as pd
from ..utils.plot_suite import plot_mean_ci, savefig

def read_rows(p: str):
    rows = []
    with open(p, "r") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["round"]))
    return rows

def col(rows, key):
    return [float(r[key]) for r in rows]

def rnd(rows):
    return [int(r["round"]) for r in rows]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/exp1/baseline/python_logs.csv")
    ap.add_argument("--sfea", default="results/exp1/sfea/python_logs.csv")
    ap.add_argument("--outdir", default="data/plots")
    args = ap.parse_args()

    # If summary_round_stats.csv exists, plot mean±95% CI; else plot single-run curves
    summary_path = os.path.join(os.path.dirname(os.path.dirname(args.base)), "summary_round_stats.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        for k in ["comm_bytes", "energy_j"]:
            plt.figure()
            for variant in ["baseline", "sfea"]:
                sub = df[df["variant"] == variant].sort_values("round")
                x = sub["round"].tolist()
                mean = sub[f"{k}_mean"].tolist()
                std = sub[f"{k}_std"].tolist()
                n = sub[f"{k}_count"].tolist()
                plot_mean_ci(x, mean, std, n, label=variant)
            plt.xlabel("round")
            plt.ylabel(k)
            plt.legend()
            plt.title(f"Exp1 {k} (mean ± 95% CI)")
            out_png = os.path.join(args.outdir, f"exp1_{k}.png")
            savefig(out_png)
            print(f"saved {out_png}")
    else:
        b = read_rows(args.base)
        s = read_rows(args.sfea)
        os.makedirs(args.outdir, exist_ok=True)
        for k in ["comm_bytes", "energy_j"]:
            plt.figure()
            plt.plot(rnd(b), col(b, k), label="baseline")
            plt.plot(rnd(s), col(s, k), label="sfea")
            plt.xlabel("round")
            plt.ylabel(k)
            plt.legend()
            plt.title(f"Exp1 {k}")
            out_png = os.path.join(args.outdir, f"exp1_{k}.png")
            savefig(out_png)
            print(f"saved {out_png}")

if __name__ == "__main__":
    main()
