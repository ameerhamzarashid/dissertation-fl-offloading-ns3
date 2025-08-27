# -*- coding: utf-8 -*-
"""
Plot Exp2 trade-off: Energy (total J) vs Latency (p95 ms)
Input: results/exp2/summary_tradeoff.csv
Output: data/plots/exp2_tradeoff.png
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_summary(preferred=None):
    # prefer per-seed summary if available
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cand_by_seed = os.path.join(base, 'results', 'exp2', 'summary_tradeoff_by_seed.csv')
    cand_single = os.path.join(base, 'results', 'exp2', 'summary_tradeoff.csv')
    if os.path.exists(cand_by_seed):
        df = pd.read_csv(cand_by_seed)
        return df
    if os.path.exists(cand_single):
        df = pd.read_csv(cand_single)
        # add a synthetic 'seed' column for labeling
        df.insert(0, 'seed', ['summary'])
        return df
    if preferred:
        return pd.read_csv(preferred)
    raise FileNotFoundError('no exp2 summary found')


def plot_presentable(df, out_png, out_pdf):
    # find energy and latency columns
    energy_col = next((c for c in df.columns if 'energy' in c.lower()), None)
    latency_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    if energy_col is None or latency_col is None:
        raise KeyError('expected energy and latency columns in summary')

    labels = df['seed'].astype(str).tolist()
    energy = df[energy_col].astype(float).to_numpy()
    latency = df[latency_col].astype(float).to_numpy()

    # sort by energy for consistent order
    order = np.argsort(energy)[::-1]
    labels = [labels[i] for i in order]
    energy = energy[order]
    latency = latency[order]

    plt.rcParams.update({'font.size': 10, 'font.family': 'serif'})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.0), gridspec_kw={'height_ratios': [2, 1]})

    ax = axes[0]
    bars = ax.bar(labels, energy, color='#2a6f97', edgecolor='k')
    ax.set_ylabel('Total energy (J)')
    ax.set_title('Exp2: per-seed energy and latency (p95)')
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.8)
    for b, v in zip(bars, energy):
        ax.annotate(f"{v:,.0f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax2 = axes[1]
    bars2 = ax2.bar(labels, latency, color='#ff8c42', edgecolor='k')
    ax2.set_ylabel('Latency p95 (ms)')
    ax2.set_xlabel('Seed')
    ax2.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.8)
    for b, v in zip(bars2, latency):
        ax2.annotate(f"{v:,.1f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                     textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def plot_energy_pie(df, out_png, out_pdf):
    energy_col = next((c for c in df.columns if 'energy' in c.lower()), None)
    if energy_col is None:
        return
    labels = df['seed'].astype(str).tolist()
    vals = df[energy_col].astype(float).to_numpy()
    if len(vals) < 2:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    explode = [0.02] * len(vals)
    wedges, texts, autotexts = ax.pie(vals, labels=labels, autopct='%1.1f%%', startangle=140,
                                      explode=explode, pctdistance=0.8)
    ax.set_title('Exp2: per-seed energy share')
    for t in texts + autotexts:
        t.set_fontsize(9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None,
                    help="path to summary CSV (optional). Prefer results/exp2/summary_tradeoff_by_seed.csv")
    ap.add_argument("--out", default="data/plots/exp2_tradeoff.png")
    args = ap.parse_args()

    df = read_summary(args.summary)
    out_png = args.out
    out_pdf = os.path.splitext(out_png)[0] + '.pdf'

    plot_presentable(df, out_png, out_pdf)
    # produce pie chart when multiple seeds present
    pie_png = os.path.splitext(out_png)[0] + '_energy_share_pie.png'
    pie_pdf = os.path.splitext(out_pdf)[0] + '_energy_share_pie.pdf'
    plot_energy_pie(df, pie_png, pie_pdf)
    print('Saved', out_png, 'and', pie_png)


if __name__ == "__main__":
    main()
