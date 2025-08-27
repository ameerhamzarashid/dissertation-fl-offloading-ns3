# -*- coding: utf-8 -*-
"""
Plot Exp3 scaling summary.
Input: results/exp3/summary_scaling.csv
Output: data/plots/exp3_scaling.png (bar chart)
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_summary(path=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    candidate = path or os.path.join(base, 'results', 'exp3', 'summary_scaling.csv')
    if not os.path.exists(candidate):
        raise FileNotFoundError(f'exp3 summary not found: {candidate}')
    return pd.read_csv(candidate)


def bytes_to_gb(b):
    return b / (1024 ** 3)


def make_bars(df, out_png, out_pdf):
    row = df.iloc[0]
    comm_gb = bytes_to_gb(float(row['total_comm_bytes']))
    energy_kj = float(row['total_energy_j']) / 1000.0
    latency_ms = float(row.get('avg_latency_ms', 0.0))

    labels = ['Comm (GB)', 'Energy (kJ)', 'Latency (ms)']
    values = [comm_gb, energy_kj, latency_ms]

    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    vmax = max([v for v in values if v > 0])
    vmin = min([v for v in values if v > 0])
    use_log = (vmax / max(vmin, 1e-12)) > 100

    bars = ax.bar(range(len(labels)), values, color=['#4c78a8', '#f58518', '#e45756'], edgecolor='k')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title('Exp3: scaling summary')
    if use_log:
        ax.set_yscale('log')
        plt.gcf().text(0.02, 0.95, 'comm: GB, energy: kJ, latency: ms', fontsize=9, ha='left')

    for rect, v, lbl in zip(bars, values, labels):
        if 'Comm' in lbl:
            s = f"{v:,.2f} GB"
        elif 'Energy' in lbl:
            s = f"{v:,.2f} kJ"
        else:
            s = f"{v:,.2f} ms"
        ax.annotate(s, xy=(rect.get_x() + rect.get_width() / 2, v),
                    xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def make_normalized(df, out_png, out_pdf):
    row = df.iloc[0]
    comm = bytes_to_gb(float(row['total_comm_bytes']))
    energy = float(row['total_energy_j'])
    latency = float(row.get('avg_latency_ms', 0.0))
    arr = np.array([comm, energy, latency], dtype=float)
    if np.all(arr == 0):
        norm = arr
    else:
        norm = (arr - arr.min()) / (arr.max() - arr.min())

    labels = ['Comm', 'Energy', 'Latency']
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, norm, color=['#4c78a8', '#f58518', '#e45756'], edgecolor='k')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Normalized (0-1)')
    ax.set_title('Exp3 normalized scaling (relative)')

    for i, rect in enumerate(bars):
        ax.annotate(f"{norm[i]:.2f}", xy=(rect.get_width(), rect.get_y() + rect.get_height() / 2),
                    xytext=(4, 0), textcoords='offset points', va='center', fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    ap.add_argument("--out", default="data/plots/exp3_scaling.png")
    args = ap.parse_args()
    df = read_summary(args.summary)
    out_png = args.out
    out_pdf = os.path.splitext(out_png)[0] + '.pdf'
    make_bars(df, out_png, out_pdf)
    norm_png = os.path.splitext(out_png)[0] + '_normalized.png'
    norm_pdf = os.path.splitext(out_pdf)[0] + '_normalized.pdf'
    make_normalized(df, norm_png, norm_pdf)
    print('Saved', out_png, 'and', norm_png)


if __name__ == "__main__":
    main()
