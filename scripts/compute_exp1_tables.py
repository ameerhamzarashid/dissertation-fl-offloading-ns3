#!/usr/bin/env python3
"""
Compute per-seed Deadline-OK (%) and run paired inferential tests for Exp1.

Outputs a small text file with Table VIII (per-seed metrics) and Table IX (stat tests).
"""
import os
import sys
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results' / 'exp1'
DEADLINE_MS = 1000.0


def read_latency_pct(path, deadline_ms=DEADLINE_MS):
    df = pd.read_csv(path)
    if 'latency_ms' not in df.columns:
        return math.nan
    total = len(df)
    ok = (df['latency_ms'] <= deadline_ms).sum()
    return 100.0 * ok / total if total > 0 else math.nan


def paired_tests(name, base, sfea, alpha=0.05, n_boot=10000):
    # differences (base - sfea)
    diffs = np.array(base) - np.array(sfea)
    mean_diff = diffs.mean()
    sd_diff = diffs.std(ddof=1)
    n = len(diffs)

    # Shapiro-Wilk normality on differences
    try:
        shapiro_p = stats.shapiro(diffs).pvalue
    except Exception:
        shapiro_p = 0.0

    if n < 3:
        test_name = 'n<3'
        p_value = float('nan')
        ci_low = float('nan')
        ci_high = float('nan')
    else:
        if shapiro_p > 0.05:
            test_name = 'ttest_rel'
            tstat, p_value = stats.ttest_rel(base, sfea)
            # 95% CI for mean difference
            sem = sd_diff / math.sqrt(n)
            dfree = n - 1
            tcrit = stats.t.ppf(1 - alpha/2, dfree)
            ci_low = mean_diff - tcrit * sem
            ci_high = mean_diff + tcrit * sem
        else:
            test_name = 'wilcoxon'
            # wilcoxon requires non-zero diffs; if all zeros it returns p=1
            try:
                wstat, p_value = stats.wilcoxon(diffs)
            except Exception:
                p_value = 1.0
            # bootstrap 95% CI for mean difference
            rng = np.random.default_rng(12345)
            boots = []
            for _ in range(n_boot):
                samp = rng.choice(diffs, size=n, replace=True)
                boots.append(samp.mean())
            ci_low = np.percentile(boots, 2.5)
            ci_high = np.percentile(boots, 97.5)

    # Cohen's d for paired samples (mean diff / sd of diffs)
    cohens_d = mean_diff / sd_diff if sd_diff != 0 else float('inf')

    return dict(name=name, n=n, mean_diff=mean_diff, sd_diff=sd_diff,
                shapiro_p=shapiro_p, test=test_name, p_value=p_value,
                ci_low=ci_low, ci_high=ci_high, cohens_d=cohens_d)


def main():
    seed_csv = RESULTS_DIR / 'summary_seed_totals.csv'
    if not seed_csv.exists():
        print('Missing', seed_csv)
        sys.exit(1)

    df = pd.read_csv(seed_csv)
    seeds = df['seed'].tolist()

    rows = []
    for s in seeds:
        row = {'seed': int(s)}
        for policy in ['baseline', 'sfea']:
            lat_path = RESULTS_DIR / policy / f'seed_{s}' / 'raw_ns3' / 'task_latency.csv'
            if lat_path.exists():
                pct = read_latency_pct(lat_path)
            else:
                pct = float('nan')
            row[f'deadline_ok_pct_{policy}'] = pct

        # include summary_seed_totals values (comm_bytes, energy_j, avg_latency_ms)
        base_row = df[df['seed'] == s]
        if len(base_row) == 1:
            row['comm_bytes_base'] = int(base_row['comm_bytes_base'].iloc[0])
            row['comm_bytes_sfea'] = int(base_row['comm_bytes_sfea'].iloc[0])
            row['energy_j_base'] = float(base_row['energy_j_base'].iloc[0])
            row['energy_j_sfea'] = float(base_row['energy_j_sfea'].iloc[0])
            row['avg_latency_ms_base'] = float(base_row['avg_latency_ms_base'].iloc[0])
            row['avg_latency_ms_sfea'] = float(base_row['avg_latency_ms_sfea'].iloc[0])
        else:
            row.update({k: float('nan') for k in ['comm_bytes_base','comm_bytes_sfea','energy_j_base','energy_j_sfea','avg_latency_ms_base','avg_latency_ms_sfea']})

        rows.append(row)

    table_df = pd.DataFrame(rows).sort_values('seed')

    # Save per-seed table (Table VIII)
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table_viii_path = out_dir / 'table_viii_exp1_per_seed.txt'
    with open(table_viii_path, 'w') as f:
        f.write('Table VIII — Per-seed online replay performance (Deadline-OK at 1000 ms)\n')
        f.write(table_df.to_string(index=False))
    print('Wrote', table_viii_path)

    # Now run paired tests for selected metrics
    metrics = [
        ('comm_bytes', 'comm_bytes_base', 'comm_bytes_sfea'),
        ('energy_j', 'energy_j_base', 'energy_j_sfea'),
        ('avg_latency_ms', 'avg_latency_ms_base', 'avg_latency_ms_sfea'),
        ('deadline_ok_pct', 'deadline_ok_pct_baseline', 'deadline_ok_pct_sfea'),
    ]

    results = []
    for name, a_col, b_col in metrics:
        a = table_df[a_col].values
        b = table_df[b_col].values
        res = paired_tests(name, a, b)
        results.append(res)

    table_ix_path = out_dir / 'table_ix_exp1_significance.txt'
    with open(table_ix_path, 'w') as f:
        f.write('Table IX — Paired significance tests (Exp1)\n')
        for r in results:
            f.write(json.dumps(r, default=lambda x: float(x)) + '\n')

    print('Wrote', table_ix_path)
    print('\nSummary:')
    print(table_df)
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
