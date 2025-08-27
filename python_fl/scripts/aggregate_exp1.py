# -*- coding: utf-8 -*-
"""
Aggregate Exp1 results across seeds and compare baseline vs SFEA.

Outputs:
- results/exp1/summary_round_stats.csv   (mean±std per round)
- results/exp1/summary_seed_totals.csv   (per-seed totals)
- results/exp1/summary_exp1.csv          (paired t-test on per-seed totals)
"""
import os, glob, argparse
import pandas as pd
import numpy as np
try:
    from scipy.stats import ttest_rel, shapiro, wilcoxon
except Exception:
    ttest_rel = shapiro = wilcoxon = None

METRICS = ["comm_bytes", "energy_j", "avg_latency_ms"]

def load_seed_timeseries(variant: str):
    files = sorted(glob.glob(f"results/exp1/{variant}/seed_*/python_logs.csv"))
    ts = {}
    for f in files:
        seed = os.path.basename(os.path.dirname(f)).replace("seed_","")
        df = pd.read_csv(f).sort_values("round").reset_index(drop=True)
        df["seed"] = seed
        ts[seed] = df
    return ts

def per_round_stats(ts_dict):
    df = pd.concat(ts_dict.values(), ignore_index=True)
    out = df.groupby("round")[METRICS].agg(["mean","std","count"])  # type: ignore
    out.columns = [f"{m}_{stat}" for m,stat in out.columns]
    out = out.reset_index()
    return out

def per_seed_totals(ts_dict):
    rows = []
    for seed, df in ts_dict.items():
        row = {"seed": seed}
        for m in METRICS:
            row[m] = df[m].sum()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/exp1")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base_ts = load_seed_timeseries("baseline")
    sfea_ts = load_seed_timeseries("sfea")
    if not base_ts or not sfea_ts:
        raise SystemExit("No per-seed results found. Run scripts/run_multiseed.sh first.")

    base_round = per_round_stats(base_ts); base_round["variant"] = "baseline"
    sfea_round = per_round_stats(sfea_ts); sfea_round["variant"] = "sfea"
    round_out = pd.concat([base_round, sfea_round], ignore_index=True)
    round_out.to_csv(os.path.join(args.outdir, "summary_round_stats.csv"), index=False)

    base_seed = per_seed_totals(base_ts).rename(columns={m:f"{m}_base" for m in METRICS})
    sfea_seed = per_seed_totals(sfea_ts).rename(columns={m:f"{m}_sfea" for m in METRICS})
    merged = pd.merge(base_seed, sfea_seed, on="seed", how="inner")
    merged.to_csv(os.path.join(args.outdir, "summary_seed_totals.csv"), index=False)

    if ttest_rel is None:
        print("scipy not installed; skipping inferential tests. pip install scipy")
        # still write merged files but without p-values
        rows = []
    else:
        rows = []
        for m in METRICS:
            a = merged[f"{m}_base"].to_numpy()
            b = merged[f"{m}_sfea"].to_numpy()
            diff = a - b
            metric_row = {
                "metric": m,
                "baseline_mean_total": float(np.mean(a)),
                "sfea_mean_total": float(np.mean(b)),
                "mean_diff_total": float(np.mean(diff)),
                "test": None,
                "p_value": None,
                "notes": None,
            }

            # Prefer Shapiro on differences to check normality for paired test
            if shapiro is not None and len(diff) >= 3:
                try:
                    stat, p_shap = shapiro(diff)
                    metric_row["notes"] = f"shapiro_p={p_shap:.4g}"
                    normal = (p_shap > 0.05)
                except Exception as e:
                    normal = True
                    metric_row["notes"] = f"shapiro_error:{e}"
            else:
                normal = True
                if shapiro is None:
                    metric_row["notes"] = "shapiro_missing"
                else:
                    metric_row["notes"] = "shapiro_n<3"

            # Choose test
            try:
                if normal:
                    t, p = ttest_rel(a, b)
                    metric_row["test"] = "ttest_rel"
                    metric_row["p_value"] = float(p)
                else:
                    # Wilcoxon signed-rank (non-parametric paired)
                    if wilcoxon is None:
                        # fallback to t-test if Wilcoxon not available
                        t, p = ttest_rel(a, b)
                        metric_row["test"] = "ttest_rel_fallback"
                        metric_row["p_value"] = float(p)
                        metric_row["notes"] = (metric_row.get("notes", "") + ";wilcoxon_missing")
                    else:
                        stat, p = wilcoxon(a, b)
                        metric_row["test"] = "wilcoxon"
                        metric_row["p_value"] = float(p)
                rows.append(metric_row)
            except Exception as e:
                metric_row["notes"] = (metric_row.get("notes", "") + f";test_error:{e}")
                rows.append(metric_row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "summary_exp1.csv"), index=False)
    print("saved:")
    print(" - results/exp1/summary_round_stats.csv")
    print(" - results/exp1/summary_seed_totals.csv")
    print(" - results/exp1/summary_exp1.csv")

if __name__ == "__main__":
    main()
