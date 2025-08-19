# Experiments: Parameters and Metrics

This folder holds experiment outputs and analysis.

Key simulation parameters (from configs):
- Users (I): 30
- Servers (K): 3
- Bandwidth B: 20 MHz; Tx power: 23 dBm; NF: 7 dB; Shadowing: 8 dB (Rayleigh fading)
- Area: 1km x 1km; BS positions: (0,0), (1000,0), (500,866)
- Task sizes D_bits in [4e6, 12e6], cycles C in [1e8, 5e8]
- User CPU f_user=2e9 Hz, Edge CPU F_edge=5e10 Hz, sigma=1e-28, Ptx=0.5 W
- Discount=0.95, batch=64, rounds=60 (epsilon 1.0→0.1)

Evaluation metrics:
- Reward (training snapshot): higher is better
- Latency_ms, Deadline_ok (training snapshot)
- Replay KPIs from ns-3: J = T + λE (lower is better), p50/p95 tail, T_local

How to analyze and plot:
- After running steps 2–7 in the guide, run:

```bash
python scripts/analyze_results.py --baseline experiments/baseline_dense --enhanced experiments/enhanced_sparse --out experiments/analysis
```

Outputs:
- CSVs: `experiments/analysis/kpi_summary.csv`, `kpi_replay_agg.csv`
- Plots: `reward.png`, `latency_ms.png`, `deadline_ok.png`, `replay_J.png`
- Report: `experiments/analysis/REPORT.md`
