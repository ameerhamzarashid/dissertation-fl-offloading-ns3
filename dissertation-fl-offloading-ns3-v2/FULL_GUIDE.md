Minute-by-minute:
1) venv + pip install reqs.
2) Train baseline: `python scripts/run_fedavg.py --config configs/run_baseline.yaml`
3) Train enhanced: `python scripts/run_fedavg.py --config configs/run_enhanced.yaml`
4) Export actions for both configs.
5) Build ns-3 helpers via CMake.
6) Generate traces.
7) Replay actions with mec_client to produce KPI CSVs.
8) Inspect plots & JSON summaries in each run directory.
One-shot: `bash scripts/make_all.sh`
