# Reproducibility Guide

## Environment
- Python 3.10+ recommended.
- Create a venv and install deps:
  ```bash
  ./scripts/setup_env.sh .venv
  source .venv/bin/activate
  ```

## Build NS-3 example
```bash
cd ns3_module
./scripts/build_ns3_module.sh Release /path/to/ns3/lib/cmake/ns3
./scripts/run_ns3.sh Release --nUe=20 --nEnb=5 --duration=300
```

## Run experiments
```bash
python -m python_fl.co_sim.bridge_server &  # start TCP bridge
python -m python_fl.scripts.run_exp1 --config configs/exp1_comm_efficiency.yaml
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml
```

## Plots
```bash
python -c "from python_fl.utils.plot_curves import plot_from_csv; \
plot_from_csv('results/exp1/python_logs.csv','data/plots/exp1_comm.png','round','comm_bytes','Comm Bytes per Round')"
```

## Seeds
- Global seed from YAML (`simulation.seed`); `utils/seed.py` sets NumPy and (optionally) PyTorch.

## Files produced
- `data/raw_logs/*.csv` from NS-3
- `results/exp*/python_logs.csv` from Python
- `data/plots/*.png` figures
