Write-Host "▶ Exp1: Communication efficiency"
python -m python_fl.scripts.run_exp1 --config configs/exp1_comm_efficiency.yaml

Write-Host "▶ Exp2: Performance trade-off"
python -m python_fl.scripts.run_exp2 --config configs/exp2_performance_tradeoff.yaml

Write-Host "▶ Exp3: Scalability"
python -m python_fl.scripts.run_exp3 --config configs/exp3_scalability.yaml

Write-Host "✅ All experiments launched (placeholders write python_logs.csv)."
