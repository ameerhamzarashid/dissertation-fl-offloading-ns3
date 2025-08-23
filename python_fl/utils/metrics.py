# -*- coding: utf-8 -*-
"""
Metrics utilities: reads NS-3 CSVs and aggregates; also simple logger.
"""
from __future__ import annotations
import os, csv, json
from typing import Optional

class MetricsAggregator:
    def __init__(self, out_dir: str = "results"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._rows = []

    def log_round(self, round_i: int, comm_bytes: int, latency_ms: float, energy_j: float, reward: float):
        self._rows.append({
            "round": round_i,
            "comm_bytes": int(comm_bytes),
            "latency_ms": float(latency_ms),
            "energy_j": float(energy_j),
            "reward": float(reward),
        })

    def flush(self):
        path = os.path.join(self.out_dir, "python_logs.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["round","comm_bytes","latency_ms","energy_j","reward"])
            w.writeheader()
            for r in self._rows:
                w.writerow(r)

def read_ns3_bytes(csv_path: str) -> int:
    tx_total = 0
    try:
        with open(csv_path, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                tx_total += int(row.get("tx_bytes", 0))
    except FileNotFoundError:
        pass
    return tx_total
