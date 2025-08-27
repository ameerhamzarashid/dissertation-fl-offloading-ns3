# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, os
from typing import Dict, List

def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def _bin_index(t: float, window_s: float) -> int:
    if window_s <= 0: return 0
    return int(t // window_s)

def aggregate_rounds(ns3_log_dir: str, window_s: float = 6.0):
    bytes_csv  = os.path.join(ns3_log_dir, "link_bytes.csv")
    energy_csv = os.path.join(ns3_log_dir, "radio_energy.csv")
    lat_csv    = os.path.join(ns3_log_dir, "task_latency.csv")

    bytes_rows  = _read_csv_rows(bytes_csv)
    energy_rows = _read_csv_rows(energy_csv)
    lat_rows    = _read_csv_rows(lat_csv)

    # COMM: sum tx bytes per bin
    comm_bins = {}
    for r in bytes_rows:
        try:
            t = float(r["time_s"]); tx = int(r["tx_bytes"])
        except Exception:
            continue
        b = _bin_index(t, window_s)
        comm_bins[b] = comm_bins.get(b, 0) + tx

    # ENERGY: per-UE deltas then bin
    try:
        energy_rows.sort(key=lambda x: float(x["time_s"]))
    except Exception:
        pass
    last_total = {}
    energy_bins = {}
    for r in energy_rows:
        try:
            t = float(r["time_s"]); ue = int(r["ue_id"]); tot = float(r.get("total_j", 0.0))
        except Exception:
            continue
        prev = last_total.get(ue, tot)
        delta = max(0.0, tot - prev)
        last_total[ue] = tot
        b = _bin_index(t, window_s)
        energy_bins[b] = energy_bins.get(b, 0.0) + delta

    # LATENCY: average per bin (optional file)
    lat_bins, lat_counts = {}, {}
    for r in lat_rows:
        try:
            t = float(r["time_s"]); lat = float(r["latency_ms"])
        except Exception:
            continue
        b = _bin_index(t, window_s)
        lat_bins[b] = lat_bins.get(b, 0.0) + lat
        lat_counts[b] = lat_counts.get(b, 0) + 1

    rounds, all_bins = [], sorted(set(comm_bins) | set(energy_bins) | set(lat_bins))
    for b in all_bins:
        avg_lat = (lat_bins[b] / lat_counts[b]) if lat_counts.get(b, 0) else 0.0
        rounds.append({
            "round": int(b),
            "comm_bytes": int(comm_bins.get(b, 0)),
            "avg_latency_ms": float(avg_lat),
            "energy_j": float(energy_bins.get(b, 0.0)),
        })
    return rounds
