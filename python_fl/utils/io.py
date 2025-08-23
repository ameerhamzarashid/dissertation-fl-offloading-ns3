# -*- coding: utf-8 -*-
"""
I/O helpers for YAML and CSV without heavy deps.
"""
from __future__ import annotations
import yaml, csv, os
from typing import Dict, Any, List

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
