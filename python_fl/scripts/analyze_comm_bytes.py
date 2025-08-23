# -*- coding: utf-8 -*-
"""
Analyze communication bytes emitted by NS-3 and Python logs.
"""
from __future__ import annotations
import argparse, os
from ..utils.metrics import read_ns3_bytes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3_csv", type=str, default="data/raw_logs/link_bytes.csv")
    args = ap.parse_args()

    tx = read_ns3_bytes(args.ns3_csv)
    print(f"Total TX bytes (NS-3): {tx}")

if __name__ == "__main__":
    main()
