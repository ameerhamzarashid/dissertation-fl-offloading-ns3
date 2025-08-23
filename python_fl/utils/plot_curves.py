# -*- coding: utf-8 -*-
"""
Plot curves using matplotlib only. One chart per figure; no styles/colors set.
"""
from __future__ import annotations
import csv, os
import matplotlib.pyplot as plt

def plot_from_csv(csv_path: str, out_png: str, x_key: str, y_key: str, title: str):
    xs, ys = [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row[x_key]))
            ys.append(float(row[y_key]))
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
