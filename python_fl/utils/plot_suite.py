# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import matplotlib.pyplot as plt

def plot_mean_ci(x, mean, std, n, label: str, color: str | None = None):
    import numpy as np
    import math
    x = list(x); mean = list(mean); std = list(std); n = list(n)
    ci = [1.96 * (s / math.sqrt(max(1, int(nn)))) if int(nn) > 0 else 0.0 for s, nn in zip(std, n)]
    lo = [m - c for m, c in zip(mean, ci)]
    hi = [m + c for m, c in zip(mean, ci)]
    line, = plt.plot(x, mean, label=label, color=color)
    c = color or line.get_color()
    plt.fill_between(x, lo, hi, color=c, alpha=0.2)

def savefig(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
