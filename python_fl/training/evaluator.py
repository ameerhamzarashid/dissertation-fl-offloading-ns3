# -*- coding: utf-8 -*-
"""
Evaluator utilities (moving averages, convergence tests, etc.).
"""
from __future__ import annotations
from typing import List
import numpy as np

def moving_average(x: List[float], k: int = 10) -> float:
    if not x:
        return 0.0
    k = min(k, len(x))
    return float(np.mean(x[-k:]))
