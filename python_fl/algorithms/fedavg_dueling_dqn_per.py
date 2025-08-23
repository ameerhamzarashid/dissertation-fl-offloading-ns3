# -*- coding: utf-8 -*-
"""
FedAvg + Dueling-DQN + PER baseline scaffold.

Training loop is orchestrated by training/train_baseline.py; this file exposes stubs
for aggregator logic and agent wiring, keeping FedAvg vanilla (no sparsification).
"""
from __future__ import annotations
import numpy as np
from typing import List

class FedAvgAggregator:
    def __init__(self):
        pass

    def aggregate(self, client_grads: List[np.ndarray]) -> np.ndarray:
        if not client_grads:
            raise ValueError("No client gradients to aggregate")
        return np.mean(np.stack(client_grads, axis=0), axis=0)
