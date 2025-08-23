# -*- coding: utf-8 -*-
"""
Energy+Latency weighted reward:
  r = - ( w_l * norm(lat_ms) + w_e * norm(energy_j) )
Supports min-max normalization tracking online.
"""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class MinMax:
    lo: float = float("inf")
    hi: float = float("-inf")
    def update(self, x: float):
        self.lo = min(self.lo, x)
        self.hi = max(self.hi, x)
    def norm(self, x: float) -> float:
        if self.hi <= self.lo:  # avoid div by zero
            return 0.0
        return (x - self.lo) / (self.hi - self.lo)

class EnergyLatencyReward:
    def __init__(self, latency_weight: float = 0.5, energy_weight: float = 0.5):
        self.w_l = latency_weight
        self.w_e = energy_weight
        self.mm_l = MinMax()
        self.mm_e = MinMax()

    def compute(self, latency_ms: float, energy_j: float) -> float:
        self.mm_l.update(latency_ms)
        self.mm_e.update(energy_j)
        L = self.mm_l.norm(latency_ms)
        E = self.mm_e.norm(energy_j)
        return - (self.w_l * L + self.w_e * E)
