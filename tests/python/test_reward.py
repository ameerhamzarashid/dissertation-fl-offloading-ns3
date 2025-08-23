# -*- coding: utf-8 -*-
from python_fl.reward_functions.energy_latency_reward import EnergyLatencyReward

def test_reward_is_non_positive():
    r = EnergyLatencyReward(0.5, 0.5)
    # First call sets min/max -> reward 0
    rv1 = r.compute(1000.0, 1.0)
    assert -1.0 <= rv1 <= 0.0
    # Worse values should not produce positive reward
    rv2 = r.compute(1200.0, 1.2)
    assert rv2 <= 0.0
