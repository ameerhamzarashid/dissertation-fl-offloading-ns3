# -*- coding: utf-8 -*-
import numpy as np
from python_fl.algorithms.topk_sparsifier import TopKSparsifier

def test_topk_keeps_expected_fraction():
    g = np.arange(100, dtype=np.float32)
    spars = TopKSparsifier(k_percent=10.0, error_feedback=False)
    s, m = spars.compress("c1", g)
    kept = int(m.sum())
    assert kept == 10, f"expected 10 nonzeros, got {kept}"

def test_error_feedback_accumulates_residual():
    g = np.ones(10, dtype=np.float32)
    spars = TopKSparsifier(k_percent=10.0, error_feedback=True)
    s1, m1 = spars.compress("c1", g)
    # only one element kept -> residual sum should be ~9
    residual_sum = float(np.sum(g - s1))
    assert residual_sum > 8.9, f"residual too small: {residual_sum}"
