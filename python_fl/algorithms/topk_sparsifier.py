
# -*- coding: utf-8 -*-
"""
Top-k sparsifier with optional error feedback (residual accumulation).
Uses NumPy for speed; integrates with SFEA training loop.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

class TopKSparsifier:
    def __init__(self, k_percent: float = 10.0, error_feedback: bool = True):
        assert 0 < k_percent <= 100, "k_percent must be in (0, 100]"
        self.k_percent = k_percent
        self.error_feedback = error_feedback
        self._residual: Dict[str, np.ndarray] = {}

    def _with_residual(self, key: str, grad: np.ndarray) -> np.ndarray:
        if not self.error_feedback:
            return grad
        r = self._residual.get(key)
        if r is None:
            return grad
        if r.shape != grad.shape:
            # reset if incompatible
            return grad
        return grad + r

    def compress(self, key: str, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (sparse_grad, mask) where mask is a boolean array indicating kept entries.
        """
        g = self._with_residual(key, grad)
        flat = g.ravel()
        n = flat.size
        k = max(1, int(np.ceil(n * (self.k_percent / 100.0))))

        # Threshold by magnitude
        if k >= n:
            mask = np.ones_like(flat, dtype=bool)
        else:
            idx = np.argpartition(np.abs(flat), -k)[-k:]
            mask = np.zeros_like(flat, dtype=bool)
            mask[idx] = True

        sparse = np.zeros_like(flat)
        sparse[mask] = flat[mask]

        sparse = sparse.reshape(g.shape)
        mask = mask.reshape(g.shape)

        if self.error_feedback:
            # residual stores the dropped components
            dropped = g - sparse
            self._residual[key] = dropped
        return sparse, mask

    def decompress(self, sparse: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Here 'sparse' already has zeros at dropped positions
        return sparse * (mask.astype(sparse.dtype) + (1 - mask.astype(sparse.dtype)))
