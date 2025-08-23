# -*- coding: utf-8 -*-
"""
SFEA (Sparse Federated Averaging) with Top-k + error feedback.

This module exposes a simple aggregate() to combine sparsified client updates.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from .topk_sparsifier import TopKSparsifier

class SFEAAggregator:
    def __init__(self, k_percent: float = 10.0, error_feedback: bool = True):
        self.sparsifier = TopKSparsifier(k_percent, error_feedback)

    def client_compress(self, client_id: str, grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.sparsifier.compress(client_id, grad)

    def aggregate(self, compressed_list: List[np.ndarray]) -> np.ndarray:
        """
        Simple mean of reconstructed dense tensors (already zero-padded).
        """
        if not compressed_list:
            raise ValueError("No client updates provided")
        stacked = np.stack(compressed_list, axis=0)
        return np.mean(stacked, axis=0)
