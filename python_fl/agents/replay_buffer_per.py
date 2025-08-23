# -*- coding: utf-8 -*-
"""
Prioritized Experience Replay (PER) buffer.
Simple tree-less implementation using arrays and proportional prioritization.
"""
from __future__ import annotations
import numpy as np

class PERBuffer:
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta0: float = 0.4):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta0
        self._next = 0
        self._size = 0

        self.obs = None
        self.act = None
        self.rew = None
        self.nxt = None
        self.done = None
        self.pri = np.zeros(self.capacity, dtype=np.float32) + 1e-6

    def _ensure(self, shape_obs):
        if self.obs is None:
            self.obs = np.zeros((self.capacity, shape_obs), dtype=np.float32)
            self.nxt = np.zeros((self.capacity, shape_obs), dtype=np.float32)
            self.act = np.zeros((self.capacity,), dtype=np.int64)
            self.rew = np.zeros((self.capacity,), dtype=np.float32)
            self.done = np.zeros((self.capacity,), dtype=np.float32)

    def push(self, obs, act, rew, nxt, done, td_err: float = 1.0):
        self._ensure(len(obs))
        idx = self._next
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew
        self.nxt[idx] = nxt
        self.done[idx] = float(done)
        self.pri[idx] = abs(td_err) + 1e-6
        self._next = (self._next + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int = 64):
        p = self.pri[:self._size] ** self.alpha
        p = p / p.sum()
        idx = np.random.choice(self._size, size=batch_size, p=p)
        # importance-sampling weights
        w = (self._size * p[idx]) ** (-self.beta)
        w = w / w.max()
        batch = (self.obs[idx], self.act[idx], self.rew[idx], self.nxt[idx], self.done[idx], w, idx)
        return batch, idx

    def update_priorities(self, idx, td_err):
        self.pri[idx] = np.abs(td_err).flatten() + 1e-6
