# -*- coding: utf-8 -*-
"""
Dueling DQN agent (PyTorch) for offloading decisions.
Action space example: {0: local, 1: offload}.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

@dataclass
class DQNConfig:
    obs_dim: int = 3     # e.g., [size_mb, cycles_giga, est_rate_mbps]
    n_actions: int = 2
    hidden: int = 128
    lr: float = 1e-3
    gamma: float = 0.99
    target_update: int = 1000

class DuelingQ(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.val = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        z = self.fc(x)
        v = self.val(z)
        a = self.adv(z)
        # Q = V + (A - mean(A))
        return v + (a - a.mean(dim=1, keepdim=True))

class DuelingDQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.q = DuelingQ(cfg.obs_dim, cfg.n_actions, cfg.hidden)
        self.tgt = DuelingQ(cfg.obs_dim, cfg.n_actions, cfg.hidden)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.step_i = 0

    def act(self, obs: np.ndarray, eps: float = 0.05) -> int:
        if np.random.rand() < eps:
            return np.random.randint(self.cfg.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = self.q(x)
            return int(torch.argmax(q, dim=1).item())

    def update(self, batch: Tuple[np.ndarray, ...], gamma: float):
        obs, act, rew, nxt, done, w, idx = batch
        obs = torch.tensor(obs, dtype=torch.float32)
        nxt = torch.tensor(nxt, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.int64)
        rew = torch.tensor(rew, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        w = torch.tensor(w, dtype=torch.float32)

        q = self.q(obs).gather(1, act.view(-1,1)).squeeze(1)
        with torch.no_grad():
            nxt_q = self.tgt(nxt).max(dim=1)[0]
            tgt = rew + gamma * (1.0 - done) * nxt_q

        # PER importance-sampling weights
        loss = ((q - tgt) ** 2) * w
        loss = loss.mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.step_i += 1
        if self.step_i % self.cfg.target_update == 0:
            self.tgt.load_state_dict(self.q.state_dict())

        td_error = (q - tgt).detach().cpu().numpy()
        return float(loss.item()), td_error
