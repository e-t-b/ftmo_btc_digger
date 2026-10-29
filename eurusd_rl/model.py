"""Acteur-critique compact : un encodeur GRU + attention temporelle par unité de temps, branches contexte et compte."""
from __future__ import annotations

import torch
import torch.nn as nn

from .market import BAR_FEATS, CTX_FEATS
from .env import N_ACC

NEG = -1e9


class TFEncoder(nn.Module):
    def __init__(self, n_in: int, h: int):
        super().__init__()
        self.gru = nn.GRU(n_in, h, batch_first=True)
        self.att = nn.Linear(h, 1)

    def forward(self, x):                       # x : (B, L, F)
        out, hn = self.gru(x)
        w = torch.softmax(self.att(out).squeeze(-1), dim=1)          # (B, L)
        pooled = (w.unsqueeze(-1) * out).sum(1)
        return torch.cat([pooled, hn[-1]], dim=-1)                   # (B, 2h)


class ActorCritic(nn.Module):
    def __init__(self, tfs=("M5", "H1"), h: int = 32, fuse: int = 64, n_actions: int = 4, wait_bias: float = 0.0):
        super().__init__()
        self.tfs = tuple(tfs)
        self.enc = nn.ModuleDict({tf: TFEncoder(len(BAR_FEATS), h) for tf in self.tfs})
        self.ctx = nn.Sequential(nn.Linear(len(CTX_FEATS), h), nn.Tanh())
        self.acc = nn.Sequential(nn.Linear(N_ACC, h), nn.Tanh())
        n_in = 2 * h * len(self.tfs) + 2 * h
        self.body = nn.Sequential(nn.Linear(n_in, fuse), nn.Tanh(), nn.Linear(fuse, fuse), nn.Tanh())
        self.pi = nn.Linear(fuse, n_actions)
        self.v = nn.Linear(fuse, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)
        with torch.no_grad():
            self.pi.bias[0] = wait_bias          # exploration étalée : entrées et sorties rares au départ
        nn.init.orthogonal_(self.v.weight, 1.0); nn.init.zeros_(self.v.bias)

    def forward(self, obs: dict):
        z = [self.enc[tf](obs[tf]) for tf in self.tfs] + [self.ctx(obs["ctx"]), self.acc(obs["acc"])]
        x = self.body(torch.cat(z, -1))
        logits = self.pi(x).masked_fill(~obs["mask"], NEG)
        return logits, self.v(x).squeeze(-1)


def to_torch(obs: dict) -> dict:
    return {k: torch.from_numpy(v) for k, v in obs.items()}
