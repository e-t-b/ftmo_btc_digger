"""Backtest d'une stratégie explicite dans le MÊME environnement que l'agent (coûts, sizing, SL/TP, limites).

Règle : à chaque décision, si aucune position n'est ouverte et que la stratégie est active, entrer dans son sens avec
un stop à son invalidation (distance au moins égale au stop minimal de coûts ; signal ignoré si > max_stop_pips),
TP = take_profit_r x distance. Sortie : stop, TP ou clôture forcée. Comptes réinitialisés chaque mois (mode monthly)
pour mesurer l'espérance sans que l'arrêt global interne ne coupe l'historique.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd

from . import env as E
from .evaluate import LOG_COLS, REASONS, metrics, segments

MAX_STOP_PIPS = 40.0


def stop_distances(dirs: np.ndarray, stop_px: np.ndarray, close: np.ndarray, min_stop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.abs(close - stop_px)
    sd = np.maximum(np.nan_to_num(raw, nan=0.0), min_stop)
    ok = (dirs != 0) & np.isfinite(raw) & (sd <= MAX_STOP_PIPS * 1e-4)
    return np.where(ok, dirs, 0), np.where(ok, sd, 1.0)


def run_rule(mk, cfg: dict, lo: int, hi: int, dirs: np.ndarray, stop_dist: np.ndarray, mode: str = "monthly"):
    mk2 = dataclasses.replace(mk, dec_stop_dist=stop_dist)
    want = np.where(dirs > 0, 1, np.where(dirs < 0, 2, 0)).astype(np.int64)
    rows_dec, rows_seg, rows_eq = [], [], []
    trades = []
    cnt = {"blocked_entries": 0, "invalid_actions": 0, "violations": 0}
    segs = segments(mk2, lo, hi, mode)
    for si, (a, b) in enumerate(segs):
        v = E.VecEnv(mk2, cfg, 1, a, b, sequential=True, log_capacity=5000)
        act = np.zeros(1, np.int64)
        eq = np.empty(b - a)
        for t, d in enumerate(range(a, b)):
            act[0] = want[d] if v.I[0, E.I_POS] == 0 else 0
            v.step(act)
            eq[t] = v.F[0, E.S_EQ]
        rows_dec.append(np.arange(a, b)); rows_seg.append(np.full(b - a, si)); rows_eq.append(eq)
        L = pd.DataFrame(v.log[0, :v.nlog[0]], columns=LOG_COLS)
        L["segment"] = si
        trades.append(L)
        cnt["blocked_entries"] += int(v.I[0, E.I_NBLOCK]); cnt["violations"] += int(v.I[0, E.I_VIOL])
    tr = pd.concat(trades, ignore_index=True)
    for c in ("entry_dec", "exit_dec", "exit_m1", "side", "reason", "segment"):
        tr[c] = tr[c].astype(np.int64)
    tr["entry_time_utc"] = pd.DatetimeIndex(mk.dec_t_utc[tr["entry_dec"].to_numpy()])
    tr["exit_time_server"] = pd.to_datetime(mk.t_server_min[tr["exit_m1"].to_numpy()] * 60, unit="s")
    tr["reason"] = tr["reason"].map(REASONS)
    dec = np.concatenate(rows_dec)
    st = pd.DataFrame({"dec": dec, "segment": np.concatenate(rows_seg), "action": 0, "equity": np.concatenate(rows_eq),
                       "viol": 0, "day": mk.dec_day[dec]})
    cnt.update(n_segments=len(segs), mode=mode)
    return tr, st, cnt


def per_year(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {}
    y = tr["entry_time_utc"].dt.year
    g = tr.groupby(y)["R"]
    return {int(k): {"n": int(v.count()), "mean_R": round(float(v.mean()), 3), "sum_R": round(float(v.sum()), 1)}
            for k, v in g}
