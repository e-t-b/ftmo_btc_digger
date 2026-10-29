"""Agrégation causale M1 -> M5/M15/M30/H1/H4/D1, en heure serveur (journée = 17:00 NY -> 17:00 NY).

Une bougie [début, fin) n'est utilisable qu'à partir de l'instant `fin` (heure serveur). La fonction
`last_closed` renvoie, pour chaque instant de décision T, l'indice de la dernière bougie dont fin <= T.
Aucun forward-fill d'une bougie en formation n'est possible avec cette interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TFS = {"M1": "1min", "M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1D"}


def aggregate(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    """m1 : colonnes t_server, open, high, low, close, tick_volume, spread_cost_pts (optionnel)."""
    t = pd.DatetimeIndex(m1["t_server"])
    start = t.floor(rule) if rule != "1D" else t.normalize()
    key = start.asi8
    # groupes consécutifs (les données sont triées)
    brk = np.r_[True, key[1:] != key[:-1]]
    gid = np.cumsum(brk) - 1
    first = np.flatnonzero(brk)
    last = np.r_[first[1:] - 1, len(key) - 1]
    hi = np.maximum.reduceat(m1["high"].to_numpy(), first)
    lo = np.minimum.reduceat(m1["low"].to_numpy(), first)
    out = pd.DataFrame({
        "start": start[first],
        "open": m1["open"].to_numpy()[first],
        "high": hi, "low": lo,
        "close": m1["close"].to_numpy()[last],
        "tick_volume": np.add.reduceat(m1["tick_volume"].to_numpy(), first),
        "n_m1": last - first + 1,
        "m1_first": first, "m1_last": last,
    })
    out["end"] = out["start"] + pd.Timedelta(rule)
    out["t_utc_end"] = pd.DatetimeIndex(m1["t_utc"].to_numpy()[last]) + pd.Timedelta("1min")
    return out


def last_closed(bar_end: np.ndarray, t_decision: np.ndarray) -> np.ndarray:
    """Indice de la dernière bougie close (fin <= t). -1 si aucune."""
    return np.searchsorted(np.asarray(bar_end), np.asarray(t_decision), side="right") - 1


def build_all(m1: pd.DataFrame, tfs=("M5", "M15", "M30", "H1", "H4", "D1")) -> dict[str, pd.DataFrame]:
    return {tf: aggregate(m1, TFS[tf]) for tf in tfs}
