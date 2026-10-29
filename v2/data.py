"""Chargement des M1 FTMO (prix BID + spread de barre) et conversion d'heure.

Heure serveur FTMO : GMT+2 l'hiver, GMT+3 pendant l'heure d'été US (journée serveur = 17:00 New York).
On convertit tout en UTC (int64, secondes) ; les heures locales (sessions, jour FTMO à Prague) sont dérivées."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "m1"


def server_offset_hours(t_server_s: np.ndarray) -> np.ndarray:
    """Décalage serveur->UTC par barre : 3 h si New York est en heure d'été ce jour-là, sinon 2 h.
    Le décalage d'un jour est déterminé par midi UTC de la date serveur (le changement d'heure a lieu à 2 h NY)."""
    day = (t_server_s // 86400).astype(np.int64)
    ud = np.unique(day)
    noon = pd.to_datetime(ud * 86400 + 43200, unit="s").tz_localize("UTC").tz_convert("America/New_York")
    dst = np.array([bool(x.dst()) for x in noon])
    off = 2 + dst.astype(np.int64)
    return off[np.searchsorted(ud, day)]


def load(symbol: str, data_dir: Path | None = None) -> dict:
    d = Path(data_dir or DATA_DIR)
    z = np.load(d / f"{symbol}.npz")
    ts = z["time"].astype(np.int64)
    order = np.argsort(ts, kind="stable")
    ts = ts[order]
    keep = np.r_[True, np.diff(ts) > 0]
    idx = order[keep]
    ts = ts[keep]
    t_utc = ts - server_offset_hours(ts) * 3600
    spec = json.loads((d / f"{symbol}_spec.json").read_text())
    return dict(symbol=symbol, t=t_utc, t_server=ts,
                o=z["open"][idx].astype(np.float64), h=z["high"][idx].astype(np.float64),
                l=z["low"][idx].astype(np.float64), c=z["close"][idx].astype(np.float64),
                spread_obs=z["spread"][idx].astype(np.float64), tv=z["tv"][idx].astype(np.int64), spec=spec)
