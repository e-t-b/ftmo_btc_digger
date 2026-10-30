"""Lot 3 : cassures de Crabel S21-S23 (docs/protocole_lot3.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalogue as K
from .lot2 import _entry, _pack

SYMS = K.FX4 + K.IDX + K.BTC
LOT = {"S21": SYMS, "S22": SYMS, "S23": SYMS}


def _d1_features(P):
    if "_crabel" in P:
        return P["_crabel"]
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    D = K._bars(t, h, l, c, o, 86400)
    noise = np.minimum(D["h"] - D["o"], D["o"] - D["l"])
    stretch = 2 * pd.Series(noise).rolling(10, min_periods=10).mean().to_numpy()
    rng = D["h"] - D["l"]
    nr7 = rng <= pd.Series(rng).rolling(7, min_periods=7).min().to_numpy()
    days = P["days"]
    od = days["ws"].to_numpy(np.int64) // 86400
    j = np.searchsorted(D["start"] // 86400, od, side="left") - 1     # dernier jour UTC terminé
    ok = j >= 0
    jj = np.maximum(j, 0)
    st = np.where(ok, stretch[jj], np.nan)
    nr = np.where(ok, nr7[jj], False)
    P["_crabel"] = (st, nr)
    return P["_crabel"]


def setups(P, sid):
    t, o, c = P["t"], P["o"], P["c"]
    st, nr = _d1_features(P)
    rows = []
    for day, r in enumerate(P["days"].to_dict("records")):
        ws, we, fc = int(r["ws"]), int(r["we"]), int(r["fc"])
        if sid in ("S22", "S23") and not nr[day]:
            continue
        if sid in ("S21", "S22"):
            j0 = K._first_at(t, ws, 120)
            if j0 is None or not (st[day] > 0):
                continue
            O = o[j0]
            up, dn = O + st[day], O - st[day]
            i1 = np.searchsorted(t, we - 3600)
            for k in range(j0, i1):
                d = 1 if c[k] > up else (-1 if c[k] < dn else 0)
                if d:
                    j = K._next_bar(t, k)
                    if j is not None and t[j] < fc:
                        rows.append((day, j, d, dn if d == 1 else up, 0.0, fc, 0.0))
                    break
        else:
            i0, i1 = np.searchsorted(t, ws), np.searchsorted(t, ws + 1800)
            if i1 - i0 < 24:
                continue
            H, L = P["h"][i0:i1].max(), P["l"][i0:i1].min()
            for k in K._m5_idx(t, ws + 1800, ws + 1800 + 3 * 3600):
                d = 1 if c[k] > H else (-1 if c[k] < L else 0)
                if d:
                    j = K._next_bar(t, k)
                    if j is not None and t[j] < fc:
                        rows.append((day, j, d, L if d == 1 else H, 0.0, fc, 0.0))
                    break
    return _pack(rows)


def run(P, sid):
    return K.run(P, sid, {}, setups=setups(P, sid))
