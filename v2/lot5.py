"""Lot 5 : Réintégration OPR (docs/protocole_lot5.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalogue as K
from .lot2 import _entry, _pack

SYMS = ("US100.cash", "US500.cash", "GER40.cash") + K.FX4
LOT = {"S27": SYMS, "S28": SYMS}
OPEN = {"us": ("America/New_York", 9, 30), "de": ("Europe/Berlin", 9, 0), "fx": ("Europe/London", 8, 0)}


def _trend(P, sid):
    key = "_trend_" + sid
    if key not in P:
        H = P["H1"]
        e50 = pd.Series(H["c"]).ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
        if sid == "S27":
            tr = np.sign(H["c"] - e50)
        else:
            e20 = pd.Series(H["c"]).ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
            tr = np.sign(e20 - e50)
        P[key] = (H["end"], np.nan_to_num(tr))
    return P[key]


def setups(P, sid):
    t, h, l, c = P["t"], P["h"], P["l"], P["c"]
    end, tr = _trend(P, sid)
    tz, hh, mm = OPEN[K.SYM_GROUP[P["symbol"]]]
    rows = []
    for day, r in enumerate(P["days"].to_dict("records")):
        op = K._ts(pd.Timestamp(r["date"]), tz, hh, mm)
        fc = int(r["fc"])
        oe = op + 900
        i0, i1 = np.searchsorted(t, op), np.searchsorted(t, oe)
        if i1 - i0 < 12:
            continue
        H, L = h[i0:i1].max(), l[i0:i1].min()
        b = np.searchsorted(end, oe, side="right") - 1
        if b < 0 or tr[b] == 0:
            continue
        d = int(tr[b])
        trapped, ext = False, None
        for k in K._m5_idx(t, oe, oe + 3 * 3600):
            if not trapped:
                if (d == 1 and c[k] > H) or (d == -1 and c[k] < L):
                    break                                   # cassure dans le sens de la tendance : journée annulée
                if (d == 1 and c[k] < L) or (d == -1 and c[k] > H):
                    trapped = True
                    ia = np.searchsorted(t, t[k] - 240)
                    ext = l[ia:k + 1].min() if d == 1 else h[ia:k + 1].max()
                continue
            ia = np.searchsorted(t, t[k] - 240)
            ext = min(ext, l[ia:k + 1].min()) if d == 1 else max(ext, h[ia:k + 1].max())
            if L <= c[k] <= H:
                j = K._next_bar(t, k)
                if j is None or t[j] >= fc:
                    break
                e = _entry(P, j, d)
                tp = H if d == 1 else L
                if d * (tp - e) <= 0:
                    break
                rows.append((day, j, d, ext, tp, fc, 0.0))
                break
    return _pack(rows)


def run(P, sid):
    return K.run(P, sid, {}, setups=setups(P, sid))
