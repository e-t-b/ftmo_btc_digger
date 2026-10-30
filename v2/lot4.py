"""Lot 4 : S24 (après la Fed), S25 (veille de férié), S26 (journée de tendance). docs/protocole_lot4.md"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalogue as K
from .catalogue_run import calendars
from .lot2 import _entry, _pack

LOT = {"S24": ("US100.cash", "US500.cash"), "S25": ("US100.cash", "US500.cash", "GER40.cash"),
       "S26": ("US100.cash", "US500.cash", "GER40.cash", "BTCUSD")}
_CAL = None


def setups(P, sid):
    global _CAL
    if _CAL is None:
        _CAL = calendars()
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    D = P["days"]
    recs = D.to_dict("records")
    rows = []
    if sid == "S25":
        dates = pd.to_datetime(D["date"]).dt.normalize()
        have = set(dates)
        first, last = dates.iloc[0], dates.iloc[-1]
    for day, r in enumerate(recs):
        ws, fc, atr = int(r["ws"]), int(r["fc"]), r["atr"]
        if not (atr > 0):
            continue
        d0 = pd.Timestamp(r["date"]).normalize()
        if sid == "S24":
            if d0.date() not in _CAL["fomc"]:
                continue
            p0 = K._price_before(t, c, K._ts(d0, "America/New_York", 14, 0), 300)
            p1 = K._price_before(t, c, K._ts(d0, "America/New_York", 14, 30), 120)
            if p0 is None or p1 is None or p1 == p0:
                continue
            d = 1 if p1 > p0 else -1
            j = K._first_at(t, K._ts(d0, "America/New_York", 14, 30))
            if j is None or d * (_entry(P, j, d) - p0) <= 0:
                continue
            rows.append((day, j, d, p0, 0.0, fc, 0.0))
        elif sid == "S25":
            nxt = d0 + pd.Timedelta(days=1)
            while nxt.dayofweek >= 5:
                nxt += pd.Timedelta(days=1)
            if nxt > last or nxt in have:
                continue
            j = K._first_at(t, ws + 60)
            if j is None:
                continue
            e = _entry(P, j, 1)
            rows.append((day, j, 1, e - 0.5 * atr, 0.0, fc, 0.0))
        else:
            j0 = K._first_at(t, ws, 120)
            if j0 is None:
                continue
            O = o[j0]
            k = np.searchsorted(t, ws + 7200) - 1
            if k <= j0 or t[k] < ws + 7200 - 300:
                continue
            H, L, px = h[j0:k + 1].max(), l[j0:k + 1].min(), c[k]
            if H <= L or abs(px - O) < 0.5 * atr:
                continue
            pos = (px - L) / (H - L)
            d = 1 if (px > O and pos >= 0.8) else (-1 if (px < O and pos <= 0.2) else 0)
            if not d:
                continue
            j = K._next_bar(t, k)
            if j is None:
                continue
            rows.append((day, j, d, O, 0.0, fc, 0.0))
    return _pack(rows)


def run(P, sid):
    return K.run(P, sid, {}, setups=setups(P, sid))
