"""Stratégie proposée par l'utilisateur : retracement Fibonacci de la première bougie M15 des sessions.

Pour chaque instrument et chaque session (Tokyo 09:00 Asia/Tokyo, Londres 08:00 Europe/London, New York 09:30
America/New_York) : première bougie M15. Si (haut − bas) ≥ 0,25 × ATR14 D1 (« bougie de liquidité ») :
- bougie baissière → long à l'ouverture de la minute suivante, TP = bas + f × (haut − bas) ;
- bougie haussière → short, TP = haut − f × (haut − bas) ;
- SL à la moitié de la distance au TP (ratio 2:1) ; sinon sortie à 16:50 New York.
f = 0,382 (niveau standard) et 0,328 (valeur donnée). Pas de filtre de coût minimum (règle fidèle).
Usage : python -m v2.fib_m15"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from . import config as C
from . import dukascopy as DK
from . import s26x
from .catalogue_run import prepared
from .jalon_a import tstat
from .lot2 import _pack, atr_at

SESSIONS = {"Tokyo": ("Asia/Tokyo", 9, 0), "Londres": ("Europe/London", 8, 0), "NY": ("America/New_York", 9, 30)}
PERIODS = {"FTMO 2015/2021 → 2024-06": ("1970-01-01", "2024-07-01"), "FTMO 2024-07 → 2026-10": ("2024-07-01", "2026-10-29")}


def setups(P, f, sessions=SESSIONS, bar=900):
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    sc, slip = P["sc"], P["slip"]
    rows, meta = [], []
    d0 = pd.Timestamp(int(t[0]), unit="s").normalize()
    d1 = pd.Timestamp(int(t[-1]), unit="s").normalize()
    for d in pd.date_range(d0, d1, freq="D"):
        if d.dayofweek >= 5:
            continue
        exit_t = K._ts(d, "America/New_York", 16, 50)
        for name, (tz, hh, mm) in sessions.items():
            a = K._ts(d, tz, hh, mm)
            if name == "Tokyo" and exit_t < a:
                continue
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, a + bar)
            if i1 - i0 < 0.8 * bar / 60 or t[i0] > a + 120:
                continue
            atr = atr_at(P, a)
            H, L, O, Cl = h[i0:i1].max(), l[i0:i1].min(), o[i0], c[i1 - 1]
            rng = H - L
            if not (atr > 0) or rng < 0.25 * atr or Cl == O:
                continue
            j = i1
            if j >= len(t) or t[j] > a + bar + 300 or t[j] >= exit_t:
                continue
            if Cl < O:
                dr, tp = 1, L + f * rng
                e = o[j] + sc[j] + slip[j]
            else:
                dr, tp = -1, H - f * rng
                e = o[j] - slip[j]
            if dr * (tp - e) <= 0:
                continue
            stop = e - dr * (abs(tp - e) / 2)
            rows.append((-1, j, dr, stop, tp, exit_t, 0.0))
            meta.append((int(t[j]), name))
    return _pack(rows), dict(meta)


def run(P, f, bar=900, zero_cost=False):
    if zero_cost:
        n = len(P["t"])
        P = dict(P, sc=np.zeros(n), slip=np.zeros(n), sslip=np.zeros(n), comm=np.zeros(n))
    S, meta = setups(P, f, bar=bar)
    old = C.MIN_STOP_COST_MULT
    C.MIN_STOP_COST_MULT = 0.0
    try:
        r = K.run(P, "FIB", {}, setups=S)
    finally:
        C.MIN_STOP_COST_MULT = old
    tr = r["trades"]
    tr["session"] = tr["entry_t"].map(meta)
    return tr


def summary(x):
    return dict(n=int(len(x)), R=round(float(x.mean()), 3) if len(x) else None,
                t=round(tstat(x), 2) if len(x) > 2 else None, win=round(float(np.mean(x > 0)), 3) if len(x) else None)


def main():
    s26x._register_dk()
    ftmo_syms = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "US500.cash", "US100.cash", "GER40.cash", "US30.cash",
                 "UK100.cash", "JP225.cash", "BTCUSD"]
    cat = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "US500.cash", "US100.cash", "GER40.cash", "BTCUSD"}
    Ps = {s: (prepared(s) if s in cat else K.prepare(s)) for s in ftmo_syms}
    dks = {s: K.prepare(s) for s in s26x.DK_MAP if s in DK.MAP}
    rows = []
    for f in (0.382, 0.328):
        allt = []
        for s, P in Ps.items():
            tr = run(P, f).assign(symbol=s, src="FTMO")
            allt.append(tr)
        for s, P in dks.items():
            tr = run(P, f).assign(symbol=s, src="Dukascopy")
            tr = tr[tr["entry_t"] < DK.END]
            allt.append(tr)
        T = pd.concat(allt, ignore_index=True)
        T["period"] = np.where(T["src"] == "Dukascopy", "Dukascopy 2013 → 2021-08",
                               np.where(T["entry_t"] < int(pd.Timestamp("2024-07-01").value // 10**9),
                                        "FTMO → 2024-06", "FTMO 2024-07 → 2026-10"))
        T["cost_R"] = T["cost_r"]
        for per, g in T.groupby("period"):
            rows.append(dict(f=f, scope="TOTAL", period=per, **summary(g["r_net"].to_numpy()),
                             gross=round(float(g["r_gross"].mean()), 3), cost=round(float(g["cost_R"].mean()), 3)))
            for (sess), gg in g.groupby("session"):
                rows.append(dict(f=f, scope=f"session {sess}", period=per, **summary(gg["r_net"].to_numpy()),
                                 gross=round(float(gg["r_gross"].mean()), 3), cost=round(float(gg["cost_R"].mean()), 3)))
            for sym, gg in g.groupby("symbol"):
                rows.append(dict(f=f, scope=sym, period=per, **summary(gg["r_net"].to_numpy()),
                                 gross=round(float(gg["r_gross"].mean()), 3), cost=round(float(gg["cost_R"].mean()), 3)))
        why = T.groupby("why")["r_net"].size().to_dict()
        rows.append(dict(f=f, scope="raisons de sortie", period="toutes", note=str({K.np.array(['stop','gap','trail','tp','time','eod'])[k]: v for k, v in why.items()})))
    R = pd.DataFrame(rows)
    R.to_csv("reports_v2/fib_m15.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
