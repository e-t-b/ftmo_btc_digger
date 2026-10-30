"""Stratégie Fibonacci sur la première bougie H1 des sessions (forex), avec un filtre « bougie de liquidité » plus sélectif.
Filtres : range >= 0,5 × ATR14 D1 ; ou range >= 2 × ATR14 H1 (bougie anormalement grande pour son unité de temps).
Modes : R = retracement (TP au niveau f depuis l'extrémité, contre la bougie) ; A = retournement à 1.618 ; B = extension.
Entrée à la clôture de la bougie, SL pour 2:1, sortie 16:50 NY sinon.  Usage : python -m v2.fib_sel"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalogue as K
from . import config as C
from . import fib_m15 as F
from .catalogue_run import prepared
from .jalon_a import tstat
from .lot2 import _pack, atr_at

FXS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")


def atr_h1_at(P, ts):
    end = P["H1"]["end"]
    k = np.searchsorted(end, ts, side="right") - 1
    return float(P["H1"]["atr"][k]) if k >= 0 else np.nan


def setups(P, filt, mode, f, rr=2.0, bar=3600):
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    sc, slip = P["sc"], P["slip"]
    rows, meta = [], {}
    d0 = pd.Timestamp(int(t[0]), unit="s").normalize()
    d1 = pd.Timestamp(int(t[-1]), unit="s").normalize()
    for d in pd.date_range(d0, d1, freq="D"):
        if d.dayofweek >= 5:
            continue
        exit_t = K._ts(d, "America/New_York", 16, 50)
        for name, (tz, hh, mm) in F.SESSIONS.items():
            a = K._ts(d, tz, hh, mm)
            if exit_t < a + bar:
                continue
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, a + bar)
            if i1 - i0 < 0.8 * bar / 60 or t[i0] > a + 120:
                continue
            H, L, O, Cl = h[i0:i1].max(), l[i0:i1].min(), o[i0], c[i1 - 1]
            rng = H - L
            if Cl == O:
                continue
            if filt == "0.5 ATR D1":
                ref = atr_at(P, a)
                if not (ref > 0) or rng < 0.5 * ref:
                    continue
            else:
                ref = atr_h1_at(P, a)                       # ATR H1 connue avant l'ouverture de la bougie
                if not (ref > 0) or rng < 2.0 * ref:
                    continue
            bear = Cl < O
            if mode == "R":
                dr, tp = (1, L + f * rng) if bear else (-1, H - f * rng)
            elif mode == "A":
                dr, tp = (1, L + f * rng) if bear else (-1, H - f * rng)
            else:
                dr, tp = (-1, H - f * rng) if bear else (1, L + f * rng)
            j = i1
            if j >= len(t) or t[j] > a + bar + 300 or t[j] >= exit_t:
                continue
            e = o[j] + sc[j] + slip[j] if dr == 1 else o[j] - slip[j]
            if dr * (tp - e) <= 0:
                continue
            rows.append((-1, j, dr, e - dr * abs(tp - e) / rr, tp, exit_t, 0.0))
            meta[int(t[j])] = name
    return _pack(rows), meta


def run(P, filt, mode, f, zero_cost=False, rr=2.0):
    if zero_cost:
        n = len(P["t"])
        P = dict(P, sc=np.zeros(n), slip=np.zeros(n), sslip=np.zeros(n), comm=np.zeros(n))
    S, meta = setups(P, filt, mode, f, rr=rr)
    old = C.MIN_STOP_COST_MULT
    C.MIN_STOP_COST_MULT = 0.0
    try:
        tr = K.run(P, "FIBS", {}, setups=S)["trades"]
    finally:
        C.MIN_STOP_COST_MULT = old
    tr["session"] = tr["entry_t"].map(meta)
    return tr


def main():
    Ps = {s: prepared(s) for s in FXS}
    cut = int(pd.Timestamp("2024-07-01").value // 10**9)
    ndays = len(pd.bdate_range("2015-01-05", "2026-10-28"))
    rows = []
    for filt in ("0.5 ATR D1", "2 ATR H1"):
        for vname, mode, f in (("retracement 38,2 %", "R", 0.382), ("retournement 1.618", "A", 1.618),
                               ("extension 1.618", "B", 1.618), ("extension 2.618", "B", 2.618)):
            for zc in (False, True):
                T = pd.concat([run(P, filt, mode, f, zc).assign(symbol=s) for s, P in Ps.items()], ignore_index=True)
                T["periode"] = np.where(T["entry_t"] < cut, "2015 → 2024-06", "2024-07 → 2026-10")
                tot = dict(filtre=filt, variante=vname, couts="sans" if zc else "avec", periode="TOTAL", n=len(T),
                           par_jour_4paires=round(len(T) / ndays, 2), R=round(T["r_net"].mean(), 3),
                           t=round(tstat(T["r_net"].to_numpy()), 2), gagnants=round(float((T["r_net"] > 0).mean()), 3),
                           cout_R=round(T["cost_r"].median(), 3))
                rows.append(tot)
                for per, g in T.groupby("periode"):
                    x = g["r_net"].to_numpy()
                    rows.append(dict(filtre=filt, variante=vname, couts=tot["couts"], periode=per, n=len(x),
                                     R=round(x.mean(), 3), t=round(tstat(x), 2), gagnants=round(np.mean(x > 0), 3)))
                print(tot, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv("reports_v2/fib_sel.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
