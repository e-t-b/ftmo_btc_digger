"""Variante de v2/fib_m15.py : entrée après deux bougies M15 consécutives de confirmation dans le sens du retracement
(dans les 2 h qui suivent la bougie de liquidité). TP au niveau Fibonacci, SL pour un ratio 2:1.
Usage : python -m v2.fib_m15_conf"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalogue as K
from . import config as C
from . import dukascopy as DK
from . import fib_m15 as F
from . import s26x
from .catalogue_run import prepared
from .lot2 import _pack, atr_at


def setups(P, f, n_conf=2, max_bars=8):
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
            if exit_t < a:
                continue
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, a + 900)
            if i1 - i0 < 12 or t[i0] > a + 120:
                continue
            atr = atr_at(P, a)
            H, L, O, Cl = h[i0:i1].max(), l[i0:i1].min(), o[i0], c[i1 - 1]
            rng = H - L
            if not (atr > 0) or rng < 0.25 * atr or Cl == O:
                continue
            dr = 1 if Cl < O else -1                    # sens du retracement attendu
            tp = L + f * rng if dr == 1 else H - f * rng
            streak = 0
            for b in range(1, max_bars + 1):
                ba = a + b * 900
                j0, j1 = np.searchsorted(t, ba), np.searchsorted(t, ba + 900)
                if j1 - j0 < 10 or t[j0] > ba + 120:
                    break
                bo, bc = o[j0], c[j1 - 1]
                streak = streak + 1 if dr * (bc - bo) > 0 else 0
                if streak >= n_conf:
                    j = j1
                    if j >= len(t) or t[j] > ba + 900 + 300 or t[j] >= exit_t:
                        break
                    e = o[j] + sc[j] + slip[j] if dr == 1 else o[j] - slip[j]
                    if dr * (tp - e) > 0:
                        rows.append((-1, j, dr, e - dr * abs(tp - e) / 2, tp, exit_t, 0.0))
                        meta[int(t[j])] = name
                    break
    return _pack(rows), meta


def run(P, f, zero_cost=False):
    if zero_cost:
        n = len(P["t"])
        P = dict(P, sc=np.zeros(n), slip=np.zeros(n), sslip=np.zeros(n), comm=np.zeros(n))
    S, meta = setups(P, f)
    old = C.MIN_STOP_COST_MULT
    C.MIN_STOP_COST_MULT = 0.0
    try:
        tr = K.run(P, "FIBC", {}, setups=S)["trades"]
    finally:
        C.MIN_STOP_COST_MULT = old
    tr["session"] = tr["entry_t"].map(meta)
    return tr


def main():
    s26x._register_dk()
    syms = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "US500.cash", "US100.cash", "GER40.cash", "US30.cash",
            "UK100.cash", "JP225.cash", "BTCUSD"]
    cat = {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "US500.cash", "US100.cash", "GER40.cash", "BTCUSD"}
    Ps = {s: (prepared(s) if s in cat else K.prepare(s)) for s in syms}
    Ps.update({s: K.prepare(s) for s in s26x.DK_MAP if s in DK.MAP})
    cut = int(pd.Timestamp("2024-07-01").value // 10**9)
    out = []
    for zc in (False, True):
        T = []
        for s, P in Ps.items():
            tr = run(P, 0.382, zc).assign(symbol=s)
            if s.startswith("DK_"):
                tr = tr[tr["entry_t"] < DK.END].assign(period="Dukascopy 2013-2021")
            else:
                tr = tr.assign(period=np.where(tr["entry_t"] < cut, "FTMO → 2024-06", "FTMO 2024-07 → 2026-10"))
            T.append(tr)
        T = pd.concat(T, ignore_index=True)
        tag = "SANS COÛT" if zc else "avec coûts"
        for per, g in T.groupby("period"):
            x = g["r_net"].to_numpy()
            out.append((tag, per, "TOTAL", len(x), x.mean(), np.mean(x > 0)))
            for sess, gg in g.groupby("session"):
                x = gg["r_net"].to_numpy()
                out.append((tag, per, sess, len(x), x.mean(), np.mean(x > 0)))
        if not zc:
            T.drop(columns=[], errors="ignore").to_csv("/tmp/fibc.csv", index=False)
    R = pd.DataFrame(out, columns=["coûts", "période", "session", "trades", "R", "gagnants"])
    R["R"] = R["R"].round(3)
    R["gagnants"] = R["gagnants"].round(3)
    R.to_csv("reports_v2/fib_m15_conf.csv", index=False)
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
