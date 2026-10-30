"""Première bougie H1 des sessions (Tokyo, Londres, NY), bougie de liquidité (range >= 0,25 ATR D1), TP au niveau 1.618.
A « retournement » : après bougie baissière → long, TP = bas + 1,618 × range (au-dessus du haut) ; inverse si haussière.
B « extension »    : dans le sens de la bougie, TP = 1,618 × range depuis le début de la bougie (au-delà de son extrémité).
Entrée à la clôture de la bougie (minute suivante), SL pour un ratio 2:1, sortie 16:50 NY sinon.
Usage : python -m v2.fib_ext"""
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


def setups(P, mode, f=1.618, rr=2.0, bar=3600):
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
            atr = atr_at(P, a)
            H, L, O, Cl = h[i0:i1].max(), l[i0:i1].min(), o[i0], c[i1 - 1]
            rng = H - L
            if not (atr > 0) or rng < 0.25 * atr or Cl == O:
                continue
            bear = Cl < O
            if mode == "A":
                dr = 1 if bear else -1
                tp = L + f * rng if bear else H - f * rng
            else:
                dr = -1 if bear else 1
                tp = H - f * rng if bear else L + f * rng
            j = i1
            if j >= len(t) or t[j] > a + bar + 300 or t[j] >= exit_t:
                continue
            e = o[j] + sc[j] + slip[j] if dr == 1 else o[j] - slip[j]
            if dr * (tp - e) <= 0:
                continue
            rows.append((-1, j, dr, e - dr * abs(tp - e) / rr, tp, exit_t, 0.0))
            meta[int(t[j])] = name
    return _pack(rows), meta


def run(P, mode, zero_cost=False):
    if zero_cost:
        n = len(P["t"])
        P = dict(P, sc=np.zeros(n), slip=np.zeros(n), sslip=np.zeros(n), comm=np.zeros(n))
    S, meta = setups(P, mode)
    old = C.MIN_STOP_COST_MULT
    C.MIN_STOP_COST_MULT = 0.0
    try:
        tr = K.run(P, "FIBX", {}, setups=S)["trades"]
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
    W = np.array(["stop", "gap", "trail", "tp", "time", "eod"])
    out = []
    for mode in ("A", "B"):
        for zc in (False, True):
            T = []
            for s, P in Ps.items():
                tr = run(P, mode, zc).assign(symbol=s)
                if s.startswith("DK_"):
                    tr = tr[tr["entry_t"] < DK.END].assign(period="Dukascopy 2013-2021")
                else:
                    tr = tr.assign(period=np.where(tr["entry_t"] < cut, "FTMO → 2024-06", "FTMO 2024-07 → 2026-10"))
                T.append(tr)
            T = pd.concat(T, ignore_index=True)
            for per, g in T.groupby("period"):
                x = g["r_net"].to_numpy()
                ex = (pd.Series(W[g["why"]]).value_counts(normalize=True).round(2).to_dict())
                out.append(dict(mode=mode, couts="sans" if zc else "avec", periode=per, detail="TOTAL", n=len(x),
                                R=round(x.mean(), 3), gagnants=round(np.mean(x > 0), 3), cout_R=round(g["cost_r"].median(), 3),
                                sorties=str(ex)))
                if not zc:
                    for key in ("session", "symbol"):
                        for v, gg in g.groupby(key):
                            y = gg["r_net"].to_numpy()
                            out.append(dict(mode=mode, couts="avec", periode=per, detail=v, n=len(y), R=round(y.mean(), 3),
                                            gagnants=round(np.mean(y > 0), 3), cout_R=round(gg["cost_r"].median(), 3)))
    R = pd.DataFrame(out)
    R.to_csv("reports_v2/fib_ext.csv", index=False)
    pd.set_option("display.width", 250)
    print(R[R.detail == "TOTAL"].to_string(index=False))
    print(R[R.detail.isin(["Tokyo", "Londres", "NY"])].drop(columns=["sorties"]).to_string(index=False))


if __name__ == "__main__":
    main()
