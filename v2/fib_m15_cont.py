"""Version « continuation » de la stratégie Fibonacci M15 des sessions.
Bougie de liquidité (haut − bas ≥ 0,25 × ATR14 D1) à l'ouverture de Tokyo, Londres, New York.
On attend dans les 2 h un retracement jusqu'au niveau 38,2 % de la bougie, puis on entre DANS LE SENS de la bougie
(au marché, minute suivant le toucher — prudent par rapport à un ordre limite).
Variantes : SL au niveau 61,8 % ou à l'extrémité de la bougie (100 %) ; TP pour un ratio 2:1 ; sinon sortie 16:50 NY.
Usage : python -m v2.fib_m15_cont"""
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


def setups(P, sl_level=0.618, rr=2.0, wait_bars=8):
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
            dr = 1 if Cl > O else -1                                   # sens de la bougie = sens du trade
            lvl = H - 0.382 * rng if dr == 1 else L + 0.382 * rng
            stop = H - sl_level * rng if dr == 1 else L + sl_level * rng
            k_end = np.searchsorted(t, min(a + 900 + wait_bars * 900, exit_t))
            for k in range(i1, k_end):
                touched = (l[k] + sc[k] <= lvl) if dr == 1 else (h[k] >= lvl)
                broke = (l[k] <= stop) if dr == 1 else (h[k] + sc[k] >= stop)
                if broke:
                    break                                               # invalidé avant l'entrée
                if touched:
                    j = k + 1
                    if j >= len(t) or t[j] - t[k] > 300 or t[j] >= exit_t:
                        break
                    e = o[j] + sc[j] + slip[j] if dr == 1 else o[j] - slip[j]
                    risk = dr * (e - stop)
                    if risk > 0:
                        rows.append((-1, j, dr, stop, e + dr * rr * risk, exit_t, 0.0))
                        meta[int(t[j])] = name
                    break
    return _pack(rows), meta


def run(P, zero_cost=False, **kw):
    if zero_cost:
        n = len(P["t"])
        P = dict(P, sc=np.zeros(n), slip=np.zeros(n), sslip=np.zeros(n), comm=np.zeros(n))
    S, meta = setups(P, **kw)
    old = C.MIN_STOP_COST_MULT
    C.MIN_STOP_COST_MULT = 0.0
    try:
        tr = K.run(P, "FIBK", {}, setups=S)["trades"]
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
    for vname, kw in (("SL 61,8 %", dict(sl_level=0.618)), ("SL 100 %", dict(sl_level=1.0))):
        for zc in (False, True):
            T = []
            for s, P in Ps.items():
                tr = run(P, zc, **kw).assign(symbol=s)
                if s.startswith("DK_"):
                    tr = tr[tr["entry_t"] < DK.END].assign(period="Dukascopy 2013-2021")
                else:
                    tr = tr.assign(period=np.where(tr["entry_t"] < cut, "FTMO → 2024-06", "FTMO 2024-07 → 2026-10"))
                T.append(tr)
            T = pd.concat(T, ignore_index=True)
            for per, g in T.groupby("period"):
                x = g["r_net"].to_numpy()
                out.append((vname, "sans coût" if zc else "avec coûts", per, "TOTAL", len(x), x.mean(), np.mean(x > 0),
                            g["cost_r"].median()))
                if not zc:
                    for key in ("session", "symbol"):
                        for v, gg in g.groupby(key):
                            x = gg["r_net"].to_numpy()
                            out.append((vname, "avec coûts", per, v, len(x), x.mean(), np.mean(x > 0), gg["cost_r"].median()))
    R = pd.DataFrame(out, columns=["variante", "coûts", "période", "détail", "trades", "R", "gagnants", "coût_R_médian"]).round(3)
    R.to_csv("reports_v2/fib_m15_cont.csv", index=False)
    pd.set_option("display.width", 200)
    print(R[R["détail"] == "TOTAL"].to_string(index=False))
    print(R[(R["détail"].isin(["Tokyo", "Londres", "NY"]))].to_string(index=False))


if __name__ == "__main__":
    main()
