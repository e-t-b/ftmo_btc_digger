"""S26 étendu (exploration sur l'entraînement uniquement) : plusieurs heures de décision, plus d'indices,
plafond par bloc d'indices corrélés. Période finale FTMO non utilisée.

Usage : python -m v2.s26x"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from . import config as C
from . import dukascopy as DK
from .catalogue_run import evaluate, prepared
from .lot2 import _entry, _pack

FTMO_END = int(pd.Timestamp("2024-07-01").value // 10**9)

# séances cash : (fuseau, ouverture h, m, fin de fenêtre h, m, fuseau clôture forcée, h, m)
K.GROUP["uk"] = ("Europe/London", 8, 0, 16, 30, "Europe/London", 16, 28)
K.GROUP["jp"] = ("Asia/Tokyo", 9, 0, 15, 0, "Asia/Tokyo", 14, 58)
for s, g in (("US30.cash", "us"), ("UK100.cash", "uk"), ("JP225.cash", "jp")):
    K.SYM_GROUP[s] = g
for s in ("US500.cash", "US100.cash", "US30.cash"):
    C.INSTRUMENTS[s]["bloc"] = "US"
C.INSTRUMENTS["UK100.cash"] = dict(C.INSTRUMENTS["GER40.cash"], floor_px=1.0, slip_px=1.0, stop_slip_px=3.0)
C.INSTRUMENTS["JP225.cash"] = dict(C.INSTRUMENTS["GER40.cash"], floor_px=5.0, slip_px=5.0, stop_slip_px=15.0, quote="USD")  # P&L suivi en R ; taille approx.

DK_MAP = {"DK_US500": "US500.cash", "DK_US100": "US100.cash", "DK_GER40": "GER40.cash",
          "DK_US30": "US30.cash", "DK_UK100": "UK100.cash", "DK_JP225": "JP225.cash"}


def _register_dk():
    for dk, ft in DK_MAP.items():
        if (DK.DK_DIR / f"{dk}.npz").exists():
            DK.MAP[dk] = ft
            C.INSTRUMENTS[dk] = dict(C.INSTRUMENTS[ft])
            K.SYM_GROUP[dk] = K.SYM_GROUP[ft]


def setups(P, hours=(2,), stop_frac=0.5, max_move=1.0, min_atr_rel=0.008):
    """Premier signal « journée de tendance » parmi les heures de décision (ouverture + h), un trade par jour."""
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    rows = []
    for day, r in enumerate(P["days"].to_dict("records")):
        ws, fc, atr = int(r["ws"]), int(r["fc"]), r["atr"]
        if not (atr > 0):
            continue
        j0 = K._first_at(t, ws, 120)
        if j0 is None:
            continue
        O = o[j0]
        if min_atr_rel is not None and atr / O < min_atr_rel:
            continue
        for hh in hours:
            k = np.searchsorted(t, ws + hh * 3600) - 1
            if k <= j0 or t[k] < ws + hh * 3600 - 300 or t[k] >= fc - 1800:
                continue
            H, L, px = h[j0:k + 1].max(), l[j0:k + 1].min(), c[k]
            if H <= L or abs(px - O) < 0.5 * atr:
                continue
            if max_move is not None and abs(px - O) > max_move * atr:
                continue
            pos = (px - L) / (H - L)
            d = 1 if (px > O and pos >= 0.8) else (-1 if (px < O and pos <= 0.2) else 0)
            if not d:
                continue
            j = K._next_bar(t, k)
            if j is None:
                continue
            e = _entry(P, j, d)
            rows.append((day, j, d, e - stop_frac * (e - O), 0.0, fc, 0.0))
            break
    return _pack(rows)


FTMO_START = int(pd.Timestamp("2021-09-01").value // 10**9)   # pas de recouvrement avec Dukascopy


def run_set(Ps, end, kw, start_ts=0):
    runs, parts = {}, []
    for s, P in Ps.items():
        r = K.run(P, "S26x", {}, setups=setups(P, **kw))
        df = r["trades"]
        df = df[(df["entry_t"] < end) & (df["entry_t"] >= start_ts)].copy()
        df["sid"], df["ci"] = "S26x", 0
        df["pkey"] = [("S26x", s, 0)] * len(df)
        runs[("S26x", s, 0)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])
        parts.append(df)
    return runs, pd.concat(parts, ignore_index=True)


def main():
    _register_dk()
    dk_syms = [s for s in DK_MAP if s in DK.MAP]
    ft_syms = [s for s in ("US500.cash", "US100.cash", "GER40.cash", "US30.cash", "UK100.cash", "JP225.cash")
               if (Path("data/m1") / f"{s}.npz").exists()]
    print("Dukascopy :", dk_syms, "| FTMO :", ft_syms, flush=True)
    srcs = {"dukascopy": ({s: K.prepare(s) for s in dk_syms}, DK.END),
            "ftmo": ({s: (prepared(s) if s in ("US500.cash", "US100.cash", "GER40.cash") else K.prepare(s)) for s in ft_syms},
                     FTMO_END)}
    base_syms = {"DK_US500", "DK_US100", "DK_GER40", "US500.cash", "US100.cash", "GER40.cash"}
    variants = {
        "A S26b (3 indices, 2 h)": (dict(hours=(2,)), True, 99),
        "B 3 indices, 1-2-3 h": (dict(hours=(1, 2, 3)), True, 99),
        "C 6 indices, 2 h": (dict(hours=(2,)), False, 99),
        "D 6 indices, 1-2-3 h": (dict(hours=(1, 2, 3)), False, 99),
        "E 6 indices, 1-2-3 h, bloc US max 2": (dict(hours=(1, 2, 3)), False, 2),
        "F 6 indices, 1-2-3 h, bloc US max 1": (dict(hours=(1, 2, 3)), False, 1),
    }
    out = {}
    for name, (kw, only_base, bloc_cap) in variants.items():
        C.RISK["max_same_bloc"] = bloc_cap
        for src, (Ps, end) in srcs.items():
            P2 = {s: P for s, P in Ps.items() if (not only_base) or s in base_syms}
            st = FTMO_START if src == "ftmo" else 0
            runs, tr = run_set(P2, end, kw, st)
            start = "2021-09-01" if src == "ftmo" else str(min(pd.Timestamp(P["days"]["date"].iloc[0]) for P in P2.values()).date())
            acc, days, M, mc = evaluate(tr, runs, P2, start, n_paths=5000)
            yrs = (pd.Timestamp(int(tr["entry_t"].max()), unit="s") - pd.Timestamp(start)).days / 365.25
            per = acc.groupby("symbol")["r_eff"].agg(["size", "mean"]).round(3)
            out[f"{name} | {src}"] = dict(n=M["n"], trades_an=round(M["n"] / yrs), R=M["expectancy_r"], t=M["t_stat"],
                                          pf=M["pf"], pct_an=round(M["total_pct"] / yrs, 1), dd=M["max_dd_pct"],
                                          MC=round(mc["p_pass"], 3), jours_MC=mc.get("median_days_total"),
                                          per_symbol=per.reset_index().to_dict("records"))
            o = out[f"{name} | {src}"]
            print(f"{name:40s} | {src:9s} n={o['n']:5d} ({o['trades_an']}/an) R={o['R']:+.3f} t={o['t']:.2f} PF={o['pf']} "
                  f"{o['pct_an']:+.1f}%/an DD={o['dd']}% MC={o['MC']:.1%} jours={o['jours_MC']}", flush=True)
    C.RISK["max_same_bloc"] = 99
    Path("reports_v2/s26x_train.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
