"""S26b : S26 avec stop à mi-chemin (entrée/ouverture), filtres mouvement <= 1 ATR et ATR D1 >= 0,8 % du prix.
Usage : python -m v2.s26b   (entraînement seulement : Dukascopy → 2021-08, FTMO → 2024-06)"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from . import dukascopy as DK
from . import lot4
from .catalogue_run import evaluate, prepared
from .lot2 import _entry, _pack

FTMO_END = int(pd.Timestamp("2024-07-01").value // 10**9)


def setups(P, stop_frac=0.5, max_move=1.0, min_atr_rel=0.008):
    base = lot4.setups(P, "S26")
    D = P["days"]
    rows = []
    for i in range(len(base["j"])):
        day, j, d = int(base["day"][i]), int(base["j"][i]), int(base["d"][i])
        O = float(base["stop"][i])
        atr = D.at[day, "atr"]
        e = _entry(P, j, d)
        if max_move is not None and d * (P["c"][j - 1] - O) > max_move * atr:
            continue
        if min_atr_rel is not None and atr / O < min_atr_rel:
            continue
        stop = e - stop_frac * (e - O)
        rows.append((day, j, d, stop, 0.0, int(base["exit_t"][i]), 0.0))
    return _pack(rows)


def run_set(Ps, end, variant):
    runs, parts = {}, []
    for s, P in Ps.items():
        if variant == "S26":
            r = K.run(P, "S26", {}, setups=lot4.setups(P, "S26"))
        else:
            r = K.run(P, "S26b", {}, setups=setups(P, **variant))
        df = r["trades"]
        df = df[df["entry_t"] < end].copy()
        df["sid"], df["ci"] = "S26x", 0
        df["pkey"] = [("S26x", s, 0)] * len(df)
        runs[("S26x", s, 0)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])
        parts.append(df)
    return runs, pd.concat(parts, ignore_index=True)


def main():
    lot4._CAL = DK.calendars_dk()
    srcs = {"dukascopy": ({s: K.prepare(s) for s in DK.MAP}, DK.END),
            "ftmo": ({s: prepared(s) for s in ("US500.cash", "US100.cash", "GER40.cash")}, FTMO_END)}
    variants = {"S26 (origine)": "S26",
                "stop mi-chemin seul": dict(stop_frac=0.5, max_move=None, min_atr_rel=None),
                "filtres seuls": dict(stop_frac=1.0, max_move=1.0, min_atr_rel=0.008),
                "S26b (stop mi-chemin + filtres)": dict(stop_frac=0.5, max_move=1.0, min_atr_rel=0.008)}
    out = {}
    for name, v in variants.items():
        for src, (Ps, end) in srcs.items():
            runs, tr = run_set(Ps, end, v)
            start = str(min(pd.Timestamp(P["days"]["date"].iloc[0]) for P in Ps.values()).date())
            acc, days, M, mc = evaluate(tr, runs, Ps, start, n_paths=5000)
            yrs = (pd.Timestamp(int(tr["entry_t"].max()), unit="s") - pd.Timestamp(start)).days / 365.25
            out[f"{name} | {src}"] = dict(n=M["n"], R=M["expectancy_r"], t=M["t_stat"], pf=M["pf"], hit=M["hit"],
                                          pct_an=round(M["total_pct"] / yrs, 1), dd=M["max_dd_pct"],
                                          MC=round(mc["p_pass"], 3), jours_MC=mc.get("median_days_total"))
            print(name, "|", src, out[f"{name} | {src}"], flush=True)
    Path("reports_v2/s26b_train.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
