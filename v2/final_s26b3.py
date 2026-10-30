"""Test final unique de S26-B3 sur la période finale FTMO (docs/protocole_final_S26B3.md).
Usage : python -m v2.final_s26b3"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from . import config as C
from . import ftmo, lot4, portfolio, s26x
from .catalogue_run import prepared
from .jalon_a import day_series, metrics, tstat

A = "2024-07-01"
B = "2026-10-28"
TA = int(pd.Timestamp(A).value // 10**9)
TB = int((pd.Timestamp(B) + pd.Timedelta(days=1)).value // 10**9)
SYMS = ("US500.cash", "US100.cash", "GER40.cash")


def collect(Ps, make):
    runs, parts = {}, []
    for s, P in Ps.items():
        r = K.run(P, "S26F", {}, setups=make(P))
        df = r["trades"]
        df = df[(df["entry_t"] >= TA) & (df["entry_t"] < TB)].copy()
        df["sid"], df["ci"] = "S26F", 0
        df["pkey"] = [("S26F", s, 0)] * len(df)
        runs[("S26F", s, 0)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])
        parts.append(df)
    return runs, pd.concat(parts, ignore_index=True)


def evaluate(Ps, runs, tr, risk):
    C.RISK["base"], C.RISK["reduced"] = risk, risk / 2
    paths = {k: (v["path_t"], v["path_u"]) for k, v in runs.items()}
    acc, days = portfolio.run(tr, paths, {s: P["spec"] for s, P in Ps.items()}, risk_frac=risk)
    days = day_series(days, A, B)
    M = metrics(acc, days)
    M["t_stat"] = round(tstat(acc["r_eff"].to_numpy()), 2)
    mc = ftmo.monte_carlo(days["R"].to_numpy(), days["worst_R"].to_numpy(), days["n"].to_numpy(), n_paths=20_000)
    C.RISK["base"], C.RISK["reduced"] = 0.005, 0.0025
    return acc, days, M, mc


def main():
    Ps = {s: prepared(s) for s in SYMS}
    runs, tr = collect(Ps, lambda P: s26x.setups(P, hours=(1, 2, 3)))
    acc, days, M, mc = evaluate(Ps, runs, tr, 0.0075)
    crit = {"1_expectancy>0": M["expectancy_r"] > 0, "2_PF>=1.1": (M["pf"] or 0) >= 1.1,
            "3_MC>=40%": mc["p_pass"] >= 0.40, "4_DD<=10%": -M["max_dd_pct"] <= 10}
    acc["hour"] = [None] * len(acc)
    D = {s: P["days"] for s, P in Ps.items()}
    acc["hour"] = [round((r.entry_t - D[r.symbol].at[int(r.day), "ws"]) / 3600) for r in acc.itertuples()]
    acc["sem"] = pd.to_datetime(acc["entry_t"], unit="s").dt.year.astype(str) + "-S" + \
        np.where(pd.to_datetime(acc["entry_t"], unit="s").dt.month <= 6, "1", "2")
    info = dict(per_symbol=acc.groupby("symbol")["r_eff"].agg(["size", "mean"]).round(3).reset_index().to_dict("records"),
                per_hour=acc.groupby("hour")["r_eff"].agg(["size", "mean"]).round(3).reset_index().to_dict("records"),
                per_semester=acc.groupby("sem")["r_eff"].agg(["size", "mean"]).round(3).reset_index().to_dict("records"))
    lot4._CAL = None
    runs0, tr0 = collect(Ps, lambda P: lot4.setups(P, "S26"))
    _, _, M0, mc0 = evaluate(Ps, runs0, tr0, 0.005)
    res = dict(window=[A, B], strategy="S26-B3", risk=0.0075, metrics=M, monte_carlo=mc, criteria=crit,
               decision="GO" if all(crit.values()) else "NO-GO", info=info,
               s26_original_info=dict(metrics=M0, monte_carlo=mc0))
    acc.drop(columns=["pkey"]).to_csv("reports_v2/final_s26b3_trades.csv", index=False)
    Path("reports_v2/final_s26b3.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("metrics", "monte_carlo", "criteria", "decision", "info")}, indent=1, default=str))
    print("S26 d'origine (info) :", {k: M0.get(k) for k in ("n", "expectancy_r", "pf", "t_stat", "total_pct", "max_dd_pct")},
          "MC", mc0["p_pass"])


if __name__ == "__main__":
    main()
