"""Jalon A pour les hypothèses intraday H1-H3 (docs/protocole_jalon_A_intraday.md).

Usage : python -m v2.jalon_intraday            (période de développement)
        python -m v2.jalon_intraday --holdout H1 (seulement pour une hypothèse ayant rempli les critères 1-7)"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import ftmo, intraday
from .data import DATA_DIR
from .jalon_a import day_series, metrics, run_portfolio, semesters, tstat, ts

OUT = Path("reports_v2")
CACHE = Path("cache")


def load_prepared(symbols) -> dict:
    CACHE.mkdir(exist_ok=True)
    out = {}
    for s in symbols:
        if not (DATA_DIR / f"{s}.npz").exists():
            continue
        f = CACHE / f"intraday_{s}.pkl"
        if f.exists():
            out[s] = pickle.loads(f.read_bytes())
        else:
            out[s] = intraday.prepare(s)
            f.write_bytes(pickle.dumps(out[s]))
    return out


def grid_runs(Ps, h, spread_mult=1.0, slip_mult=1.0, only=None):
    runs = {}
    G = intraday.grid(h)
    for s, P in Ps.items():
        for ci, cfg in enumerate(G):
            if only is not None and ci not in only.get(s, ()):
                continue
            r = intraday.run(P, h, cfg, spread_mult, slip_mult)
            df = r["trades"]
            df["ci"] = ci
            df["pkey"] = [(s, ci)] * len(df)
            runs[(s, ci)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])
    return runs


def walk_forward(runs, h, symbols, end):
    G = intraday.grid(h)
    sel_rows, oos = [], []
    for s in symbols:
        for a, b, lab in semesters(C.INSTRUMENTS[s]["first_test"], end):
            best, best_t, best_n = None, -np.inf, 0
            for ci in range(len(G)):
                tr = runs[(s, ci)]["trades"]
                x = tr.loc[tr["exit_t"] < a, "r_net"].to_numpy()
                if len(x) < C.WF_MIN_TRADES:
                    continue
                tt = tstat(x)
                if tt > best_t:
                    best, best_t, best_n = ci, tt, len(x)
            traded = best is not None and best_t >= C.WF_MIN_T
            sel_rows.append(dict(symbol=s, semester=lab, a=a, b=b, ci=best,
                                 cfg=intraday.tag(G[best]) if best is not None else None,
                                 train_t=round(best_t, 2) if best is not None else None, train_n=best_n, traded=traded))
            if traded:
                tr = runs[(s, best)]["trades"]
                oos.append(tr[(tr["entry_t"] >= a) & (tr["entry_t"] < b)].assign(semester=lab, hyp=h))
    return pd.DataFrame(sel_rows), (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())


def robustness(runs, h, sel, end):
    G = intraday.grid(h)
    e = ts(end) + 86400
    out = {}
    for s in sel["symbol"].unique():
        cs = sel[(sel["symbol"] == s) & sel["traded"]]["ci"]
        if cs.empty:
            continue
        ci = Counter(cs.astype(int)).most_common(1)[0][0]

        def mean_r(c):
            tr = runs[(s, G.index(c))]["trades"]
            return float(tr.loc[tr["entry_t"] < e, "r_net"].mean())
        nb = {intraday.tag(n): round(mean_r(n), 4) for n in intraday.neighbours(h, G[ci])}
        own = round(mean_r(G[ci]), 4)
        out[s] = dict(cfg=intraday.tag(G[ci]), mean_r=own, neighbours=nb,
                      ok=bool(own > 0 and sum(v > 0 for v in nb.values()) > len(nb) / 2))
    return out


def evaluate(oos, runs, Ps, start, end, label, n_paths):
    acc, days = run_portfolio(oos, runs, Ps, start, end) if len(oos) else (pd.DataFrame(), pd.DataFrame())
    M = metrics(acc, days) if len(acc) else dict(n=0)
    if len(acc):
        r = acc["r_eff"].to_numpy()
        M["t_stat"] = round(tstat(r), 2)
        mc = ftmo.monte_carlo(days["R"].to_numpy(), days["worst_R"].to_numpy(), days["n"].to_numpy(), n_paths=n_paths)
        per_sym = acc.groupby("symbol").agg(n=("r_eff", "size"), exp_r=("r_eff", "mean"),
                                           pnl_pct=("pnl", lambda x: x.sum() / C.ACCOUNT * 100)).round(4)
        M["per_symbol"] = per_sym.reset_index().to_dict("records")
    else:
        mc = dict(p_pass=0.0)
    return acc, days, M, mc


def checks(M, M_st, mc, rob):
    cr = C.CRITERIA
    return {
        "1_trades>=300": M.get("n", 0) >= cr["min_trades"],
        "2_expectancy>0_t>=2": M.get("expectancy_r", -9) > 0 and (M.get("t_stat") or 0) >= 2.0,
        "3_PF>=1.3": (M.get("pf") or 0) >= cr["min_pf"],
        "4_maxDD<=6%": M.get("n", 0) > 0 and -M.get("max_dd_pct", -99) <= cr["max_dd"] * 100,
        "5_MC_pass>=50%": mc.get("p_pass", 0) >= cr["min_pass"],
        "6_stress_expectancy>0": M_st.get("expectancy_r", -9) > 0,
        "7_robustness": bool(rob) and all(v["ok"] for v in rob.values()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=20_000)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    end = C.DEV_END
    results, all_oos, all_runs, all_P = {}, [], {}, {}
    for h in ("H1", "H2", "H3"):
        Ps = load_prepared(C.HYP[h]["symbols"])
        all_P.update(Ps)
        runs = grid_runs(Ps, h)
        sel, oos = walk_forward(runs, h, list(Ps), end)
        sel.to_csv(OUT / f"wf_selection_{h}.csv", index=False)
        start = min(C.INSTRUMENTS[s]["first_test"] for s in Ps)
        acc, days, M, mc = evaluate(oos, runs, Ps, start, end, h, a.paths)
        only = {s: set(sel[(sel["symbol"] == s) & sel["traded"]]["ci"].astype(int)) for s in Ps}
        runs_st = grid_runs(Ps, h, 1.5, 2.0, only=only)
        oos_st = []
        for _, r in sel[sel["traded"]].iterrows():
            tr = runs_st[(r["symbol"], int(r["ci"]))]["trades"]
            oos_st.append(tr[(tr["entry_t"] >= r["a"]) & (tr["entry_t"] < r["b"])])
        oos_st = pd.concat(oos_st, ignore_index=True) if oos_st else pd.DataFrame()
        _, _, M_st, _ = evaluate(oos_st, runs_st, Ps, start, end, h + "_stress", 2000) if len(oos_st) else (0, 0, dict(n=0), {})
        rob = robustness(runs, h, sel, end)
        # diagnostic : moyenne de toute la grille sur la période de développement (en échantillon)
        e = ts(end) + 86400
        diag = {}
        for s in Ps:
            v = [runs[(s, ci)]["trades"] for ci in range(len(intraday.grid(h)))]
            v = [x[x["entry_t"] < e] for x in v]
            diag[s] = dict(n_med=int(np.median([len(x) for x in v])),
                           gross_mean=round(float(np.mean([x["r_gross"].mean() for x in v])), 4),
                           net_mean=round(float(np.mean([x["r_net"].mean() for x in v])), 4),
                           net_best=round(float(np.max([x["r_net"].mean() for x in v])), 4),
                           n_configs_pos=int(sum(x["r_net"].mean() > 0 for x in v)),
                           first_data=str(pd.Timestamp(int(Ps[s]["t"][0]), unit="s").date()))
        ck = checks(M, M_st, mc, rob)
        results[h] = dict(symbols=list(Ps), period=dict(start=start, end=end), metrics=M, stress=M_st,
                          monte_carlo=mc, robustness=rob, in_sample_grid=diag, checks=ck, dev_pass=all(ck.values()),
                          semesters_traded=int(sel["traded"].sum()), semesters_total=int(len(sel)))
        if len(acc):
            acc.drop(columns=["pkey"]).to_csv(OUT / f"oos_trades_dev_{h}.csv", index=False)
        print(h, json.dumps(dict(metrics={k: v for k, v in M.items() if k != "per_symbol"}, mc=mc, checks=ck), default=str), flush=True)
    (OUT / "jalon_a_intraday_dev.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
