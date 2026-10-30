"""Jalon A : walk-forward, portefeuille, stress, robustesse, Monte Carlo FTMO (protocole §5-8).

Usage : python -m v2.jalon_a            (période de développement uniquement)
        python -m v2.jalon_a --holdout  (refusé si les critères de développement ne sont pas remplis)"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import ftmo, portfolio, prep, sim

OUT = Path("reports_v2")
CACHE = Path("cache")


def ts(x: str) -> int:
    return int(pd.Timestamp(x).value // 10**9)


def load_prepared() -> dict:
    CACHE.mkdir(exist_ok=True)
    out = {}
    for s in C.BREAKOUT_SYMBOLS:
        f = CACHE / f"{s}.pkl"
        if f.exists():
            out[s] = pickle.loads(f.read_bytes())
        else:
            out[s] = prep.prepare(s)
            f.write_bytes(pickle.dumps(out[s]))
    return out


def grid_runs(Ps: dict, spread_mult=1.0, slip_mult=1.0, tp_r=0.0, only: dict | None = None) -> dict:
    """runs[(sym, ci)] = dict(trades DataFrame, path_t, path_u)."""
    runs = {}
    for s, P in Ps.items():
        for ci, cfg in enumerate(C.GRID):
            if only is not None and ci not in only.get(s, ()):
                continue
            r = sim.run(P, cfg, tp_r=tp_r, spread_mult=spread_mult, slip_mult=slip_mult)
            df = pd.DataFrame({k: v for k, v in r["trades"].items() if k != "symbol"})
            df["symbol"] = s
            df["ci"] = ci
            df["pkey"] = [(s, ci)] * len(df)
            runs[(s, ci)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])
    return runs


def semesters(first: str, end: str) -> list[tuple[int, int, str]]:
    out = []
    a = pd.Timestamp(first)
    e = pd.Timestamp(end) + pd.Timedelta(days=1)
    while a < e:
        b = min(a + pd.DateOffset(months=6), e)
        out.append((ts(str(a)), ts(str(b)), f"{a.year}-S{1 if a.month <= 6 else 2}"))
        a = a + pd.DateOffset(months=6)
    return out


def tstat(r: np.ndarray) -> float:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(len(r)))


def walk_forward(runs: dict, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sel_rows, oos = [], []
    for s in C.BREAKOUT_SYMBOLS:
        ic = C.INSTRUMENTS[s]
        for a, b, lab in semesters(ic["first_test"], end):
            best, best_t, best_n = None, -np.inf, 0
            for ci in range(len(C.GRID)):
                tr = runs[(s, ci)]["trades"]
                x = tr.loc[tr["exit_t"] < a, "r_net"].to_numpy()
                if len(x) < C.WF_MIN_TRADES:
                    continue
                tt = tstat(x)
                if tt > best_t:
                    best, best_t, best_n = ci, tt, len(x)
            traded = best is not None and best_t >= C.WF_MIN_T
            sel_rows.append(dict(symbol=s, semester=lab, ci=best, cfg=C.cfg_tag(C.GRID[best]) if best is not None else None,
                                 train_t=round(best_t, 2) if best is not None else None, train_n=best_n, traded=traded))
            if traded:
                tr = runs[(s, best)]["trades"]
                oos.append(tr[(tr["entry_t"] >= a) & (tr["entry_t"] < b)].assign(semester=lab))
    return pd.DataFrame(sel_rows), (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())


def metrics(acc: pd.DataFrame, days: pd.DataFrame) -> dict:
    if acc.empty:
        return dict(n=0)
    r = acc["r_eff"].to_numpy()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(days["pnl"].to_numpy())
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    dd_close = float((eq - peak).min() / C.ACCOUNT)
    intraday = np.r_[0.0, eq[:-1]] + days["worst"].to_numpy()
    dd_intra = float((intraday - np.maximum(peak, 0)).min() / C.ACCOUNT)
    return dict(n=int(len(r)), expectancy_r=round(float(r.mean()), 4), pf=round(float(gp / gl), 3) if gl > 0 else None,
                hit=round(float((r > 0).mean()), 3), total_pct=round(float(acc["pnl"].sum() / C.ACCOUNT * 100), 2),
                max_dd_pct=round(dd_close * 100, 2), max_dd_intraday_pct=round(dd_intra * 100, 2),
                worst_day_pct=round(float(days["pnl"].min() / C.ACCOUNT * 100), 2),
                worst_intraday_day_pct=round(float(days["worst"].min() / C.ACCOUNT * 100), 2),
                day_stops=int(days["halted"].sum()))


def day_series(days: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Toutes les journées ouvrées (jours FTMO) de la fenêtre, journées sans trade à zéro."""
    cal = pd.bdate_range(start, end).values.astype("datetime64[D]")
    d = days.set_index("fday")
    out = pd.DataFrame(index=pd.Index(cal, name="fday"))
    for k in ("R", "worst_R", "n", "pnl", "worst"):
        out[k] = d[k].reindex(cal).fillna(0.0).to_numpy() if len(d) else 0.0
    out["halted"] = d["halted"].reindex(cal).fillna(False).to_numpy().astype(bool) if len(d) else False
    return out.reset_index()


def run_portfolio(oos: pd.DataFrame, runs: dict, Ps: dict, start: str, end: str):
    paths = {k: (v["path_t"], v["path_u"]) for k, v in runs.items()}
    specs = {s: P["spec"] for s, P in Ps.items()}
    acc, days = portfolio.run(oos, paths, specs)
    return acc, day_series(days, start, end)


def robustness(runs: dict, sel: pd.DataFrame, end: str) -> dict:
    out = {}
    e = ts(end) + 86400
    for s in C.BREAKOUT_SYMBOLS:
        cs = sel[(sel["symbol"] == s) & sel["traded"]]["ci"]
        if cs.empty:
            continue
        ci = Counter(cs.astype(int)).most_common(1)[0][0]
        cfg = C.GRID[ci]

        def mean_r(c):
            j = C.GRID.index(c)
            tr = runs[(s, j)]["trades"]
            return float(tr.loc[tr["entry_t"] < e, "r_net"].mean())
        nb = {C.cfg_tag(n): round(mean_r(n), 4) for n in C.neighbours(cfg)}
        own = round(mean_r(cfg), 4)
        out[s] = dict(cfg=C.cfg_tag(cfg), mean_r=own, neighbours=nb,
                      ok=bool(own > 0 and sum(v > 0 for v in nb.values()) > len(nb) / 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--paths", type=int, default=20_000)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    Ps = load_prepared()
    runs = grid_runs(Ps)
    end = C.DEV_END
    sel, oos = walk_forward(runs, end)
    sel.to_csv(OUT / "wf_selection.csv", index=False)
    start = min(C.INSTRUMENTS[s]["first_test"] for s in C.BREAKOUT_SYMBOLS)
    acc, days = run_portfolio(oos, runs, Ps, start, end)
    acc.drop(columns=["pkey"]).to_csv(OUT / "oos_trades_dev.csv", index=False)
    days.to_csv(OUT / "oos_days_dev.csv", index=False)
    M = metrics(acc, days)
    per_sym = {s: dict(n=int((g := acc[acc["symbol"] == s])["r_eff"].size),
                       expectancy_r=round(float(g["r_eff"].mean()), 4) if len(g) else None,
                       pnl_pct=round(float(g["pnl"].sum() / C.ACCOUNT * 100), 2))
               for s in C.BREAKOUT_SYMBOLS}
    acc["year"] = pd.to_datetime(acc["entry_t"], unit="s").dt.year
    per_year = acc.groupby("year").agg(n=("r_eff", "size"), exp_r=("r_eff", "mean"),
                                       pnl_pct=("pnl", lambda x: x.sum() / C.ACCOUNT * 100)).round(3)
    # stress des coûts sur les configurations sélectionnées
    only = {s: set(sel[(sel["symbol"] == s) & sel["traded"]]["ci"].astype(int)) for s in C.BREAKOUT_SYMBOLS}
    runs_st = grid_runs(Ps, spread_mult=1.5, slip_mult=2.0, only=only)
    oos_st = []
    for _, r in sel[sel["traded"]].iterrows():
        tr = runs_st[(r["symbol"], int(r["ci"]))]["trades"]
        a_, b_ = [x for x in semesters(C.INSTRUMENTS[r["symbol"]]["first_test"], end) if x[2] == r["semester"]][0][:2]
        oos_st.append(tr[(tr["entry_t"] >= a_) & (tr["entry_t"] < b_)])
    oos_st = pd.concat(oos_st, ignore_index=True) if oos_st else pd.DataFrame()
    acc_st, days_st = run_portfolio(oos_st, runs_st, Ps, start, end) if len(oos_st) else (pd.DataFrame(), days)
    M_st = metrics(acc_st, days_st)
    rob = robustness(runs, sel, end)
    mc = ftmo.monte_carlo(days["R"].to_numpy(), days["worst_R"].to_numpy(), days["n"].to_numpy(), n_paths=a.paths)
    # information : variante objectif fixe 4R, configuration la plus retenue, toute la période de développement
    ref = {}
    for s, rb in rob.items():
        ci = C.GRID.index(next(c for c in C.GRID if C.cfg_tag(c) == rb["cfg"]))
        tr = sim.run(Ps[s], C.GRID[ci], tp_r=4.0)["trades"]
        m = tr["entry_t"] < ts(end) + 86400
        ref[s] = dict(trailing_mean_r=rb["mean_r"], tp4r_mean_r=round(float(tr["r_net"][m].mean()), 4))
    crit = C.CRITERIA
    checks = {
        "1_trades>=300": M.get("n", 0) >= crit["min_trades"],
        "2_expectancy>=0.20R": M.get("expectancy_r", -9) >= crit["min_expectancy_r"],
        "3_PF>=1.3": (M.get("pf") or 0) >= crit["min_pf"],
        "4_maxDD<=6%": -M.get("max_dd_pct", -99) <= crit["max_dd"] * 100,
        "5_MC_pass>=50%": mc["p_pass"] >= crit["min_pass"],
        "6_stress_expectancy>0": M_st.get("expectancy_r", -9) > 0,
        "7_robustness": bool(rob) and all(v["ok"] for v in rob.values()),
    }
    res = dict(period=dict(start=start, end=end), metrics=M, per_symbol=per_sym, per_year=per_year.reset_index().to_dict("records"),
               stress=M_st, robustness=rob, monte_carlo=mc, tp4r_reference=ref, checks=checks, dev_pass=all(checks.values()),
               semesters_traded=int(sel["traded"].sum()), semesters_total=int(len(sel)))
    if a.holdout:
        if not res["dev_pass"]:
            raise SystemExit("Holdout refusé : critères de développement non remplis (protocole §2).")
        raise SystemExit("Holdout : à implémenter seulement si nécessaire.")
    (OUT / "jalon_a_dev.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
