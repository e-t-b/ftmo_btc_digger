"""Portefeuille swing : rejoue les trades par symbole jour par jour, en R, avec plafonds de risque.

Chaque trade est engagé à `risk` (fraction du compte) au jour d'entrée ; son résultat quotidien est marqué à la
clôture, et le pire de la journée est estimé avec le plus bas (long) / plus haut + spread (short), tous les
instruments étant supposés atteindre leur pire au même moment (prudent). Swaps inclus nuit par nuit.
Sortie : journal des trades acceptés et série quotidienne compatible avec v2.ftmo.monte_carlo
(R et worst_R en unités du risque de base, relatifs à l'equity d'ouverture du jour)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import swing as S


def trade_path(D: dict, t) -> pd.DataFrame:
    """Valeur en R d'un trade à chaque clôture de sa vie (dernier jour = résultat réalisé net) et pire du jour."""
    ks = np.arange(t.i_entry, t.i_exit + 1)
    d, e, R = t.dsign, t.entry, t.R
    sp = D["spread"][ks]
    close = d * ((D["c"][ks] + (sp if d == -1 else 0)) - e) / R
    worst = d * (((D["l"][ks]) if d == 1 else (D["h"][ks] + sp)) - e) / R
    worst = np.maximum(worst, d * (t.stop - e) / R - 0.25)            # le stop borne la perte (marge glissement)
    sw = np.array([S.swap_per_unit_day(D, k, d) * S.triple_day(D, k) for k in ks]) / R
    cum_sw = np.r_[0.0, np.cumsum(sw)[:-1]]                            # swap payé les nuits précédentes
    close = close + cum_sw
    worst = np.minimum(worst + cum_sw, close)
    close[-1] = t.r_net
    worst[-1] = min(worst[-1], t.r_net)
    if t.i_exit == t.i_entry:
        worst[-1] = min(worst[-1], 0.0)
    return pd.DataFrame(dict(date=D["date"][ks], close=close, worst=worst))


def run(trades: pd.DataFrame, Ds: dict, risk=0.005, base=0.005, max_open=0.03, max_per_sym=1,
        max_same_side_fx=None, start=None, end=None, account=10_000.0):
    """trades : concat des S.to_frame (avec colonne sym) ; Ds : {sym: D}. Renvoie (acc, days)."""
    tr = trades.sort_values(["entry_date", "sym"]).reset_index(drop=True)
    if start is not None:
        tr = tr[tr["entry_date"] >= np.datetime64(start)]
    if end is not None:
        tr = tr[tr["entry_date"] < np.datetime64(end)]
    paths = {}
    by_day = {}
    open_ = []          # (idx, exit_date)
    accepted = []
    cal = np.unique(np.concatenate([Ds[s]["date"] for s in tr["sym"].unique()])) if len(tr) else np.array([], "datetime64[ns]")
    if start is not None:
        cal = cal[cal >= np.datetime64(start)]
    if end is not None:
        cal = cal[cal < np.datetime64(end)]
    entries = {k: g for k, g in tr.groupby("entry_date")}
    for day in cal:
        open_ = [(i, x) for i, x in open_ if x >= day]
        g = entries.get(day)
        if g is not None:
            for i, r in g.iterrows():
                key = (r["sym"], r.get("sid", ""))
                if sum(1 for j, _ in open_ if (tr.at[j, "sym"], tr.at[j, "sid"] if "sid" in tr else "") == key) >= max_per_sym:
                    continue
                if (len(open_) + 1) * risk > max_open + 1e-12:
                    continue
                open_.append((i, r["exit_date"]))
                accepted.append(i)
                p = trade_path(Ds[r["sym"]], r)
                for dt, cl, wo in zip(p["date"].to_numpy(), p["close"].to_numpy(), p["worst"].to_numpy()):
                    by_day.setdefault(dt, []).append((i, cl, wo))
    acc = tr.loc[accepted].copy()
    acc["r_eff"] = acc["r_net"] * risk / base
    acc["pnl"] = acc["r_net"] * risk * account
    # série quotidienne : variation de l'equity marquée à la clôture, pire relatif à l'ouverture du jour
    last = {}
    rows = []
    for day in cal:
        items = by_day.get(day, [])
        dR, wR = 0.0, 0.0
        for i, cl, wo in items:
            prev = last.get(i, 0.0)
            dR += cl - prev
            wR += wo - prev
            last[i] = cl
        f = risk / base
        rows.append(dict(fday=day, R=dR * f, worst_R=min(wR, dR, 0.0) * f, n=len(items),
                         pnl=dR * risk * account, worst=min(wR, dR, 0.0) * risk * account, halted=False))
    days = pd.DataFrame(rows)
    return acc, days


def metrics(acc: pd.DataFrame, days: pd.DataFrame, account=10_000.0) -> dict:
    if acc.empty:
        return dict(n=0)
    r = acc["r_net"].to_numpy()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(days["pnl"].to_numpy())
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    intr = np.r_[0.0, eq[:-1]] + days["worst"].to_numpy()
    yrs = max((pd.Timestamp(days["fday"].iloc[-1]) - pd.Timestamp(days["fday"].iloc[0])).days / 365.25, 1e-9)
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
    return dict(n=int(len(r)), trades_an=round(len(r) / yrs, 1), expectancy_r=round(float(r.mean()), 3),
                t_stat=round(float(r.mean() / se), 2) if se > 0 else None,
                pf=round(float(gp / gl), 2) if gl > 0 else None, hit=round(float((r > 0).mean()), 3),
                pct_an=round(float(eq[-1] / account * 100 / yrs), 2), max_dd_pct=round(float((eq - peak).min() / account * 100), 2),
                max_dd_intra_pct=round(float((intr - peak).min() / account * 100), 2),
                worst_day_pct=round(float(days["worst"].min() / account * 100), 2),
                swap_r=round(float(acc["swap_r"].mean()), 3), days_held=round(float(acc["days"].mean()), 1))
