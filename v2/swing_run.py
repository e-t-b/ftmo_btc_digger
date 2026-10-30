"""Étude swing (docs/protocole_swing.md). Usage :
  python -m v2.swing_run dev       # développement : stratégies, Holm, choix du risque
  python -m v2.swing_run holdout   # test unique sur la période finale (après commit du choix dev)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import ftmo
from . import swing as S
from . import swing_pf as PF
from . import swing_strats as ST

PROTO = json.loads((Path(__file__).resolve().parent.parent / "docs" / "protocole_swing.json").read_text())
DEV_END = PROTO["dev_end"]
HO_START, HO_END = PROTO["holdout"]


def universe():
    out = {}
    for f in sorted(S.D1_DIR.glob("*_D1.csv")):
        s = f.name[:-7]
        if s in PROTO["exclude"]:
            continue
        out[s] = S.load(s)
    return out


def symbols_for(sid, Ds):
    if sid == "M1":
        return [s for s in Ds if s in PROTO["indices"]]
    return list(Ds)


def apply_swap_model(D, model):
    """base : coût symétrique = moyenne (long, short) si négative, appliquée aux deux sens (le portage historique
    est inconnu) ; current : swaps actuels tels quels (stress)."""
    s = dict(D["spec"])
    if model == "base":
        m = min(0.0, (s["swap_long"] + s["swap_short"]) / 2)
        s["swap_long"] = s["swap_short"] = m
    return dict(D, spec=s, _ind=D.get("_ind")) if "_ind" in D else dict(D, spec=s)


def trades_for(sid, Ds, swap_model="base", cost_mult=1.0):
    parts = []
    for s in symbols_for(sid, Ds):
        D = apply_swap_model(Ds[s], swap_model)
        df = ST.run_symbol(D, sid, cost_mult=cost_mult)
        if len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def window(df, a, b):
    w = df[(df["entry_date"] >= np.datetime64(a)) & (df["entry_date"] < np.datetime64(b))]
    if np.datetime64(b) > np.datetime64(DEV_END):          # Dukascopy : développement seulement
        w = w[~w["sym"].str.startswith("DK_") | (w["entry_date"] < np.datetime64(DEV_END))]
    return w


def evaluate(tr, Ds, a, b, risk, n_paths=10_000):
    acc, days = PF.run(tr, Ds, risk=risk, base=C.RISK["base"], max_open=PROTO["max_open"], start=a, end=b)
    M = PF.metrics(acc, days)
    old = dict(C.RISK)
    C.RISK.update(PROTO["risk_rules"])
    mc = ftmo.monte_carlo(days["R"].to_numpy(), days["worst_R"].to_numpy(), days["n"].to_numpy(),
                          n_paths=n_paths, horizon=PROTO["mc_horizon"])
    C.RISK.clear(); C.RISK.update(old)
    return acc, days, M, mc


def dev():
    Ds = universe()
    start = PROTO["dev_start"]
    out = dict(universe={s: str(pd.Timestamp(D["date"][0]).date()) for s, D in Ds.items()}, strategies={})
    pvals = {}
    for sid in ST.STRATS:
        tr = window(trades_for(sid, Ds), start, DEV_END)
        r = tr["r_net"].to_numpy()
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        p = float(stats.t.sf(t, len(r) - 1))
        pvals[sid] = p
        st = window(trades_for(sid, Ds, "current", 1.5), start, DEV_END)["r_net"].mean()
        sems = tr.assign(y=pd.to_datetime(tr["entry_date"]).dt.year // 2 * 2).groupby("y")["r_net"].mean()
        per = tr.groupby("sym")["r_net"].agg(["size", "mean"]).round(3)
        out["strategies"][sid] = dict(n=len(r), R=round(float(r.mean()), 3), t=round(float(t), 2), p=p,
                                      R_stress=round(float(st), 3), pos_2yr_blocks=f"{int((sems > 0).sum())}/{len(sems)}",
                                      by_2yr=sems.round(3).to_dict(),
                                      per_symbol=per.reset_index().to_dict("records"))
        print(sid, out["strategies"][sid]["n"], out["strategies"][sid]["R"], out["strategies"][sid]["t"],
              "stress", out["strategies"][sid]["R_stress"], out["strategies"][sid]["pos_2yr_blocks"], flush=True)
    # Holm sur la famille cumulée du projet
    fam = json.loads(Path("reports_v2/famille.json").read_text())
    allp = {k: v["p"] for k, v in fam.items() if v.get("p") is not None}
    allp.update({f"SW_{k}": v for k, v in pvals.items()})
    m = len(allp)
    order_all = sorted(allp.items(), key=lambda kv: kv[1])
    keep = []
    for k, (sid, p) in enumerate(order_all):
        if p * (m - k) <= PROTO["alpha"]:
            if sid.startswith("SW_"):
                keep.append(sid[3:])
        else:
            break
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    for sid, p in pvals.items():
        fam[f"SW_{sid}"] = dict(p=p, lot="swing")
    Path("reports_v2/famille.json").write_text(json.dumps(fam, indent=1))
    # Holm sur la seule famille swing, publiée à titre d'information
    keep_local = []
    for k, (sid, p) in enumerate(order):
        if p * (len(order) - k) <= PROTO["alpha"]:
            keep_local.append(sid)
        else:
            break
    out["holm"] = dict(m_family=m, kept=keep, kept_swing_only=keep_local)
    rule = PROTO["selection"]
    sel = keep if rule == "holm_family" else keep_local
    out["selected"] = sel
    # choix du risque sur le développement
    risks = {}
    if sel:
        tr = pd.concat([window(trades_for(s, Ds), start, DEV_END) for s in sel], ignore_index=True)
        for rk in PROTO["risk_grid"]:
            acc, days, M, mc = evaluate(tr, Ds, start, DEV_END, rk)
            risks[str(rk)] = dict(metrics=M, mc=mc)
            print("risque", rk, M, round(mc["p_pass"], 3), mc["median_days_total"], flush=True)
        ok = {k: v for k, v in risks.items() if -v["metrics"]["max_dd_intra_pct"] <= PROTO["dev_max_dd"]}
        best = max(ok, key=lambda k: ok[k]["mc"]["p_pass"]) if ok else None
        out["risk"] = dict(grid=risks, chosen=best)
    Path("reports_v2/swing_dev.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def holdout():
    d = json.loads(Path("reports_v2/swing_dev.json").read_text())
    sel, rk = d["selected"], d.get("risk", {}).get("chosen")
    if not sel or rk is None:
        print("Aucune stratégie retenue en développement : pas de test final.")
        return
    Ds = universe()
    tr = pd.concat([window(trades_for(s, Ds), HO_START, HO_END) for s in sel], ignore_index=True)
    acc, days, M, mc = evaluate(tr, Ds, HO_START, HO_END, float(rk), n_paths=20_000)
    trs = pd.concat([window(trades_for(s, Ds, "current", 1.5), HO_START, HO_END) for s in sel], ignore_index=True)
    _, _, Ms, mcs = evaluate(trs, Ds, HO_START, HO_END, float(rk), n_paths=5_000)
    crit = {"1_expectancy>0": M["expectancy_r"] > 0, "2_PF>=1.2": (M["pf"] or 0) >= 1.2,
            "3_MC>=40%": mc["p_pass"] >= 0.40, "4_DD<=10%": -M["max_dd_intra_pct"] <= 10}
    per_sid = acc.groupby("sid")["r_net"].agg(["size", "mean"]).round(3).reset_index().to_dict("records")
    per_year = acc.assign(y=pd.to_datetime(acc["entry_date"]).dt.year).groupby("y")["r_net"].agg(["size", "mean"]).round(3)
    res = dict(selected=sel, risk=rk, metrics=M, mc=mc, criteria=crit, decision="GO" if all(crit.values()) else "NO-GO",
               stress=dict(metrics=Ms, mc=mcs), per_strategy=per_sid, per_year=per_year.reset_index().to_dict("records"))
    acc.to_csv("reports_v2/swing_holdout_trades.csv", index=False)
    Path("reports_v2/swing_holdout.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"dev": dev, "holdout": holdout}[sys.argv[1]]()
