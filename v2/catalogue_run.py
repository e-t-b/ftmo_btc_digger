"""Évaluation du catalogue S1-S15 en développement (docs/protocole_catalogue.md §1-2).

Usage : python -m v2.catalogue_run"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import catalogue as K
from . import config as C
from . import ftmo, portfolio
from .jalon_a import day_series, metrics, semesters, ts, tstat

OUT = Path("reports_v2")
CACHE = Path("cache")
END = C.DEV_END
E = ts(END) + 86400
ALPHA = 0.05
FAMILY = 19


def calendars():
    return {k: set(pd.to_datetime(pd.read_csv(f"data/calendar/{k}.csv")["date"]).dt.date) for k in ("nfp", "cpi", "fomc")}


def prepared(sym):
    f = CACHE / f"cat_{sym}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    P = K.prepare(sym)
    f.write_bytes(pickle.dumps(P))
    return P


def _store(runs, sid, sym, ci, r):
    df = r["trades"]
    df = df[df["entry_t"] < E].copy()                     # période de développement uniquement
    df["sid"], df["ci"] = sid, ci
    df["pkey"] = [(sid, sym, ci)] * len(df)
    runs[(sid, sym, ci)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])


def grid_runs(Ps, cal, stress=False, only=None):
    runs = {}
    for sid, sp in K.STRATS.items():
        if sid in ("S9", "S12"):
            continue
        for sym in sp["symbols"]:
            P = K.stressed(Ps[sym]) if stress else Ps[sym]
            for ci, cfg in enumerate(K.grid(sid)):
                if only is not None and (sid, sym, ci) not in only:
                    continue
                _store(runs, sid, sym, ci, K.run(P, sid, cfg, cal))
    return runs


def s12_runs(Ps, stress=False):
    """S12 : heure la plus marquée de l'entraînement (|t| >= 2), tradée au semestre suivant."""
    runs, sel = {}, []
    for sym in K.STRATS["S12"]["symbols"]:
        P = K.stressed(Ps[sym]) if stress else Ps[sym]
        t, o, c = P["t"], P["o"], P["c"]
        D = P["days"]
        grp = K.SYM_GROUP[sym]
        # créneaux : heures pleines de la fenêtre (BTC : heures UTC 0-20, hors bascule du jour FTMO)
        slots = []
        for _, r in D.iterrows():
            if grp == "btc":
                base = int(pd.Timestamp(r["date"]).value // 10**9)
                cand = [base + hh * 3600 for hh in range(0, 21)]
            else:
                cand = list(range(int(r["ws"]), int(r["we"]) - 3600 + 1, 3600))
            for k, a in enumerate(cand):
                if a + 3600 <= int(r["fc"]):
                    slots.append((r.name, k, a))
        sl = pd.DataFrame(slots, columns=["day", "slot", "a"])
        ja = np.searchsorted(t, sl["a"].to_numpy(np.int64))
        jb = np.searchsorted(t, (sl["a"] + 3600).to_numpy(np.int64)) - 1
        okk = (ja < len(t)) & (jb >= 0) & (jb > ja)
        sl = sl[okk].copy()
        ja, jb = ja[okk], jb[okk]
        okt = (t[ja] < sl["a"].to_numpy() + 300) & (t[jb] >= sl["a"].to_numpy() + 3000)
        sl, ja, jb = sl[okt].copy(), ja[okt], jb[okt]
        sl["ret"] = np.log(c[jb] / o[ja])
        sl["j"] = ja
        rows = []
        for a, b, lab in semesters(K.FIRST_TEST[sym], END):
            tr = sl[sl["a"] + 3600 <= a]
            g = tr.groupby("slot")["ret"]
            st = (g.mean() / g.std(ddof=1) * np.sqrt(g.size())).dropna()
            if st.empty:
                continue
            best = st.abs().idxmax()
            tt = float(st[best])
            traded = abs(tt) >= 2.0
            sel.append(dict(sid="S12", symbol=sym, semester=lab, cfg=f"slot{best}", train_t=round(tt, 2), traded=traded))
            if not traded:
                continue
            d = 1 if tt > 0 else -1
            te = sl[(sl["slot"] == best) & (sl["a"] >= a) & (sl["a"] < b)]
            for _, x in te.iterrows():
                j = int(x["j"])
                atr = P["atrH1"][j - 1] if j > 0 else np.nan
                if not (atr > 0):
                    continue
                e = P["o"][j] + P["sc"][j] + P["slip"][j] if d == 1 else P["o"][j] - P["slip"][j]
                rows.append((int(x["day"]), j, d, e - d * atr, 0.0, int(x["a"]) + 3600, 0.0))
        if rows:
            cols = ("day", "j", "d", "stop", "tp", "exit_t", "trail")
            S = {k: np.asarray(v, dtype=np.int64 if k in ("day", "j", "d", "exit_t") else float)
                 for k, v in zip(cols, zip(*rows))}
            order = np.argsort(S["j"], kind="stable")
            S = {k: v[order] for k, v in S.items()}
            r = K.run(P, "S12", {}, setups=S)
            _store(runs, "S12", sym, 0, r)
    return runs, pd.DataFrame(sel)


def walk_forward(runs, sid):
    G = K.grid(sid)
    sel, oos = [], []
    for sym in K.STRATS[sid]["symbols"]:
        if sid in K.NO_SELECTION:
            tr = runs[(sid, sym, 0)]["trades"]
            sel.append(dict(sid=sid, symbol=sym, semester="all", ci=0, cfg="fixed", train_t=None, traded=True))
            oos.append(tr.assign(semester="all"))
            continue
        for a, b, lab in semesters(K.FIRST_TEST[sym], END):
            best, bt = None, -np.inf
            for ci in range(len(G)):
                x = runs[(sid, sym, ci)]["trades"]
                x = x.loc[x["exit_t"] < a, "r_net"].to_numpy()
                if len(x) < K.MIN_TRAIN.get(sid, C.WF_MIN_TRADES):
                    continue
                tt = tstat(x)
                if tt > bt:
                    best, bt = ci, tt
            traded = best is not None and bt >= C.WF_MIN_T
            sel.append(dict(sid=sid, symbol=sym, semester=lab, a=a, b=b, ci=best,
                            cfg=K.tag(G[best]) if best is not None else None,
                            train_t=round(bt, 2) if best is not None else None, traded=traded))
            if traded:
                tr = runs[(sid, sym, best)]["trades"]
                oos.append(tr[(tr["entry_t"] >= a) & (tr["entry_t"] < b)].assign(semester=lab))
    return pd.DataFrame(sel), (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())


def evaluate(cands, runs, Ps, start, n_paths=20_000):
    if cands is None or len(cands) == 0:
        return None, None, dict(n=0, t_stat=None, p=1.0), dict(p_pass=0.0)
    paths = {k: (v["path_t"], v["path_u"]) for k, v in runs.items()}
    specs = {s: P["spec"] for s, P in Ps.items()}
    acc, days = portfolio.run(cands, paths, specs)
    days = day_series(days, start, END)
    M = metrics(acc, days)
    r = acc["r_eff"].to_numpy()
    tt = tstat(r) if len(r) > 1 else float("nan")
    M["t_stat"] = round(tt, 3) if tt == tt else None
    M["p"] = float(1 - norm.cdf(tt)) if tt == tt else 1.0
    mc = ftmo.monte_carlo(days["R"].to_numpy(), days["worst_R"].to_numpy(), days["n"].to_numpy(), n_paths=n_paths)
    return acc, days, M, mc


def holm(pvals: dict, m: int = FAMILY, alpha: float = ALPHA) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, stop = {}, False
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        ok = (not stop) and p <= thr
        if not ok:
            stop = True
        out[k] = dict(p=p, threshold=thr, significant=ok)
    return out


def robustness(runs, sid, sel):
    if sid in K.NO_SELECTION or sid == "S12":
        return None
    G = K.grid(sid)
    res = {}
    for sym in K.STRATS[sid]["symbols"]:
        cs = sel[(sel["symbol"] == sym) & sel["traded"]]["ci"]
        if cs.empty:
            continue
        ci = Counter(cs.astype(int)).most_common(1)[0][0]
        mr = lambda c: float(runs[(sid, sym, G.index(c))]["trades"]["r_net"].mean())
        nb = {K.tag(n): round(mr(n), 4) for n in K.neighbours(sid, G[ci])}
        own = round(mr(G[ci]), 4)
        res[sym] = dict(cfg=K.tag(G[ci]), mean_r=own, neighbours=nb,
                        ok=bool(own > 0 and sum(v > 0 for v in nb.values()) > len(nb) / 2))
    return res


def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    cal = calendars()
    syms = sorted({s for sp in K.STRATS.values() for s in sp["symbols"]})
    Ps = {s: prepared(s) for s in syms}
    runs = grid_runs(Ps, cal)
    r12, sel12 = s12_runs(Ps)
    runs.update(r12)
    table, sels, oos_all = {}, [], {}
    for sid in K.STRATS:
        if sid == "S9":
            continue
        if sid == "S12":
            sel = sel12
            oos = pd.concat([v["trades"] for k, v in r12.items()], ignore_index=True) if r12 else pd.DataFrame()
        else:
            sel, oos = walk_forward(runs, sid)
        sels.append(sel)
        syms_s = K.STRATS[sid]["symbols"]
        start = min(K.FIRST_TEST[s] for s in syms_s) if sid not in K.NO_SELECTION else \
            str(min(pd.Timestamp(Ps[s]["days"]["date"].iloc[0]) for s in syms_s).date())
        acc, days, M, mc = evaluate(oos, runs, Ps, start)
        oos_all[sid] = (oos, start)
        per_sym = acc.groupby("symbol").agg(n=("r_eff", "size"), exp_r=("r_eff", "mean")).round(4).reset_index() \
            .to_dict("records") if acc is not None else []
        table[sid] = dict(metrics=M, monte_carlo=mc, per_symbol=per_sym, robustness=robustness(runs, sid, sel),
                          semesters_traded=int(sel["traded"].sum()) if len(sel) else 0, semesters_total=int(len(sel)),
                          start=start)
        print(sid, {k: M.get(k) for k in ("n", "expectancy_r", "pf", "t_stat", "max_dd_pct")}, "MC", round(mc.get("p_pass", 0), 4), flush=True)
    # stratégies déjà testées (comptées dans la famille)
    prev = {}
    f = OUT / "oos_trades_dev.csv"
    if f.exists():
        x = pd.read_csv(f)["r_eff"].to_numpy()
        tt = tstat(x)
        prev["B0_breakout"] = float(1 - norm.cdf(tt))
    else:
        prev["B0_breakout"] = 1.0
    prev.update({"H1": 1.0, "H2": 1.0, "H3": 1.0, "S9": 1.0})
    pv = {sid: table[sid]["metrics"]["p"] for sid in table}
    pv.update(prev)
    H = holm(pv)
    # stress + survivantes
    survivors = []
    for sid in table:
        if not H[sid]["significant"]:
            table[sid]["stress"] = None
            continue
        sel = [s for s in sels if len(s) and s["sid"].iloc[0] == sid][0]
        only = {(sid, r["symbol"], int(r["ci"])) for _, r in sel[sel["traded"]].iterrows()} if sid != "S12" else None
        if sid == "S12":
            rs, _ = s12_runs(Ps, stress=True)
            oos_st = pd.concat([v["trades"] for v in rs.values()], ignore_index=True)
        else:
            rs = grid_runs(Ps, cal, stress=True, only=only)
            _, oos_st = walk_forward_fixed(rs, sel, sid)
        _, _, Mst, _ = evaluate(oos_st, rs, Ps, oos_all[sid][1], n_paths=2000)
        table[sid]["stress"] = Mst
        rob = table[sid]["robustness"]
        ok = Mst.get("expectancy_r", -1) > 0 and (rob is None or all(v["ok"] for v in rob.values()))
        table[sid]["survivor"] = bool(ok)
        if ok:
            survivors.append(sid)
    combo = None
    if survivors:
        cands = pd.concat([oos_all[s][0] for s in survivors], ignore_index=True)
        start = min(oos_all[s][1] for s in survivors)
        _, _, Mc, mcc = evaluate(cands, runs, Ps, start)
        cr = C.CRITERIA
        checks = {"n>=300": Mc["n"] >= cr["min_trades"], "PF>=1.3": (Mc.get("pf") or 0) >= cr["min_pf"],
                  "maxDD<=6%": -Mc["max_dd_pct"] <= cr["max_dd"] * 100, "MC>=50%": mcc["p_pass"] >= cr["min_pass"]}
        combo = dict(survivors=survivors, metrics=Mc, monte_carlo=mcc, checks=checks, pass_=all(checks.values()))
    res = dict(family=FAMILY, alpha=ALPHA, holm=H, strategies=table, survivors=survivors, combination=combo,
               holdout_allowed=bool(combo and combo["pass_"]))
    pd.concat(sels, ignore_index=True).to_csv(OUT / "catalogue_wf_selection.csv", index=False)
    (OUT / "catalogue_dev.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(dict(holm=H, survivors=survivors, combination=combo), indent=1, default=str))


def walk_forward_fixed(runs_st, sel, sid):
    """Mêmes sélections que le run de base, trades recalculés avec les coûts stressés."""
    oos = []
    for _, r in sel[sel["traded"]].iterrows():
        key = (sid, r["symbol"], int(r["ci"]))
        tr = runs_st[key]["trades"]
        if sid in K.NO_SELECTION:
            oos.append(tr)
        else:
            oos.append(tr[(tr["entry_t"] >= r["a"]) & (tr["entry_t"] < r["b"])])
    return sel, (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame())


if __name__ == "__main__":
    main()
