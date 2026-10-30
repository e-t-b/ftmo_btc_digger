"""Confirmation sur l'historique Dukascopy des indices (docs/protocole_dukascopy.md).

Usage : python -m v2.dukascopy"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from . import config as C
from .catalogue_run import evaluate, holm
from .data import load as load_ftmo
from .data import server_offset_hours
from .jalon_a import tstat

OUT = Path("reports_v2")
DK_DIR = Path(__file__).resolve().parent.parent / "data" / "dukascopy"
MAP = {"DK_US500": "US500.cash", "DK_US100": "US100.cash", "DK_GER40": "GER40.cash"}
END = int(pd.Timestamp("2021-09-01").value // 10**9)
CONFIRM = {"S2": dict(k=0.3, s=0.5, filt=1), "S10": {}, "S11": {}, "S25": {}, "S26": {}}
CONFIRM_SYMS = {"S2": ("DK_US500", "DK_US100", "DK_GER40"), "S10": ("DK_US500", "DK_US100"),
                "S11": ("DK_US500", "DK_US100", "DK_GER40"), "S25": ("DK_US500", "DK_US100", "DK_GER40"),
                "S26": ("DK_US500", "DK_US100", "DK_GER40")}
EXPLORE = ("S3", "S4", "S5", "S6", "S7", "S8", "S9", "S16", "S17", "S21", "S22", "S23", "S24", "S27", "S28", "S15")

for dk, ft in MAP.items():
    C.INSTRUMENTS[dk] = dict(C.INSTRUMENTS[ft])
    K.SYM_GROUP[dk] = K.SYM_GROUP[ft]
C.INSTRUMENTS["PAIR_DK"] = dict(C.INSTRUMENTS["PAIR_US100_US500"])
K.SYM_GROUP["PAIR_DK"] = "us"
K.PAIRS["PAIR_DK"] = ("DK_US100", "DK_US500")


def _cost_model(ft: str):
    """Profil de spread FTMO (p75 par quart d'heure serveur) et coûts, en fraction du prix."""
    from .prep import spread_cost
    ic = C.INSTRUMENTS[ft]
    m = load_ftmo(ft)
    _, prof = spread_cost(m, ic)
    a = int(pd.Timestamp(C.PROFILE_FIT[0]).value // 10**9)
    b = int((pd.Timestamp(C.PROFILE_FIT[1]) + pd.Timedelta(days=1)).value // 10**9)
    med = float(np.median(m["c"][(m["t"] >= a) & (m["t"] < b)]))
    return dict(prof=prof / med, floor=ic["floor_px"] / med, slip=ic["slip_px"] / med, sslip=ic["stop_slip_px"] / med,
                spec=m["spec"])


_ORIG_BASE = K._base


def _base_dk(sym):
    if sym not in MAP:
        return _ORIG_BASE(sym)
    z = np.load(DK_DIR / f"{sym}.npz")
    t = z["t_utc"].astype(np.int64)
    o, h, l, c = (z[k].astype(float) for k in ("open", "high", "low", "close"))
    keep = (t < END) & np.r_[True, np.diff(t) > 0]
    t, o, h, l, c = t[keep], o[keep], h[keep], l[keep], c[keep]
    tv = z["vol"][keep].astype(float)
    cm = _cost_model(MAP[sym])
    # quart d'heure serveur : UTC + 2 h (+3 h en heure d'été US), comme FTMO
    t_srv = t + 2 * 3600
    t_srv = t + server_offset_hours(t_srv) * 3600
    qh = ((t_srv % 86400) // 900).astype(np.int64)
    sc = np.maximum(cm["prof"][qh], cm["floor"]) * c
    return dict(symbol=sym, t=t, o=o, h=h, l=l, c=c, tv=np.maximum(tv * 1000, 1.0), sc=sc,
                comm=np.zeros(len(t)), slip=cm["slip"] * c, sslip=cm["sslip"] * c, spec=cm["spec"])


K._base = _base_dk


def calendars_dk():
    from .catalogue_run import calendars
    cal = calendars()
    cal["fomc"] = cal["fomc"] | set(pd.to_datetime(pd.read_csv("data/calendar/fomc_2013_2020.csv")["date"]).dt.date)
    return cal


def runner(P, sid, cfg, cal, stress=False):
    Q = K.stressed(P) if stress else P
    if sid in ("S24", "S25", "S26"):
        from . import lot4
        lot4._CAL = cal
        return lot4.run(Q, sid)
    if sid in ("S16", "S17"):
        from . import lot2
        return lot2.run(Q, sid)
    if sid in ("S21", "S22", "S23"):
        from . import lot3
        return lot3.run(Q, sid)
    if sid in ("S27", "S28"):
        from . import lot5
        return lot5.run(Q, sid)
    if sid == "S9":
        return K.run(Q, "S9", {"fixed": 1}, cal)
    return K.run(Q, sid, cfg, cal)


def run_strategy(Ps, sid, cfg, syms, cal, stress=False):
    runs, parts = {}, []
    for s in syms:
        r = runner(Ps[s], sid, cfg, cal, stress)
        df = r["trades"]
        df = df[df["entry_t"] < END].copy()
        df["sid"], df["ci"] = sid, 0
        df["pkey"] = [(sid, s, 0)] * len(df)
        runs[(sid, s, 0)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])
        parts.append(df)
    return runs, pd.concat(parts, ignore_index=True)


def main():
    OUT.mkdir(exist_ok=True)
    cal = calendars_dk()
    Ps = {s: K.prepare(s) for s in MAP}
    for s, P in Ps.items():
        print(s, "jours", len(P["days"]), pd.Timestamp(P["days"]["date"].iloc[0]).date(), "->",
              pd.Timestamp(P["days"]["date"].iloc[-1]).date(), "coût médian bps", round(float(np.median(P["sc"] / P["c"]) * 1e4), 2), flush=True)
    start = str(min(pd.Timestamp(P["days"]["date"].iloc[0]) for P in Ps.values()).date())
    conf = {}
    for sid, cfg in CONFIRM.items():
        runs, tr = run_strategy(Ps, sid, cfg, CONFIRM_SYMS[sid], cal)
        acc, days, M, mc = evaluate(tr, runs, Ps, start)
        _, tr_st = run_strategy(Ps, sid, cfg, CONFIRM_SYMS[sid], cal, stress=True)
        xs = tr_st["r_net"].to_numpy()
        per = acc.groupby("symbol").agg(n=("r_eff", "size"), exp_r=("r_eff", "mean")).round(4).reset_index().to_dict("records") \
            if acc is not None else []
        acc["year"] = pd.to_datetime(acc["entry_t"], unit="s").dt.year
        yr = acc.groupby("year")["r_eff"].agg(["size", "mean"]).round(3).reset_index().to_dict("records")
        conf[sid] = dict(metrics=M, monte_carlo=mc, per_symbol=per, per_year=yr,
                         stress=dict(n=int(len(xs)), expectancy_r=round(float(xs.mean()), 4) if len(xs) else None))
        print(sid, {k: M.get(k) for k in ("n", "expectancy_r", "pf", "t_stat", "p", "max_dd_pct")}, per, flush=True)
    H = holm({k: v["metrics"]["p"] for k, v in conf.items()}, m=len(conf))
    for sid in conf:
        conf[sid]["holm"] = H[sid]
        conf[sid]["confirmed"] = bool(H[sid]["significant"] and (conf[sid]["stress"]["expectancy_r"] or -1) > 0)
    expl = {}
    for sid in EXPLORE:
        syms = ("PAIR_DK",) if sid == "S15" else (("DK_US500", "DK_US100") if sid == "S24" else tuple(MAP))
        if sid == "S15" and "PAIR_DK" not in Ps:
            Ps["PAIR_DK"] = K.prepare("PAIR_DK")
        cfg = K.grid(sid)[0] if (sid in K.STRATS and sid != "S9") else {}
        _, tr = run_strategy(Ps, sid, cfg, syms, cal)
        x = tr["r_net"].to_numpy()
        expl[sid] = dict(n=int(len(x)), expectancy_r=round(float(x.mean()), 4) if len(x) else None,
                         t=round(tstat(x), 2) if len(x) > 2 else None)
        print("explo", sid, expl[sid], flush=True)
    res = dict(window=dict(start=start, end="2021-08-31"), confirmatory=conf, exploratory=expl,
               confirmed=[k for k, v in conf.items() if v["confirmed"]])
    (OUT / "dukascopy.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(dict(holm=H, confirmed=res["confirmed"]), indent=1, default=str))


if __name__ == "__main__":
    main()
