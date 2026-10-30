"""M1 : méta-modèle de sélection de trades selon le contexte (docs/protocole_meta.md).

Usage : python -m v2.meta"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from . import catalogue as K
from .catalogue_run import calendars, evaluate, grid_runs, holm, prepared
from .jalon_a import semesters, ts, tstat
from .lot_run import collect, lot_modules

OUT = Path("reports_v2")
END = "2024-06-30"
CAT_SIDS = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S10", "S11", "S13", "S14", "S15")
LOTS = ("lot2", "lot3", "lot4", "lot5")
PARAMS = dict(objective="regression", n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=7, verbose=-1)
CAT_FEATS = ["sid", "symbol", "dir"]


def pool(stress=False):
    cal = calendars()
    only = {(sid, s, 0) for sid in CAT_SIDS for s in K.STRATS[sid]["symbols"]}
    syms = sorted({s for _, s, _ in only})
    Ps = {s: prepared(s) for s in syms}
    runs = grid_runs(Ps, cal, stress=stress, only=only)
    for lot in LOTS:
        strat_syms, runner = lot_modules(lot)
        for s in {x for v in strat_syms.values() for x in v}:
            if s not in Ps:
                Ps[s] = prepared(s)
        r, _ = collect(Ps, strat_syms, runner, stress=stress)
        runs.update(r)
    trades = pd.concat([v["trades"] for v in runs.values()], ignore_index=True)
    return Ps, runs, trades, cal


def _d1(P):
    if "_metaD" not in P:
        t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
        D = K._bars(t, h, l, c, o, 86400)
        atr = pd.Series(K._tr(D["h"], D["l"], D["c"])).rolling(14, min_periods=14).mean().to_numpy()
        ema = pd.Series(D["c"]).ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
        P["_metaD"] = (D["end"], atr, np.sign(D["c"] - ema), D["h"] - D["l"], D["c"])
    return P["_metaD"]


def features(trades, Ps, cal):
    rows = []
    for s, g in trades.groupby("symbol"):
        P = Ps[s]
        t, c = P["t"], P["c"]
        end, atr, trd, rng, cls = _d1(P)
        et = g["entry_t"].to_numpy(np.int64)
        j = np.searchsorted(t, et)
        k = np.maximum(j - 1, 0)                                   # dernière minute clôturée avant l'entrée
        b = np.searchsorted(end, t[k] + 60, side="right") - 1      # dernier jour UTC terminé
        ok = b >= 0
        bb = np.maximum(b, 0)
        A = np.where(ok, atr[bb], np.nan)
        px = c[k]
        d = g["dir"].to_numpy()

        def ret(m):
            kk = np.searchsorted(t, t[k] - m * 60)
            return d * (px - c[np.minimum(kk, len(c) - 1)]) / A
        f = pd.DataFrame(index=g.index)
        f["atr_rel"] = A / px
        f["trend_x_dir"] = np.where(ok, trd[bb], 0) * d
        f["prev_range_atr"] = np.where(ok, rng[bb], np.nan) / A
        f["gap_atr"] = d * (px - np.where(ok, cls[bb], np.nan)) / A
        f["atr15_rel"] = P["atr15"][k] / A
        f["atrH1_rel"] = P["atrH1"][k] / A
        f["ret15"], f["ret60"], f["ret240"] = ret(15), ret(60), ret(240)
        f["stop_atr"] = g["R"].to_numpy() / A
        dt = pd.to_datetime(et, unit="s")
        f["hour"], f["minute"], f["dow"], f["month"] = dt.hour, dt.hour * 60 + dt.minute, dt.dayofweek, dt.month
        dd = pd.Series(dt.date)
        for ev in ("nfp", "cpi", "fomc"):
            f[ev] = dd.isin(cal[ev]).to_numpy().astype(int)
        rows.append(f)
    F = pd.concat(rows).loc[trades.index]
    F["cost_r"] = trades["cost_r"]
    # historique propre : 50 derniers trades (même stratégie, même instrument) clôturés avant l'entrée
    F["hist_mean"], F["hist_n"] = np.nan, 0
    for (sid, s), g in trades.groupby(["sid", "symbol"]):
        ex = g.sort_values("exit_t")
        ext, exr = ex["exit_t"].to_numpy(), ex["r_net"].to_numpy()
        cs = np.r_[0, np.cumsum(exr)]
        n_before = np.searchsorted(ext, g["entry_t"].to_numpy(), side="left")
        lo = np.maximum(n_before - 50, 0)
        cnt = n_before - lo
        F.loc[g.index, "hist_n"] = cnt
        F.loc[g.index, "hist_mean"] = np.where(cnt > 0, (cs[n_before] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    F["sid"] = trades["sid"].astype("category")
    F["symbol"] = trades["symbol"].astype("category")
    F["dir"] = trades["dir"].astype("category")
    return F


def main():
    OUT.mkdir(exist_ok=True)
    Ps, runs, T, cal = pool()
    T = T.reset_index(drop=True)
    # objectif fixe connu à l'entrée : stratégies / configurations avec objectif
    tp_sids = {"S5", "S6", "S7", "S15", "S16", "S27", "S28"}   # grid[0] avec objectif fixe
    F = features(T, Ps, cal)
    F["has_tp"] = T["sid"].isin(tp_sids).astype(int)
    y = T["r_net"].clip(-3, 10).to_numpy()
    print("réservoir :", len(T), "trades,", T["sid"].nunique(), "stratégies", flush=True)
    sel_idx, base_idx, log = [], [], []
    for a, b, lab in semesters("2018-01-01", END):
        tr = T["exit_t"] < a
        te = (T["entry_t"] >= a) & (T["entry_t"] < b)
        base_idx += list(T.index[te])
        if tr.sum() < 2000 or te.sum() == 0:
            log.append(dict(semester=lab, n_train=int(tr.sum()), n_test=int(te.sum()), n_selected=0))
            continue
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(F[tr], y[tr], categorical_feature=CAT_FEATS)
        pred = m.predict(F[te])
        chosen = T.index[te][pred > 0]
        sel_idx += list(chosen)
        log.append(dict(semester=lab, n_train=int(tr.sum()), n_test=int(te.sum()), n_selected=int(len(chosen)),
                        mean_r_selected=round(float(T.loc[chosen, "r_net"].mean()), 4) if len(chosen) else None,
                        mean_r_all=round(float(T.loc[T.index[te], "r_net"].mean()), 4)))
        print(log[-1], flush=True)
    S = T.loc[sel_idx]
    B = T.loc[base_idx]
    S.drop(columns=["pkey"]).to_csv(OUT / "meta_selected_trades.csv", index=False)
    (OUT / "meta_semesters.json").write_text(json.dumps(log, indent=1))
    print("étape suivante : python -m v2.meta_eval base ; stress ; final")


if __name__ == "__main__":
    main()
