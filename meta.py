"""Méta-étiquetage : un GBM filtre les candidats d'ouverture de session (cassure/rebond/fausse cassure) selon le contexte.
Développement uniquement (avant le 2026-04-29). Walk-forward par trimestre, entraînement >= 12 mois, un seul trade par session
(le candidat de meilleur score), seuils fixés d'avance en R net prédit : 0,00 / 0,05 / 0,10."""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import common
from data import load_mt5_csv
from anomalies import CASH, HOLDOUT, Px, ts_utc
from session_bt import prep, session_opens, run_config, PAIR_SESSIONS

DATA = sys.argv[1]
NAMES = {"US100.cash": "ny", "US500.cash": "ny", "US30.cash": "ny", "GER40.cash": "frankfurt", "UK100.cash": "london", "JP225.cash": "tokyo"}
FEATS = ["kind_i", "N", "dir", "trend", "opr_bps", "R_bps", "opr_rel", "k_after", "spread_bps", "gap", "prev_ret", "ret5", "vol20", "dow", "hour", "dir_x_gap", "dir_x_prev", "dir_x_trend"]
parts = []
for name, (tz, op, cl) in CASH.items():
    m1 = load_mt5_csv(f"{DATA}/{name}_M1.csv"); m1 = m1[m1.index < HOLDOUT]
    P = prep(m1)
    opens = session_opens(m1.index[0] + pd.Timedelta(days=40), m1.index[-1] - pd.Timedelta(days=1), PAIR_SESSIONS[name])
    days = pd.bdate_range(m1.index[0].normalize() + pd.Timedelta(days=40), m1.index[-1].normalize())
    px = Px(m1)
    po, _ = px.at(ts_utc(days, tz, op)); pc, _ = px.at(ts_utc(days, tz, cl))
    D = pd.DataFrame({"po": po, "pc": pc}, index=days)
    D["gap"] = D.po / D.pc.shift(1) - 1; D["prev_ret"] = D.pc.shift(1) / D.pc.shift(2) - 1
    D["ret5"] = D.pc.shift(1) / D.pc.shift(6) - 1; D["vol20"] = D.pc.pct_change().rolling(20).std().shift(1)
    for N in (15, 30):
        tr = run_config(P, opens, name, N, {"brk", "ret", "rev"}, 0)
        if tr.empty: continue
        tr["N"], tr["idx"] = N, name
        tr["date"] = tr["entry_time"].dt.normalize()
        parts.append(tr)
X = pd.concat(parts, ignore_index=True)
Dall = {}
for name, (tz, op, cl) in CASH.items():
    pass
X["kind_i"] = X["kind"].map({"brk": 0, "ret": 1, "rev": 2})
# contexte journalier
ctx = []
for name, (tz, op, cl) in CASH.items():
    m1 = load_mt5_csv(f"{DATA}/{name}_M1.csv"); m1 = m1[m1.index < HOLDOUT]
    days = pd.bdate_range(m1.index[0].normalize() + pd.Timedelta(days=40), m1.index[-1].normalize()); px = Px(m1)
    po, _ = px.at(ts_utc(days, tz, op)); pc, _ = px.at(ts_utc(days, tz, cl))
    D = pd.DataFrame({"po": po, "pc": pc}, index=days)
    D["gap"] = D.po / D.pc.shift(1) - 1; D["prev_ret"] = D.pc.shift(1) / D.pc.shift(2) - 1
    D["ret5"] = D.pc.shift(1) / D.pc.shift(6) - 1; D["vol20"] = D.pc.pct_change().rolling(20).std().shift(1)
    D["idx"] = name; ctx.append(D[["gap", "prev_ret", "ret5", "vol20", "idx"]].reset_index().rename(columns={"index": "date"}))
C = pd.concat(ctx)
X = X.merge(C, on=["date", "idx"], how="left")
X["opr_rel"] = X["opr_bps"] / X.groupby("idx")["opr_bps"].transform(lambda s: s.rolling(60, min_periods=10).median().shift(1))
X["dow"], X["hour"] = X["entry_time"].dt.dayofweek, X["entry_time"].dt.hour + X["entry_time"].dt.minute / 60
X["dir_x_gap"], X["dir_x_prev"], X["dir_x_trend"] = X["dir"] * X["gap"], X["dir"] * X["prev_ret"], X["dir"] * X["trend"]
X = X.dropna(subset=FEATS + ["R_net"]).sort_values("entry_time").reset_index(drop=True)
X["Rc"] = X["R_net"].clip(-1.5, 2.5)
print(f"Candidats : {len(X)} | R net moyen (tous) = {X.R_net.mean():.3f}")
q = X["entry_time"].dt.to_period("Q")
qs = sorted(q.unique()); first_test = qs[4]
res = []
prm = dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=150, subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=30, verbose=-1, n_jobs=2)
for tq in [x for x in qs if x >= first_test]:
    tr, te = X[q < tq], X[q == tq]
    if len(te) == 0 or len(tr) < 800: continue
    m = lgb.LGBMRegressor(**prm).fit(tr[FEATS], tr["Rc"])
    te = te.assign(pred=m.predict(te[FEATS]))
    res.append(te)
R = pd.concat(res)
R["sid"] = R["idx"] + R["date"].astype(str)
best = R.sort_values("pred", ascending=False).drop_duplicates("sid")      # un trade par session : le meilleur score
print(f"Hors-échantillon walk-forward : {len(R)} candidats / {len(best)} sessions | IC (Spearman pred vs R_net) = {R[['pred','R_net']].corr('spearman').iloc[0,1]:.3f}")
r_all = R.groupby("sid").first()
print(f"Référence : tous candidats R net = {R.R_net.mean():.3f}")
dec = pd.qcut(R["pred"], 10, labels=False, duplicates="drop")
print("Déciles de score -> R net moyen :", [round(float(v), 3) for v in R.groupby(dec)["R_net"].mean()])
start, end = R["entry_time"].min().normalize(), R["entry_time"].max().normalize()
for thr in (0.0, 0.05, 0.10):
    sel = best[best["pred"] > thr]
    if len(sel) < 30: print(thr, "trop peu"); continue
    d = common.daily_returns(sel.assign(net=sel["net"]), start, end)
    t = sel["R_net"].mean() / sel["R_net"].std(ddof=1) * np.sqrt(len(sel))
    print(f"seuil {thr:.2f} : n={len(sel)} R net={sel.R_net.mean():.3f} (t={t:.2f}) hit={float((sel.net>0).mean()):.3f} Sharpe={common.sharpe(d):.2f} pire jour={d.min()*100:.2f}% total={sel.net.sum()*100:.1f}%")
R.to_csv("results_meta_oos.csv", index=False)
