"""Détecteur d'entrées « poolé » : un seul modèle appris sur 9 instruments (3 paires + 6 indices) pour multiplier les données.
Cible : résultat net d'une entrée (triple barrière bid/ask, k=2 vol, horizon 16 barres M15) en multiples de R.
Walk-forward trimestriel (entraînement sur tous les instruments avant le trimestre, embargo 1 jour), évalué sur les paires USD et sur les indices.
Développement uniquement (avant le 2026-04-29). Mesure principale : R brut par décile de score (le signal existe-t-il avant coûts ?)."""
import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

import common
from data import load_mt5_csv
from ml_fx import build_features, label_events

HOLDOUT = pd.Timestamp("2026-04-29")
K, H = 2.0, 16
INSTR = {"EURUSD": (0.45, 0.2, 0), "GBPUSD": (0.38, 0.2, 0), "US100.cash": (0.0, 0.2, 1), "US500.cash": (0.0, 0.2, 1),
         "US30.cash": (0.0, 0.2, 1), "GER40.cash": (0.0, 0.2, 1), "UK100.cash": (0.0, 0.2, 1), "JP225.cash": (0.0, 0.2, 1)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="mt5_export")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    parts = []
    for name, (comm, slip, cls) in INSTR.items():
        m1 = load_mt5_csv(f"{a.data_dir}/{name}_M1.csv")
        m1 = m1[m1.index < HOLDOUT]
        X, base, vol = build_features(m1, None)
        rl, rs, xl, xs = label_events(base, vol, K, H)
        cost = 2 * (comm + slip) / 1e4
        D = X.copy()
        D["cls"] = cls
        D["Rl"], D["Rs"] = (rl - cost) / (K * vol), (rs - cost) / (K * vol)
        D["Gl"], D["Gs"] = rl / (K * vol), rs / (K * vol)
        D["xl"], D["xs"] = xl - np.arange(len(D)), xs - np.arange(len(D))      # durée en barres
        D["inst"], D["time"] = name, base.index
        parts.append(D.dropna(subset=["Rl", "Rs"]))
        print(f"{name}: {len(parts[-1]):,} événements", flush=True)
    A = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    feats = [c for c in A.columns if c not in ("Rl", "Rs", "Gl", "Gs", "xl", "xs", "inst", "time")]
    feats = [c for c in feats if A[c].notna().mean() > 0.9]
    for c in ("Rl", "Rs", "Gl", "Gs"):
        A[c] = A[c].clip(-3, 3)
    q = A["time"].dt.to_period("Q")
    qs = sorted(q.unique())
    prm = dict(objective="l2", n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=800, subsample=0.7, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=30, random_state=a.seed, verbose=-1, n_jobs=-1)
    out = []
    for tq in qs[6:]:
        t0 = tq.start_time
        tr = A[(A["time"] < t0 - pd.Timedelta(days=1))].iloc[::2]
        te = A[q == tq]
        if len(te) == 0: continue
        pl = lgb.LGBMRegressor(**prm).fit(tr[feats], tr["Rl"]).predict(te[feats])
        ps = lgb.LGBMRegressor(**prm).fit(tr[feats], tr["Rs"]).predict(te[feats])
        out.append(te.assign(pl=pl, ps=ps))
        print(f"{tq}: entraînement {len(tr):,} / test {len(te):,}", flush=True)
    R = pd.concat(out, ignore_index=True)
    R["grp"] = np.where(R["cls"] == 1, "indices", "paires USD")
    R["side"] = np.where(R["pl"] >= R["ps"], 1, -1)
    R["score"] = np.maximum(R["pl"], R["ps"])
    R["Rnet"] = np.where(R["side"] == 1, R["Rl"], R["Rs"])
    R["Rbrut"] = np.where(R["side"] == 1, R["Gl"], R["Gs"])
    R["dur"] = np.where(R["side"] == 1, R["xl"], R["xs"])
    for g, G in R.groupby("grp"):
        print(f"\n=== {g} : {len(G):,} barres hors échantillon ===")
        ic = G[["pl", "Rl"]].corr("spearman").iloc[0, 1], G[["ps", "Rs"]].corr("spearman").iloc[0, 1]
        print(f"IC Spearman long/short : {ic[0]:.3f} / {ic[1]:.3f}")
        dec = pd.qcut(G["score"], 10, labels=False, duplicates="drop")
        T = G.groupby(dec).agg(n=("Rnet", "size"), R_brut=("Rbrut", "mean"), R_net=("Rnet", "mean"))
        print(T.round(3).T.to_string())
        top = G[G["score"] >= G["score"].quantile(0.9)]
        print(f"Décile supérieur : R brut {top.Rbrut.mean():.3f} (±{top.Rbrut.std() / np.sqrt(len(top)):.3f}) | R net {top.Rnet.mean():.3f}")
    R.to_pickle("results_pooled_oos.pkl")
