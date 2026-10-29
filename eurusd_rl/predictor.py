"""Étage 1 — Prédicteur supervisé (LightGBM) : la combinaison de tous nos signaux prédit-elle le résultat d'un trade ?

Pré-enregistré le 2026-10-29 à 22:20 (Paris), avant toute mesure.
Lignes : décisions où une entrée est permise, aux quarts d'heure serveur (:00, :15, :30, :45), jusqu'au 2025-12-31.
Cibles : pour chaque ligne, le trade que l'environnement prendrait (labels.py), en ACHAT et en VENTE, stop = distance
de stop de la décision (1 ATR M15, minimum de coûts), cible 2R (principale) et 4R (secondaire). Classe = R net > 0.
Modèles : un LightGBM par (cible, sens) = 4 modèles, hyperparamètres FIXES (ci-dessous), aucune recherche.
Validation : walk-forward annuel purgé ; année test Y = 2016 … 2025 (2025 = janvier-juin) ; entraînement sur les
lignes antérieures à Y moins 2 jours ; arrêt précoce sur les 10 % les plus récents de la fenêtre d'entraînement.
Porte (par modèle, sur les prédictions hors échantillon 2016 -> 2025-06) :
  - corrélation de Spearman >= 0,8 entre le décile de score (rang dans l'année test) et le R moyen réalisé ;
  - décile supérieur : R net moyen >= +0,05, t >= 3,0 calculé sur les moyennes JOURNALIÈRES (dépendance intra-jour),
    >= 300 trades, >= 65 % des années positives.
Confirmation : modèle final entraîné sur tout l'entraînement ; seuil = 90e centile de ses scores sur 2024-07 -> 2025-06 ;
R net moyen des lignes de la calibration (2025-07 -> 2025-12) au-dessus du seuil.
La validation 2026 et le test protégé ne sont pas lus (données tronquées au 2025-12-31).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data import load_config, load_m1
from .features_pred import build
from .labels import make_labels
from .market import build_market

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=500, feature_fraction=0.5,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, n_estimators=2000, random_state=1, n_jobs=2,
              verbose=-1)
GATE = {"min_spearman": 0.8, "min_mean_R": 0.05, "min_t_daily": 3.0, "min_trades": 300, "min_year_share": 0.65}
OUT = Path(__file__).parent / "reports"
CACHE = Path(__file__).resolve().parent.parent / "runs" / "predictor_cache.parquet"


def dataset(cfg):
    m1 = load_m1(cfg)
    m1 = m1[pd.DatetimeIndex(m1["t_utc"]) < pd.Timestamp("2026-01-01")].reset_index(drop=True)
    mk = build_market(cfg, m1)
    T = pd.DatetimeIndex(mk.dec_t_server)
    idx = np.flatnonzero(mk.dec_can_enter & (T.minute % 15 == 0))
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        assert np.array_equal(df["dec"].to_numpy(), idx)
        return df
    X = build(mk, m1, idx)
    for tp in (2.0, 4.0):
        lab = make_labels(mk, cfg, idx, tp)
        X[f"y_R_long_tp{int(tp)}"] = lab["R_long"]
        X[f"y_R_short_tp{int(tp)}"] = lab["R_short"]
    X["dec"] = idx
    X["date"] = pd.DatetimeIndex(mk.dec_t_utc[idx]).normalize()
    CACHE.parent.mkdir(exist_ok=True)
    X.to_parquet(CACHE)
    return X


def fit(Xtr, ytr):
    n = len(Xtr)
    cut = int(n * 0.9)
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xtr.iloc[:cut], ytr[:cut], eval_set=[(Xtr.iloc[cut:], ytr[cut:])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def daily_t(R: np.ndarray, dates: np.ndarray) -> float:
    d = pd.Series(R).groupby(pd.DatetimeIndex(dates)).mean()
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 2 else 0.0


def evaluate_oos(score, R, dates, years):
    df = pd.DataFrame({"s": score, "R": R, "date": dates, "y": years})
    df["rank"] = df.groupby("y")["s"].rank(pct=True)
    df["dec"] = np.minimum((df["rank"] * 10).astype(int), 9)
    by_dec = df.groupby("dec")["R"].agg(["count", "mean"]).round(4)
    rho = float(spearmanr(by_dec.index, by_dec["mean"]).correlation)
    top = df[df["rank"] > 0.9]
    yy = top.groupby("y")["R"].mean()
    res = {"spearman_decile": round(rho, 3), "deciles": by_dec.reset_index().to_dict(orient="records"),
           "top": {"n": int(len(top)), "mean_R": round(float(top["R"].mean()), 4),
                   "t_daily": round(daily_t(top["R"].to_numpy(), top["date"].to_numpy()), 2),
                   "year_share_pos": round(float((yy > 0).mean()), 3), "by_year": yy.round(3).to_dict(),
                   "win_rate": round(float((top["R"] > 0).mean()), 4)},
           "base_mean_R": round(float(df["R"].mean()), 4), "base_win_rate": round(float((df["R"] > 0).mean()), 4),
           "auc": round(float(roc_auc_score(df["R"] > 0, df["s"])), 4)}
    g = res["top"]
    res["pass"] = bool(rho >= GATE["min_spearman"] and g["mean_R"] >= GATE["min_mean_R"] and g["t_daily"] >= GATE["min_t_daily"]
                       and g["n"] >= GATE["min_trades"] and g["year_share_pos"] >= GATE["min_year_share"])
    return res


def main():
    t0 = time.time()
    cfg = load_config()
    D = dataset(cfg)
    feats = [c for c in D.columns if not c.startswith("y_") and c not in ("dec", "date")]
    dates = pd.DatetimeIndex(D["date"])
    tr_end = pd.Timestamp(cfg["splits"]["train"][1]) + pd.Timedelta(days=1)
    cal = (dates >= pd.Timestamp(cfg["splits"]["calibration"][0])) & (dates < pd.Timestamp(cfg["splits"]["calibration"][1]) + pd.Timedelta(days=1))
    in_train = dates < tr_end
    print(f"{len(D)} lignes, {len(feats)} features, entraînement {in_train.sum()}, calibration {cal.sum()}", flush=True)
    out = {"params": PARAMS, "gate": GATE, "n_features": len(feats), "models": {}}
    for tp in (2, 4):
        for side in ("long", "short"):
            key = f"{side}_tp{tp}"
            R = D[f"y_R_{side}_tp{tp}"].to_numpy()
            y = (R > 0).astype(int)
            oos = np.full(len(D), np.nan)
            imp = pd.Series(0.0, index=feats)
            for Y in range(2016, 2026):
                a = pd.Timestamp(f"{Y}-01-01")
                b = min(pd.Timestamp(f"{Y + 1}-01-01"), tr_end)
                te = (dates >= a) & (dates < b)
                trn = dates < a - pd.Timedelta(days=2)
                m = fit(D.loc[trn, feats], y[trn])
                oos[te] = m.predict_proba(D.loc[te, feats])[:, 1]
                imp += pd.Series(m.booster_.feature_importance("gain"), index=feats)
            sel = np.isfinite(oos)
            res = evaluate_oos(oos[sel], R[sel], dates[sel].values, dates[sel].year.to_numpy())
            res["top_features"] = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()
            # confirmation sur la calibration
            mf = fit(D.loc[in_train, feats], y[in_train])
            ref = (dates >= pd.Timestamp("2024-07-01")) & in_train
            thr = float(np.quantile(mf.predict_proba(D.loc[ref, feats])[:, 1], 0.9))
            sc = mf.predict_proba(D.loc[cal, feats])[:, 1]
            Rc = R[cal]
            topc = sc >= thr
            res["calibration"] = {"threshold": round(thr, 4), "n_top": int(topc.sum()),
                                  "mean_R_top": round(float(Rc[topc].mean()), 4) if topc.any() else None,
                                  "mean_R_all": round(float(Rc.mean()), 4),
                                  "auc": round(float(roc_auc_score(Rc > 0, sc)), 4)}
            out["models"][key] = res
            t = res["top"]
            print(f"{key:12s} AUC={res['auc']} Spearman={res['spearman_decile']} | base R={res['base_mean_R']} | "
                  f"top10% n={t['n']} R={t['mean_R']} t_jour={t['t_daily']} années+={t['year_share_pos']:.0%} | "
                  f"porte={'OUI' if res['pass'] else 'non'} | calib top R={res['calibration']['mean_R_top']} "
                  f"(n={res['calibration']['n_top']})", flush=True)
            print("   déciles R:", [d["mean"] for d in res["deciles"]], flush=True)
    out["elapsed_s"] = round(time.time() - t0)
    (OUT / "predictor_stage1.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
