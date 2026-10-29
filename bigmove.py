"""Détection des grands mouvements : cassures de la plage des 8 dernières heures, cible 4R (stop = 1 ATR H1), détention <= 8 h.
Un modèle poolé (8 instruments) estime le résultat net de chaque cassure ; walk-forward trimestriel, développement avant 2026-04-29.
Mesures : taux de réussite à 4R (seuil d'équilibre 20 % + coûts), R net par tranche de score, nombre de trades par jour."""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common
from data import load_mt5_csv
from ml_fx import build_features
from strategies import complete_bars

HOLDOUT = pd.Timestamp("2026-04-29")
TP_R, N_RANGE, HZ = 4.0, 32, 32
INSTR = {"EURUSD": (0.45, 0.2, 0), "GBPUSD": (0.38, 0.2, 0), "US100.cash": (0.0, 0.2, 1), "US500.cash": (0.0, 0.2, 1),
         "US30.cash": (0.0, 0.2, 1), "GER40.cash": (0.0, 0.2, 1), "UK100.cash": (0.0, 0.2, 1), "JP225.cash": (0.0, 0.2, 1)}
BAR = pd.Timedelta("15min")


def candidates(name, m1, comm, slip, cls):
    X, base, vol = build_features(m1, None)
    idx = base.index
    dec = idx + BAR
    o, h, l, c = (base[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sp = base["spread_px"].to_numpy(float)
    # ATR H1 et D1 connus à la décision
    def atr(rule, n):
        b = complete_bars(m1, rule)
        tr = pd.concat([b["high"] - b["low"], (b["high"] - b["close"].shift()).abs(), (b["low"] - b["close"].shift()).abs()], axis=1).max(axis=1)
        s = tr.rolling(n).mean()
        s.index = s.index + pd.Timedelta(rule)
        return s.reindex(dec, method="ffill").set_axis(idx)
    atr_h1, atr_d = atr("1h", 14), atr("1D", 14)
    a15 = pd.Series(h - l, index=idx).rolling(14).mean()
    hh = pd.Series(h, index=idx).rolling(N_RANGE).max().shift(1)
    ll = pd.Series(l, index=idx).rolling(N_RANGE).min().shift(1)
    rng8 = (hh - ll)
    # contexte de compression / consommation de la plage
    bw = (pd.Series(c, index=idx).rolling(20).std() * 4) / pd.Series(c, index=idx).rolling(20).mean()
    ctx = pd.DataFrame({
        "rng8_atrd": rng8 / atr_d, "a15_ah1": a15 / atr_h1, "bw_pct": bw.rolling(480).rank(pct=True), "ah1_ad": atr_h1 / atr_d,
        "adr_used": (pd.Series(h, index=idx).groupby(idx.normalize()).cummax() - pd.Series(l, index=idx).groupby(idx.normalize()).cummin()) / atr_d,
        "hod": dec.hour + dec.minute / 60, "ret_h1_atr": (pd.Series(c, index=idx) - pd.Series(c, index=idx).shift(4)) / atr_h1,
        "ret_h4_atr": (pd.Series(c, index=idx) - pd.Series(c, index=idx).shift(16)) / atr_h1,
    })
    up = (c > hh.to_numpy()) & (h > 0)
    dn = (c < ll.to_numpy())
    hour = dec.hour.to_numpy()
    ok = (hour >= 7) & (hour <= 15) & atr_h1.notna().to_numpy() & atr_d.notna().to_numpy() & (rng8.to_numpy() > 0)
    ts = idx.to_numpy("datetime64[ns]").astype("int64")
    step = 15 * 60 * 10 ** 9
    cost = 2 * (comm + slip) / 1e4
    rows = []
    n = len(idx)
    for i in np.flatnonzero((up | dn) & ok):
        e = i + 1
        if e + HZ >= n or ts[e + HZ - 1] - ts[e] != (HZ - 1) * step:
            continue
        d = 1 if up[i] else -1
        R = atr_h1.iat[i]
        if d == 1:
            en = o[e] + sp[e]; stop, tp = en - R, en + TP_R * R
            hs, ht = l[e:e + HZ] <= stop, h[e:e + HZ] >= tp
        else:
            en = o[e]; stop, tp = en + R, en - TP_R * R
            hs, ht = h[e:e + HZ] + sp[e:e + HZ] >= stop, l[e:e + HZ] + sp[e:e + HZ] <= tp
        hit = hs | ht
        if hit.any():
            j = int(hit.argmax()); ex = stop if hs[j] else tp; win = int((not hs[j]) and ht[j])
        else:
            j = HZ - 1; ex = c[e + j] if d == 1 else c[e + j] + sp[e + j]; win = 0
        r_g = d * (ex - en) / R
        r_n = r_g - cost * en / R
        ext = (h[e:e + HZ].max() - en) / R if d == 1 else (en - (l[e:e + HZ].min() + sp[e])) / R
        rows.append((idx[i], i, d, r_g, r_n, win, ext, (R / en) * 1e4, cost * en / R))
    E = pd.DataFrame(rows, columns=["time", "i", "d", "R_g", "R_n", "win", "mfe_R", "R_bps", "cost_R"])
    F = pd.concat([X.iloc[E["i"]].reset_index(drop=True), ctx.iloc[E["i"]].reset_index(drop=True)], axis=1)
    for col in [x for x in F.columns if any(s in x for s in ("ret", "sess", "dist", "macd", "slope"))]:
        F[col + "_d"] = F[col] * E["d"].to_numpy()
    F["d"], F["cls"] = E["d"].to_numpy(), cls
    F["brk_str"] = (np.where(E["d"] == 1, c[E["i"]] - hh.to_numpy()[E["i"]], ll.to_numpy()[E["i"]] - c[E["i"]])) / atr_h1.to_numpy()[E["i"]]
    out = pd.concat([E.drop(columns="i"), F], axis=1)
    out["inst"] = name
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="mt5_export")
    a = ap.parse_args()
    parts = []
    for name, (comm, slip, cls) in INSTR.items():
        m1 = load_mt5_csv(f"{a.data_dir}/{name}_M1.csv"); m1 = m1[m1.index < HOLDOUT]
        e = candidates(name, m1, comm, slip, cls)
        parts.append(e)
        print(f"{name}: {len(e):,} cassures | réussite 4R {e.win.mean()*100:.1f}% | R net moyen {e.R_n.mean():+.3f} | coût moyen {e.cost_R.mean():.3f}R | MFE>=4R : {(e.mfe_R>=4).mean()*100:.1f}%", flush=True)
    A = pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
    meta = ["time", "d", "R_g", "R_n", "win", "mfe_R", "R_bps", "cost_R", "inst"]
    feats = [c for c in A.columns if c not in meta]
    feats = [c for c in feats if A[c].notna().mean() > 0.9]
    A["y"] = A["R_n"].clip(-1.5, 4.5)
    q = A["time"].dt.to_period("Q"); qs = sorted(q.unique())
    prm = dict(objective="l2", n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.7, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=30, random_state=3, verbose=-1, n_jobs=-1)
    out = []
    for tq in qs[6:]:
        tr = A[A["time"] < tq.start_time - pd.Timedelta(days=1)]; te = A[q == tq]
        if len(te) == 0 or len(tr) < 3000: continue
        m = lgb.LGBMRegressor(**prm).fit(tr[feats], tr["y"])
        out.append(te.assign(pred=m.predict(te[feats])))
    R = pd.concat(out, ignore_index=True)
    days = (R["time"].max() - R["time"].min()).days
    print(f"\nHors échantillon : {len(R):,} cassures sur {days} jours | réussite 4R de base {R.win.mean()*100:.1f}% | R net de base {R.R_n.mean():+.3f}")
    print(f"AUC (score -> réussite 4R) : {roc_auc_score(R.win, R.pred):.3f}")
    dec = pd.qcut(R["pred"], 10, labels=False, duplicates="drop")
    T = R.groupby(dec).agg(n=("win", "size"), reussite_pct=("win", lambda s: 100 * s.mean()), R_brut=("R_g", "mean"), R_net=("R_n", "mean"), mfe4_pct=("mfe_R", lambda s: 100 * (s >= 4).mean()))
    print(T.round(2).T.to_string())
    for qt in (0.9, 0.97, 0.99):
        S = R[R["pred"] >= R["pred"].quantile(qt)]
        t = S.R_n.mean() / S.R_n.std(ddof=1) * np.sqrt(len(S))
        print(f"top {100*(1-qt):.0f}% : n={len(S)} ({len(S)/days:.2f}/jour tous instruments) réussite {S.win.mean()*100:.1f}% | R brut {S.R_g.mean():+.3f} | R net {S.R_n.mean():+.3f} (t={t:.2f})")
    R.to_pickle("results_bigmove_oos.pkl")
