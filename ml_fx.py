"""Pipeline ML forex (FTMO) : features multi-échelles + sessions + cross-paire -> deux régresseurs LightGBM
(PnL net attendu d'un long / d'un short, en multiples de la barrière) -> walk-forward purgé -> trades nets.

Exécution modélisée comme MT5 : prix = BID, ask = bid + spread ; barrière-cible/stop testées sur bid (long) ou ask (short) ;
commission, slippage et swap par aller-retour. Décision à la clôture de la barre 15 min, entrée à l'open de la suivante.
Fenêtres contenant un trou de données (week-end) exclues. Les événements d'entraînement dont la sortie tombe après
(début du test - embargo) sont purgés.

Usage : python ml_fx.py --pair EURUSD --other GBPUSD --k 2 --horizon 16
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common
from data import load_mt5_csv
from strategies import complete_bars

BAR = pd.Timedelta("15min")
TFS = [("5min", "m5"), ("15min", "m15"), ("1h", "h1"), ("4h", "h4"), ("1D", "d1")]
DEFAULT_COSTS = {  # bps par côté (commission, slippage) ; swap en bps/jour (long, short, coût positif)
    "EURUSD": dict(comm=0.45, slip=0.2, swap_long=0.9, swap_short=0.0),
    "GBPUSD": dict(comm=0.38, slip=0.2, swap_long=0.4, swap_short=0.45),
}
DAY_NS, ROLL_NS = 86_400 * 10 ** 9, 21 * 3600 * 10 ** 9


# ----------------------------------------------------------------------------- features
def tf_features(bars, rule, p, with_vol=True):
    c = bars["close"]
    lc = np.log(c)
    f = pd.DataFrame(index=bars.index)
    rv = lc.diff().rolling(20).std()
    for n in (1, 4, 12):
        f[f"{p}_ret{n}"] = lc.diff(n)
    f[f"{p}_ret1_z"] = lc.diff(1) / rv
    f[f"{p}_ret4_z"] = lc.diff(4) / (rv * 2)
    f[f"{p}_rv20"] = rv
    rng = (bars["high"] - bars["low"]) / c
    f[f"{p}_range_z"] = rng / rng.rolling(50).mean()
    if with_vol:
        f[f"{p}_volz"] = bars["tick_volume"] / bars["tick_volume"].rolling(50).mean()
    lo, hi = c.rolling(20).min(), c.rolling(20).max()
    f[f"{p}_pos"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f.index = f.index + pd.Timedelta(rule)           # index = heure de COMPLÉTION de la bougie
    return f


def build_features(m1, m1_other=None):
    base = complete_bars(m1, "15min")
    idx = base.index
    dec = idx + BAR
    parts = []
    for rule, pre in TFS:
        bars = base if rule == "15min" else complete_bars(m1, rule)
        parts.append(tf_features(bars, rule, pre).reindex(dec, method="ffill").set_axis(idx))
    if m1_other is not None:
        for rule, pre in (("15min", "x15"), ("1h", "xh1"), ("4h", "xh4")):
            f = tf_features(complete_bars(m1_other, rule), rule, pre, with_vol=False)
            f = f[[c for c in f.columns if "ret" in c]]
            parts.append(f.reindex(dec, method="ffill").set_axis(idx))
    X = pd.concat(parts, axis=1)
    c = base["close"]
    vol = np.log(c).diff().ewm(span=96, min_periods=96).std()
    # temps et sessions
    hr = (dec.hour + dec.minute / 60).to_numpy()
    X["hod_sin"], X["hod_cos"], X["dow"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), dec.dayofweek.to_numpy()
    for h in (0, 7, 13):
        anchor = base["open"].where((idx.hour == h) & (idx.minute == 0)).ffill()
        X[f"sess{h}_ret"] = np.log(c / anchor) / vol
    day = idx.normalize()
    d = complete_bars(m1, "1D")
    for name, col in (("pdh", "high"), ("pdl", "low"), ("pdc", "close")):
        prev = d[col].shift(1).reindex(day).to_numpy()
        X[f"{name}_dist"] = (c.to_numpy() - prev) / c.to_numpy() / vol.to_numpy()
    X["day_hi_dist"] = (c - base["high"].groupby(day).cummax()) / c / vol
    X["day_lo_dist"] = (c - base["low"].groupby(day).cummin()) / c / vol
    X["spread_z"] = base["spread_px"] / base["spread_px"].rolling(200).median()
    X["vol_ratio"] = X["m15_rv20"] / X["d1_rv20"]
    return X, base, vol.to_numpy()


# ----------------------------------------------------------------------------- étiquettes (triple barrière bid/ask)
def label_events(base, vol, k, H):
    o, h, l, c = (base[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    sp = base["spread_px"].to_numpy(float)
    ts = base.index.to_numpy("datetime64[ns]").astype("int64")
    n, step = len(base), 15 * 60 * 10 ** 9
    rl, rs = np.full(n, np.nan), np.full(n, np.nan)
    xl, xs = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    for i in range(n - H - 1):
        b = k * vol[i]
        e = i + 1
        if not np.isfinite(b) or ts[e] - ts[i] != step or ts[e + H - 1] - ts[e] != (H - 1) * step:
            continue
        hw, lw, spw = h[e:e + H], l[e:e + H], sp[e:e + H]
        # long : entrée ask, cibles sur bid
        en = o[e] + sp[e]
        tp, sl = en * (1 + b), en * (1 - b)
        hs, ht = lw <= sl, hw >= tp
        hit = hs | ht
        if hit.any():
            j = int(hit.argmax())
            ex = sl if hs[j] else tp
            xl[i] = e + j
        else:
            ex, xl[i] = c[e + H - 1], e + H - 1
        rl[i] = (ex - en) / en
        # short : entrée bid, cibles sur ask
        en = o[e]
        tp, sl = en * (1 - b), en * (1 + b)
        hs, ht = hw + spw >= sl, lw + spw <= tp
        hit = hs | ht
        if hit.any():
            j = int(hit.argmax())
            ex = sl if hs[j] else tp
            xs[i] = e + j
        else:
            ex, xs[i] = c[e + H - 1] + sp[e + H - 1], e + H - 1
        rs[i] = (en - ex) / en
    return rl, rs, xl, xs


def net_returns(r, entry_idx, exit_idx, ts, comm, slip, swap_bps):
    t0, t1 = ts[entry_idx], ts[exit_idx]
    n_roll = (t1 - ROLL_NS) // DAY_NS - (t0 - ROLL_NS) // DAY_NS
    return r - 2 * (comm + slip) / 1e4 - n_roll * swap_bps / 1e4


# ----------------------------------------------------------------------------- walk-forward
def fit_predict(Xtr, ytr, Xte, a):
    out = []
    for y in ytr:
        m = lgb.LGBMRegressor(objective="l2", n_estimators=a.n_estimators, learning_rate=0.03, num_leaves=15,
                              min_child_samples=500, subsample=0.7, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=20, random_state=a.seed, verbose=-1, n_jobs=-1)
        m.fit(Xtr, y)
        out.append((m.predict(Xte), m.booster_.feature_importance("gain")))
    return out


def run_simulation(D, pred_l, pred_s, thr):
    score = np.maximum(pred_l, pred_s)
    side = np.where(pred_l >= pred_s, 1, -1)
    take = np.flatnonzero(score > thr)
    rows, busy = [], -1
    for q in take:
        if D["i"][q] < busy:
            continue
        s = side[q]
        x = D["xl"][q] if s == 1 else D["xs"][q]
        rows.append((D["t_entry"][q], D["t_exit"][q, 0 if s == 1 else 1], s,
                     (D["gross_l"] if s == 1 else D["gross_s"])[q], (D["net_l"] if s == 1 else D["net_s"])[q]))
        busy = x
    return pd.DataFrame(rows, columns=["entry_time", "exit_time", "side", "gross", "net"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--other")
    ap.add_argument("--data-dir", default="mt5_export")
    ap.add_argument("--k", type=float, default=2.0, help="barrière = k × vol EWM 15m")
    ap.add_argument("--horizon", type=int, default=16, help="barres de 15 min (16 = 4 h)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.05, 0.10], help="PnL net attendu minimal (en R)")
    ap.add_argument("--min-train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--embargo-hours", type=float, default=24)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-cross", action="store_true")
    ap.add_argument("--v2", action="store_true", help="ajoute features_v2 (indicateurs, microstructure, BTC, calendrier)")
    ap.add_argument("--out", default="results_ml")
    a = ap.parse_args()

    costs = DEFAULT_COSTS[a.pair]
    cache = Path(a.out) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    m1 = load_mt5_csv(f"{a.data_dir}/{a.pair}_M1.csv")
    m1o = load_mt5_csv(f"{a.data_dir}/{a.other}_M1.csv") if a.other and not a.no_cross else None
    fcache = cache / f"feat_{a.pair}_{'x' if m1o is not None else 'solo'}{'_v2' if a.v2 else ''}.pkl"
    if fcache.exists():
        X, base, vol = pd.read_pickle(fcache)
    else:
        X, base, vol = build_features(m1, m1o)
        if a.v2:
            from features_v2 import extra_features
            m1b = load_mt5_csv(f"{a.data_dir}/BTCUSD_M1.csv")
            X = pd.concat([X, extra_features(m1, base, m1b)], axis=1)
        pd.to_pickle((X, base, vol), fcache)
    ts = base.index.to_numpy("datetime64[ns]").astype("int64")
    print(f"{a.pair}: {len(base):,} barres 15 min | {len(X.columns)} features | {base.index[0]} -> {base.index[-1]}")

    rl, rs, xl, xs = label_events(base, vol, a.k, a.horizon)
    n = len(base)
    ok = np.isfinite(rl) & np.isfinite(rs) & X.notna().all(axis=1).to_numpy()
    i = np.flatnonzero(ok)
    e = i + 1
    b = a.k * vol[i]
    gross_l, gross_s = rl[i], rs[i]
    net_l = net_returns(rl[i], e, xl[i], ts, costs["comm"], costs["slip"], costs["swap_long"])
    net_s = net_returns(rs[i], e, xs[i], ts, costs["comm"], costs["slip"], costs["swap_short"])
    yl, ys = net_l / b, net_s / b
    Xv = X.to_numpy(np.float32)[i]
    t_dec = (base.index[i] + BAR).to_numpy()
    t_exit_l = (base.index[xl[i]] + BAR).to_numpy()
    t_exit_s = (base.index[xs[i]] + BAR).to_numpy()
    t_end = np.maximum(t_exit_l, t_exit_s)
    print(f"{len(i):,} événements | coût A/R fixe {2 * (costs['comm'] + costs['slip']):.2f} bps | barrière médiane {np.median(b) * 1e4:.1f} bps "
          f"| E[net long] {net_l.mean() * 1e4:.2f} bps, E[net short] {net_s.mean() * 1e4:.2f} bps (non conditionnel)")

    # walk-forward
    P = np.full((len(i), 2), np.nan)
    fold = np.full(len(i), -1)
    imps = []
    t0 = pd.Timestamp(t_dec.min())
    ts_ = (t0 + pd.DateOffset(months=a.min_train_months)).to_period("M").to_timestamp()
    f = 0
    while ts_ < pd.Timestamp(t_dec.max()):
        te = ts_ + pd.DateOffset(months=a.test_months)
        tr = np.flatnonzero(t_end < np.datetime64(ts_ - pd.Timedelta(hours=a.embargo_hours)))
        ev = np.flatnonzero((t_dec >= np.datetime64(ts_)) & (t_dec < np.datetime64(te)))
        if len(tr) > 5000 and len(ev):
            (pl, il), (ps, is_) = fit_predict(Xv[tr], (yl[tr], ys[tr]), Xv[ev], a)
            P[ev, 0], P[ev, 1], fold[ev] = pl, ps, f
            imps.append(il + is_)
        ts_, f = te, f + 1
    oos = fold >= 0
    D = dict(i=i[oos], xl=xl[i][oos], xs=xs[i][oos], t_entry=(base.index[e[oos]]).to_numpy(),
             t_exit=np.stack([t_exit_l[oos], t_exit_s[oos]], axis=1),
             gross_l=gross_l[oos], gross_s=gross_s[oos], net_l=net_l[oos], net_s=net_s[oos])
    pl, ps = P[oos, 0], P[oos, 1]
    start, end = pd.Timestamp(t_dec[oos].min()), pd.Timestamp(t_dec[oos].max())

    # diagnostic : PnL net réalisé par décile du score prédit (tous événements OOS)
    score = np.maximum(pl, ps)
    side = np.where(pl >= ps, 1, -1)
    real_bps = np.where(side == 1, net_l[oos], net_s[oos]) * 1e4
    real_gross = np.where(side == 1, gross_l[oos], gross_s[oos]) * 1e4
    b_oos = b[oos] * 1e4
    dec = pd.qcut(score, 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"decile": dec, "score_R": score, "net_bps": real_bps, "gross_bps": real_gross, "barriere_bps": b_oos}
                       ).groupby("decile").agg(score_moy=("score_R", "mean"), barriere_bps=("barriere_bps", "mean"),
                                               gross_bps_realise=("gross_bps", "mean"), net_bps_realise=("net_bps", "mean"),
                                               n=("net_bps", "size")).round(3)
    # IC en bps (le score en R est contaminé par le rapport coût/barrière) : score prédit converti en bps vs net réalisé
    ic = float(spearmanr(score * b_oos, real_bps)[0])
    print(f"\nOOS {start.date()} -> {end.date()} | {oos.sum():,} événements | IC en bps (Spearman prédit/réalisé) = {ic:.4f}")
    print("Par décile du score prédit (0 = pire, 9 = meilleur), PnL réalisé du côté choisi :")
    print(tab.T.to_string())

    report = {"pair": a.pair, "k": a.k, "horizon": a.horizon, "oos": [str(start), str(end)], "ic_bps": round(ic, 4),
              "decile_gross_bps": tab["gross_bps_realise"].to_dict(), "decile_net_bps": tab["net_bps_realise"].to_dict(),
              "thresholds": {}}
    allt = {}
    for thr in a.thresholds:
        tr_ = run_simulation(D, pl, ps, thr)
        s = common.summarize(tr_, start, end)
        report["thresholds"][str(thr)] = s
        allt[thr] = tr_
        print(f"\nseuil {thr:+.2f} R :", {k_: s.get(k_) for k_ in ("n_trades", "trades_per_day", "hit_rate", "gross_bps", "net_bps", "sharpe", "psr", "max_dd_pct")})
    imp = pd.Series(np.mean(imps, axis=0), index=list(X.columns)).sort_values(ascending=False)
    report["top_features"] = (imp / imp.sum()).head(8).round(4).to_dict()
    print("\nTop features :", report["top_features"])
    out = Path(a.out)
    tag = f"{a.pair}_k{a.k:g}_H{a.horizon}"
    (out / f"report_{tag}.json").write_text(json.dumps(report, indent=2, default=str))
    for thr, t_ in allt.items():
        t_.to_csv(out / f"trades_{tag}_thr{thr:g}.csv", index=False)


if __name__ == "__main__":
    main()
