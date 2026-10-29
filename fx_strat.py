"""Stratégies intraday paires USD, reprises de zéro. Hypothèses de microstructure pré-enregistrées (3 familles, 16 configs/paire).

Indicateurs retenus : ATR(14) M15 (unité de risque), EMA20/50 M15, RSI(14) M15, EMA50/200 + ADX(14) H1 (régime), plage asiatique.
Fenêtre d'entrée : ouverture de Londres -> 12:00 New York. Sortie forcée : 16:00 New York ou 6 h de détention.
Famille A  asian  : cassure de la plage asiatique (follow) ou fausse cassure (fade) à l'ouverture de Londres.
Famille B  pull   : repli dans la tendance H1 (EMA50>EMA200) : RSI M15 sous un seuil puis reprise au-dessus de l'EMA20 M15.
Famille C  mr     : excès de l'écart à l'EMA50 M15 (z en ATR) + RSI extrême, en régime sans tendance (ADX H1 < 22) : retour à la moyenne.
Trade : entrée à l'ouverture de la barre M5 suivante (ask/bid), stop ATR, TP <= 2R, break-even à +1R, filtre R >= 3 x coût aller-retour.
Risque 0,5 %/trade, 3 trades/jour max, arrêt du jour à -2,5 %, de la semaine à -6 %. Développement : avant 2026-04-29 (holdout verrouillé).
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import resample
from data import load_mt5_csv
from features_v2 import _adx, _rsi
from session_bt import COSTS

HOLDOUT = pd.Timestamp("2026-04-29")
RISK, MAX_DAY, DAY_HALT, WEEK_HALT, MIN_R_COSTS, HOLD_BARS = 0.005, 3, 0.025, 0.06, 3.0, 72


def prep(m1):
    m5 = resample(m1, "5min")
    P = {k: m5[k].to_numpy(float) for k in ("open", "high", "low", "close")}
    P["s"], P["idx"] = m5["spread_px"].to_numpy(float), m5.index
    P["t"] = m5.index.values.astype("datetime64[s]").astype("int64")
    dec = m5.index + pd.Timedelta("5min")
    m15 = resample(m1, "15min")
    c = m15["close"]
    tr = pd.concat([m15["high"] - m15["low"], (m15["high"] - c.shift()).abs(), (m15["low"] - c.shift()).abs()], axis=1).max(axis=1)
    f15 = pd.DataFrame({"atr": tr.rolling(14).mean(), "e20": c.ewm(span=20, adjust=False).mean(), "e50": c.ewm(span=50, adjust=False).mean(), "rsi": _rsi(c)})
    f15.index = f15.index + pd.Timedelta("15min")
    h1 = resample(m1, "1h")
    adx, _ = _adx(h1)
    fh = pd.DataFrame({"h_e50": h1["close"].ewm(span=50, adjust=False).mean(), "h_e200": h1["close"].ewm(span=200, adjust=False).mean(), "h_adx": adx})
    fh.index = fh.index + pd.Timedelta("1h")
    F = pd.concat([f15.reindex(dec, method="ffill").set_axis(m5.index), fh.reindex(dec, method="ffill").set_axis(m5.index)], axis=1)
    for k in F.columns:
        P[k] = F[k].to_numpy(float)
    # rsi M15 : minimum sur les 4 dernières barres M15 (= 12 barres M5)
    P["rsi_min12"] = pd.Series(P["rsi"]).rolling(12, min_periods=1).min().to_numpy()
    P["rsi_max12"] = pd.Series(P["rsi"]).rolling(12, min_periods=1).max().to_numpy()
    # fenêtres horaires
    loc_ny = m5.index.tz_localize("UTC").tz_convert("America/New_York")
    P["ny_min"] = (loc_ny.hour * 60 + loc_ny.minute).to_numpy()
    loc_ldn = m5.index.tz_localize("UTC").tz_convert("Europe/London")
    P["ldn_min"] = (loc_ldn.hour * 60 + loc_ldn.minute).to_numpy()
    P["date"] = m5.index.normalize()
    return P


def window(P):
    """Entrées autorisées : 08:00 Londres -> 12:00 New York."""
    n = len(P["t"])
    ldn_ok = P["ldn_min"] >= 8 * 60
    ny_ok = P["ny_min"] < 12 * 60
    # entre 08:00 Londres et 12:00 NY : même jour calendaire UTC (Londres 07-08h UTC < NY midi 16-17h UTC)
    return ldn_ok & ny_ok & ((P["ldn_min"] < 17 * 60))


def signals_asian(P, mode):
    """Retourne liste de (i_décision, dir, stop_prix)."""
    out = []
    ok = window(P)
    dates = pd.Series(P["date"])
    for d, grp in dates.groupby(dates):
        ii = grp.index.to_numpy()
        ldn = P["ldn_min"][ii]
        asia = ii[(P["idx"][ii].hour * 60 + P["idx"][ii].minute) < 6 * 60 + 55]    # 00:00 -> ~07:00 UTC, avant l'ouverture de Londres (été/hiver)
        asia = asia[ldn[:len(asia)] < 8 * 60] if len(asia) else asia
        if len(asia) < 60:
            continue
        hi, lo = P["high"][asia].max(), P["low"][asia].min()
        start = ii[(ldn >= 8 * 60)]
        if len(start) == 0:
            continue
        start = start[:36]                                  # 3 h après l'ouverture de Londres
        atr0 = P["atr"][start[0]]
        if not np.isfinite(atr0) or not (hi - lo >= 1.0 * atr0) or not (hi - lo <= 6 * atr0):
            continue
        mid, side, ext = (hi + lo) / 2, 0, None
        for i in start:
            c = P["close"][i]
            a = P["atr"][i]
            if not np.isfinite(a):
                continue
            if mode == "follow":
                if c > hi + 0.1 * (hi - lo) * 0 and c > hi:
                    out.append((i, 1, c - np.clip(c - mid, a, 3 * a))); break
                if c < lo:
                    out.append((i, -1, c + np.clip(mid - c, a, 3 * a))); break
            else:  # fade : après une sortie de plage, retour clôturé dans la plage
                if side == 0:
                    if c > hi: side, ext = 1, P["high"][i]
                    elif c < lo: side, ext = -1, P["low"][i]
                else:
                    ext = max(ext, P["high"][i]) if side == 1 else min(ext, P["low"][i])
                    if side == 1 and c < hi:
                        out.append((i, -1, c + np.clip(ext + 0.1 * a - c, a, 3 * a))); break
                    if side == -1 and c > lo:
                        out.append((i, 1, c - np.clip(c - (ext - 0.1 * a), a, 3 * a))); break
    return [x for x in out if ok[x[0]]]


def signals_pull(P, rsi_thr, inv=False):
    c, o, a = P["close"], P["open"], P["atr"]
    up = (P["h_e50"] > P["h_e200"]) & (c > P["h_e200"])
    dn = (P["h_e50"] < P["h_e200"]) & (c < P["h_e200"])
    long_ = up & (P["rsi_min12"] < rsi_thr) & (c > P["e20"]) & (c > o)
    short_ = dn & (P["rsi_max12"] > 100 - rsi_thr) & (c < P["e20"]) & (c < o)
    ok = window(P) & np.isfinite(a)
    out, last = [], -10**9
    for i in np.flatnonzero((long_ | short_) & ok):
        if i - last < 12: continue
        d = 1 if long_[i] else -1
        if inv: d = -d
        out.append((i, d, c[i] - d * 1.5 * a[i])); last = i
    return out


def signals_mr(P, zthr):
    c, o, a = P["close"], P["open"], P["atr"]
    z = (c - P["e50"]) / a
    rng = P["h_adx"] < 22
    short_ = rng & (z > zthr) & (P["rsi"] > 70) & (c < o)
    long_ = rng & (z < -zthr) & (P["rsi"] < 30) & (c > o)
    ok = window(P) & np.isfinite(a)
    out, last = [], -10**9
    for i in np.flatnonzero((long_ | short_) & ok):
        if i - last < 12: continue
        d = 1 if long_[i] else -1
        out.append((i, d, c[i] - d * 1.5 * a[i])); last = i
    return out


def simulate(P, sigs, pair, tp_r):
    cs = COSTS[pair]
    cost_frac = (2 * cs["comm"] + 2 * cs["slip"]) / 1e4
    spread_med = float(np.median(P["s"] / P["close"]) * 1e4)
    min_R_bps = MIN_R_COSTS * (2 * cs["comm"] + 2 * cs["slip"] + spread_med)
    o, h, l, s, t, n = P["open"], P["high"], P["low"], P["s"], P["t"], len(P["t"])
    trades, busy, day_n, day_pnl, week_pnl = [], -1, {}, {}, {}
    for i, d, stop in sorted(sigs):
        if i <= busy or i + 2 >= n:
            continue
        ts = P["idx"][i]
        dk, wk = ts.normalize(), (ts.isocalendar().year, ts.isocalendar().week)
        if day_n.get(dk, 0) >= MAX_DAY or day_pnl.get(dk, 0.0) <= -DAY_HALT or week_pnl.get(wk, 0.0) <= -WEEK_HALT:
            continue
        j = i + 1
        entry = o[j] + s[j] if d == 1 else o[j]
        R = (entry - stop) if d == 1 else (stop - entry)
        if R <= 0 or R / entry * 1e4 < min_R_bps:
            continue
        tgt, cur_stop, be_done = entry + d * tp_r * R, stop, False
        ny_close = (P["ny_min"][j:j + HOLD_BARS + 1] >= 16 * 60)
        hold_end = j + (int(np.argmax(ny_close)) if ny_close.any() else HOLD_BARS)
        hold_end = min(hold_end, j + HOLD_BARS, n - 1)
        ex = None
        for k in range(j, hold_end + 1):
            ok_, hk, lk, sk = o[k], h[k], l[k], s[k]
            if k >= hold_end and k > j:
                ex, why = (ok_ if d == 1 else ok_ + sk), "time"; break
            if d == 1:
                if k > j and ok_ <= cur_stop: ex, why = ok_, "gap"; break
                if lk <= cur_stop: ex, why = cur_stop, "stop"; break
                if hk >= tgt: ex, why = tgt, "tp"; break
                if not be_done and hk >= entry + R: cur_stop, be_done = entry, True
            else:
                if k > j and ok_ + sk >= cur_stop: ex, why = ok_ + sk, "gap"; break
                if hk + sk >= cur_stop: ex, why = cur_stop, "stop"; break
                if lk + sk <= tgt: ex, why = tgt, "tp"; break
                if not be_done and lk + sk <= entry - R: cur_stop, be_done = entry, True
        if ex is None:
            k = hold_end; ex, why = (o[k] if d == 1 else o[k] + s[k]), "time"
        r_g = d * (ex - entry) / entry
        r_n = r_g - cost_frac
        pnl = min(RISK / (R / entry), cs["lev"]) * r_n
        te = P["idx"][k]
        trades.append(dict(entry_time=P["idx"][j], exit_time=te, dir=d, R_bps=R / entry * 1e4, why=why,
                           R_gross=r_g / (R / entry), R_net=r_n / (R / entry), net=pnl))
        busy = k
        day_n[dk] = day_n.get(dk, 0) + 1
        te_d = te.normalize()
        day_pnl[te_d] = day_pnl.get(te_d, 0.0) + pnl
        w2 = (te.isocalendar().year, te.isocalendar().week)
        week_pnl[w2] = week_pnl.get(w2, 0.0) + pnl
    return pd.DataFrame(trades)


def summarize(tr, start, end):
    if tr.empty:
        return dict(n=0)
    d = common.daily_returns(tr, start, end)
    wk = tr.groupby([tr["exit_time"].dt.isocalendar().year, tr["exit_time"].dt.isocalendar().week])["net"].sum()
    x = tr["R_net"]
    t = x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 2 else 0
    return dict(n=len(tr), hit=round(float((tr.net > 0).mean()), 3), R_brut=round(float(tr.R_gross.mean()), 3), R_net=round(float(x.mean()), 3),
                t=round(float(t), 2), sharpe=round(common.sharpe(d), 2), pire_jour=round(float(d.min() * 100), 2),
                pire_sem=round(float(wk.min() * 100), 2), tot=round(float(tr.net.sum() * 100), 1))


CONFIGS = ([("A_follow", "asian", "follow", tp) for tp in (1.5, 2.0)] + [("A_fade", "asian", "fade", tp) for tp in (1.5, 2.0)] +
           [(f"B_rsi{r}", "pull", r, tp) for r in (35, 40, 45) for tp in (1.5, 2.0)] +
           [(f"C_z{z}", "mr", z, tp) for z in (2.0, 2.5, 3.0) for tp in (1.0, 2.0)])
if __import__("os").environ.get("FX_D"):      # famille D (ajoutée APRÈS avoir vu B : essais comptés à part) : fade du rebond contre la tendance H1
    CONFIGS = [(f"D_rsi{r}", "pullinv", r, tp) for r in (35, 40, 45) for tp in (1.5, 2.0)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="mt5_export")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD")
    ap.add_argument("--prior-trials", type=int, default=0)
    ap.add_argument("--out", default="results_fx")
    a = ap.parse_args()
    Path(a.out).mkdir(exist_ok=True)
    rows, series = [], {}
    split = pd.Timestamp("2025-07-01")
    for pair in a.pairs.split(","):
        if pair not in COSTS:
            COSTS[pair] = dict(comm=0.45, slip=0.2, lev=30.0)
        m1 = load_mt5_csv(f"{a.data_dir}/{pair}_M1.csv")
        m1 = m1[m1.index < HOLDOUT]
        P = prep(m1)
        start, end = m1.index[0] + pd.Timedelta(days=40), m1.index[-1]
        cache = {}
        for name, fam, par, tp in CONFIGS:
            key = (fam, par)
            if key not in cache:
                cache[key] = signals_asian(P, par) if fam == "asian" else signals_pull(P, par) if fam == "pull" else signals_pull(P, par, True) if fam == "pullinv" else signals_mr(P, par)
            tr = simulate(P, cache[key], pair, tp)
            tag = f"{pair}_{name}_tp{tp}"
            if len(tr): tr.to_csv(f"{a.out}/trades_{tag}.csv", index=False)
            full = summarize(tr, start, end)
            din = summarize(tr[tr.exit_time < split] if len(tr) else tr, start, split)
            dout = summarize(tr[tr.exit_time >= split] if len(tr) else tr, split, end)
            series[tag] = common.daily_returns(tr, start, end) if len(tr) else np.zeros((end.normalize() - start.normalize()).days + 1)
            rows.append(dict(tag=tag, **{f"f_{k}": v for k, v in full.items()}, in_n=din.get("n"), in_Rnet=din.get("R_net"), in_sh=din.get("sharpe"),
                             out_n=dout.get("n"), out_Rnet=dout.get("R_net"), out_sh=dout.get("sharpe")))
            print(f"{tag:28s} {full}", flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(f"{a.out}/summary.csv", index=False)
    act = {k: v for k, v in series.items() if (v != 0).sum() >= 20}
    sr = np.array([v.mean() / v.std(ddof=1) for v in act.values()])
    print(f"\nEssais actifs : {len(sr)}")
    if len(sr):
        bk = max(act, key=lambda k: act[k].mean() / max(act[k].std(ddof=1), 1e-12))
        print(f"Meilleur Sharpe : {bk} -> {common.sharpe(act[bk]):.2f} | DSR (essais corrigés, +{a.prior_trials} antérieurs) = {common.dsr(act[bk], sr, len(sr) + a.prior_trials):.4f}")
