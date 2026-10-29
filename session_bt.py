"""Stratégie d'ouverture de session (OPR), pré-enregistrée.

Sessions (heure locale, DST géré) : Tokyo 09:00, Londres 08:00, New York 09:30.
OPR = plage des N premières minutes (N = 15 ou 30). Ensuite, pendant 120 min, sur barres M5 clôturées :
  brk : cassure  = clôture au-delà de l'OPR, dans le sens de la tendance M15 (EMA20/EMA50 + pente EMA50), stop = milieu de l'OPR
  ret : rebond   = après cassure, retour sur le bord de l'OPR et clôture dans le sens de la cassure, stop = sous/sur la barre de rebond
  rev : fausse cassure = clôture qui revient dans l'OPR après cassure, stop = extrême de la cassure
  pat : si 1, exige un motif de bougie M5 (englobante ou pin bar/marteau) dans le sens du trade
Trade : entrée à l'ouverture de la barre suivante (ask/bid), TP = 2R (plafond demandé), sortie temporelle 240 min après l'ouverture.
Risque : 0,5 % du capital initial par trade (R), 2 trades max par session, arrêt du jour à -2,5 %, arrêt de la semaine à -6 %
(marges sous les limites FTMO de 5 % / 10 %). Filtre : R >= 3 x coût aller-retour. Coûts bid/ask réels + commission + glissement.
"""
import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import common
from common import resample
from data import load_mt5_csv

SESSIONS = {"tokyo": ("Asia/Tokyo", 9, 0), "london": ("Europe/London", 8, 0), "ny": ("America/New_York", 9, 30),
            "frankfurt": ("Europe/Berlin", 9, 0)}
# sessions testées par instrument : forex/BTC = les 3 sessions ; indices cash = leur propre ouverture
PAIR_SESSIONS = {"US100.cash": ["ny"], "US500.cash": ["ny"], "US30.cash": ["ny"], "GER40.cash": ["frankfurt"],
                 "UK100.cash": ["london"], "JP225.cash": ["tokyo"]}
COSTS = {"EURUSD": dict(comm=0.45, slip=0.2, lev=30.0), "GBPUSD": dict(comm=0.38, slip=0.2, lev=30.0),
         "BTCUSD": dict(comm=6.5, slip=1.0, lev=2.0)}
for _k in ("US100.cash", "US500.cash", "US30.cash", "GER40.cash", "UK100.cash", "JP225.cash"):
    COSTS[_k] = dict(comm=0.0, slip=0.2, lev=20.0)          # indices FTMO : 0 commission, spread réel dans les données
WINDOW_MIN, HOLD_MIN, TP_R = 120, 240, 2.0
RISK, MAX_PER_SESSION, DAY_HALT, WEEK_HALT, MIN_R_COSTS = 0.005, 2, 0.025, 0.06, 3.0


def prep(m1):
    m5 = resample(m1, "5min")
    o, h, l, c = (m5[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    body, rng = np.abs(c - o), h - l
    up, lo = h - np.maximum(o, c), np.minimum(o, c) - l
    po, pc, pb = np.r_[np.nan, o[:-1]], np.r_[np.nan, c[:-1]], np.r_[np.nan, body[:-1]]
    bull = ((c > o) & (pc < po) & (c >= po) & (o <= pc) & (body > pb)) | ((lo >= 2 * body) & (lo >= 0.6 * rng) & (up <= 0.25 * rng))
    bear = ((c < o) & (pc > po) & (c <= po) & (o >= pc) & (body > pb)) | ((up >= 2 * body) & (up >= 0.6 * rng) & (lo <= 0.25 * rng))
    m15 = resample(m1, "15min")
    e20, e50 = m15["close"].ewm(span=20, adjust=False).mean(), m15["close"].ewm(span=50, adjust=False).mean()
    tr = pd.Series(np.where((e20 > e50) & (e50 > e50.shift(4)), 1, np.where((e20 < e50) & (e50 < e50.shift(4)), -1, 0)), index=m15.index)
    tr.index = tr.index + pd.Timedelta("15min")                      # connu seulement à la clôture de la barre M15
    trend = tr.reindex(m5.index + pd.Timedelta("5min"), method="ffill").fillna(0).to_numpy()
    t = m5.index.values.astype("datetime64[s]").astype("int64")
    return dict(t=t, o=o, h=h, l=l, c=c, s=m5["spread_px"].to_numpy(float), bull=bull, bear=bear, trend=trend)


def session_opens(first, last, names=None):
    out = []
    for name, (tz, hh, mm) in SESSIONS.items():
        if names is not None and name not in names:
            continue
        for d in pd.date_range(first.normalize() - pd.Timedelta(days=1), last.normalize() + pd.Timedelta(days=1), freq="B"):
            ts = pd.Timestamp(d.year, d.month, d.day, hh, mm, tz=tz).tz_convert("UTC").tz_localize(None)
            out.append((ts, name))
    return sorted(out)


def manage(P, j, d, stop, tgt, hold_end, entry, costs_frac):
    """Suit la position à partir de la barre j. d=+1 long (stop sur bid), d=-1 short (stop sur ask)."""
    n = len(P["t"])
    for k in range(j, min(hold_end + 1, n)):
        o, h, l, s = P["o"][k], P["h"][k], P["l"][k], P["s"][k]
        if k >= hold_end:
            return (o if d == 1 else o + s), k, "time"
        if d == 1:
            if k > j and o <= stop: return o, k, "gap"
            if l <= stop: return stop, k, "stop"
            if h >= tgt: return tgt, k, "tp"
        else:
            if k > j and o + s >= stop: return o + s, k, "gap"
            if h + s >= stop: return stop, k, "stop"
            if l + s <= tgt: return tgt, k, "tp"
    k = min(hold_end, n - 1)
    return (P["o"][k] if d == 1 else P["o"][k] + P["s"][k]), k, "time"


def run_config(P, opens, pair, N, modes, pat):
    cs = COSTS[pair]
    cost_frac = (2 * cs["comm"] + 2 * cs["slip"]) / 1e4
    t, o, h, l, c, s = P["t"], P["o"], P["h"], P["l"], P["c"], P["s"]
    n_opr, W, hold = N // 5, WINDOW_MIN // 5, HOLD_MIN // 5
    trades, day_pnl, week_pnl = [], {}, {}
    spread_bps_med = float(np.median(s / c) * 1e4)
    min_R_bps = MIN_R_COSTS * (2 * cs["comm"] + 2 * cs["slip"] + spread_bps_med)
    for ts, name in opens:
        ots = int(ts.value // 10**9)
        i0 = int(np.searchsorted(t, ots))
        if i0 + hold + 2 >= len(t) or t[i0] != ots or t[i0 + n_opr - 1] != ots + (n_opr - 1) * 300 or t[i0 + hold] != ots + hold * 300:
            continue
        H, L = h[i0:i0 + n_opr].max(), l[i0:i0 + n_opr].min()
        rng = H - L
        if rng <= 0:
            continue
        mid = (H + L) / 2
        broke_up = broke_dn = False
        ext_hi, ext_lo = -np.inf, np.inf
        cnt, i, busy_until = 0, i0 + n_opr, -1
        while i < i0 + n_opr + W and cnt < MAX_PER_SESSION:
            d_key, w_key = ts.normalize(), (ts.isocalendar().year, ts.isocalendar().week)
            if day_pnl.get(d_key, 0.0) <= -DAY_HALT or week_pnl.get(w_key, 0.0) <= -WEEK_HALT:
                break
            if i <= busy_until:
                i += 1
                continue
            tr, sig, stop, kind = P["trend"][i], 0, None, ''
            ci, hi, li = c[i], h[i], l[i]
            if broke_up: ext_hi = max(ext_hi, hi)
            if broke_dn: ext_lo = min(ext_lo, li)
            long_pat = (not pat) or P["bull"][i]
            short_pat = (not pat) or P["bear"][i]
            if "ret" in modes and broke_up and li <= H + 0.1 * rng and ci > H and tr >= 0 and long_pat:
                sig, stop, broke_up = 1, li - 0.15 * rng, False; kind = 'ret'
            elif "ret" in modes and broke_dn and hi >= L - 0.1 * rng and ci < L and tr <= 0 and short_pat:
                sig, stop, broke_dn = -1, hi + 0.15 * rng, False; kind = 'ret'
            elif "rev" in modes and broke_up and ci < H and short_pat:
                sig, stop, broke_up = -1, ext_hi + 0.05 * rng, False; kind = 'rev'
            elif "rev" in modes and broke_dn and ci > L and long_pat:
                sig, stop, broke_dn = 1, ext_lo - 0.05 * rng, False; kind = 'rev'
            elif "brk" in modes and (not broke_up) and (not broke_dn) and ci > H and ci - H <= rng and tr >= 1 and long_pat:
                sig, stop = 1, mid; kind = 'brk'
            elif "brk" in modes and (not broke_up) and (not broke_dn) and ci < L and L - ci <= rng and tr <= -1 and short_pat:
                sig, stop = -1, mid; kind = 'brk'
            if ci > H and not broke_dn and not broke_up and sig != -1:
                broke_up, ext_hi = True, max(ext_hi, hi)
            if ci < L and not broke_up and not broke_dn and sig != 1:
                broke_dn, ext_lo = True, min(ext_lo, li)
            if sig:
                j = i + 1
                entry = o[j] + s[j] if sig == 1 else o[j]
                R = (entry - stop) if sig == 1 else (stop - entry)
                if R > 0 and R / entry * 1e4 >= min_R_bps:
                    tgt = entry + sig * TP_R * R
                    hold_end = i0 + hold
                    if j < hold_end:
                        ex, k, why = manage(P, j, sig, stop, tgt, hold_end, entry, cost_frac)
                        r_net = sig * (ex - entry) / entry - cost_frac
                        r_gross = sig * (ex - entry) / entry
                        notional = min(RISK / (R / entry), cs["lev"])
                        pnl = notional * r_net
                        te = pd.Timestamp(int(t[k]), unit="s")
                        trades.append(dict(entry_time=pd.Timestamp(int(t[j]), unit="s"), exit_time=te, session=name, dir=sig, kind=kind, opr_bps=rng / mid * 1e4, trend=tr, k_after=i - i0, spread_bps=s[j] / o[j] * 1e4,
                                           R_bps=R / entry * 1e4, why=why, R_net=r_net / (R / entry), R_gross=r_gross / (R / entry),
                                           net=pnl))
                        day_pnl[te.normalize()] = day_pnl.get(te.normalize(), 0.0) + pnl
                        wk = (te.isocalendar().year, te.isocalendar().week)
                        week_pnl[wk] = week_pnl.get(wk, 0.0) + pnl
                        cnt += 1
                        busy_until = k
            i += 1
    return pd.DataFrame(trades)


def stats(tr, start, end):
    if tr.empty:
        return dict(n=0)
    d = common.daily_returns(tr, start, end)
    wk = tr.groupby([tr["exit_time"].dt.isocalendar().year, tr["exit_time"].dt.isocalendar().week])["net"].sum()
    cum = np.cumsum(d)
    return dict(n=len(tr), hit=round(float((tr["net"] > 0).mean()), 3), R_brut=round(float(tr["R_gross"].mean()), 3),
                R_net=round(float(tr["R_net"].mean()), 3), sharpe=round(common.sharpe(d), 2),
                total_pct=round(float(tr["net"].sum() * 100), 1), pire_jour_pct=round(float(d.min() * 100), 2),
                pire_semaine_pct=round(float(wk.min() * 100), 2), maxdd_pct=round(float((cum - np.maximum.accumulate(cum)).min() * 100), 1),
                psr=round(common.psr(d), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="mt5_export")
    ap.add_argument("--pairs", default="EURUSD,GBPUSD,BTCUSD")
    ap.add_argument("--split", default="2025-07-01")
    ap.add_argument("--out", default="results_sess")
    ap.add_argument("--prior-trials", type=int, default=0, help="essais déjà comptés (DSR)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    rows, series = [], {}
    modes_grid = {"brk": {"brk"}, "ret": {"ret"}, "rev": {"rev"}, "all": {"brk", "ret", "rev"}}
    for pair in a.pairs.split(","):
        m1 = load_mt5_csv(f"{a.data_dir}/{pair}_M1.csv")
        P = prep(m1)
        first, last = m1.index[0], m1.index[-1]
        opens = session_opens(first + pd.Timedelta(days=30), last - pd.Timedelta(days=1), PAIR_SESSIONS.get(pair))
        start, end, split = first + pd.Timedelta(days=30), last, pd.Timestamp(a.split)
        print(f"{pair}: {len(m1):,} M1 | {len(opens)} ouvertures | {start.date()} -> {end.date()}", flush=True)
        for N, mname, pat in product((15, 30), modes_grid, (0, 1)):
            tr = run_config(P, opens, pair, N, modes_grid[mname], pat)
            tag = f"{pair}_N{N}_{mname}_pat{pat}"
            if len(tr): tr.to_csv(out / f"trades_{tag}.csv", index=False)
            full, tr_in, tr_out = stats(tr, start, end), stats(tr[tr["exit_time"] < split] if len(tr) else tr, start, split), \
                stats(tr[tr["exit_time"] >= split] if len(tr) else tr, split, end)
            series[tag] = common.daily_returns(tr, start, end) if len(tr) else np.zeros((end.normalize() - start.normalize()).days + 1)
            rows.append(dict(tag=tag, pair=pair, N=N, mode=mname, pat=pat, **{f"full_{k}": v for k, v in full.items()},
                             **{f"in_{k}": v for k, v in tr_in.items() if k in ("n", "R_net", "sharpe")},
                             **{f"out_{k}": v for k, v in tr_out.items() if k in ("n", "R_net", "R_brut", "sharpe", "total_pct", "pire_jour_pct", "pire_semaine_pct", "maxdd_pct")}))
    T = pd.DataFrame(rows)
    T.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    cols = ["tag", "full_n", "full_hit", "full_R_brut", "full_R_net", "full_sharpe", "full_pire_jour_pct", "full_pire_semaine_pct", "in_sharpe", "out_n", "out_R_net", "out_sharpe"]
    print(T[cols].to_string(index=False))
    act = {k: v for k, v in series.items() if (v != 0).sum() >= 20}
    sr = np.array([v.mean() / v.std(ddof=1) for v in act.values()])
    print(f"\nEssais actifs : {len(sr)}")
    for pair in a.pairs.split(","):
        sub = T[(T.pair == pair) & (T.tag.isin(act))]
        if sub.empty: continue
        b = sub.sort_values("in_sharpe", ascending=False).iloc[0]
        d = series[b.tag]
        print(f"{pair}: meilleur en échantillon = {b.tag} (Sharpe in {b.in_sharpe}) -> hors échantillon : n={b.out_n}, R net={b.out_R_net}, "
              f"Sharpe={b.out_sharpe}, pire jour {b.out_pire_jour_pct}%, pire semaine {b.out_pire_semaine_pct}% | DSR plein échantillon={common.dsr(d, sr, len(sr) + a.prior_trials):.4f}")
