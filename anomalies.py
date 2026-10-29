"""Batterie d'anomalies documentées sur indices cash FTMO (pré-enregistrée ; période de développement uniquement).

Période : 2023-02 -> 2026-04-28 (les 6 derniers mois sont VERROUILLÉS pour la validation finale).
Coûts : spread réel de la barre d'entrée et de sortie + 0,2 bps de glissement par côté, 0 commission, + swap si position tenue la nuit.
Stratégies (une position par jour au plus, rendement net en bps du notionnel) :
  imom_ovn  : momentum intrajournalier (Gao et al.) : signe du rendement clôture veille -> +30 min, position les 30 dernières minutes
  imom_open : idem avec le rendement ouverture -> +30 min seulement
  ovn_long  : long de la clôture à l'ouverture suivante (dérive de nuit)
  gap_fade  : contre le gap d'ouverture si |gap| > 1 écart-type glissant, sortie à midi
  gap_go    : avec le gap d'ouverture (même filtre)
  open30    : rendement des 30 premières minutes prolongé jusqu'à midi
  trend_d   : tendance journalière (signe du rendement sur 50 j), tenue de clôture à clôture (swap compté)
  long_only : long ouverture -> clôture (biais haussier de l'indice, référence)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

import common
from data import load_mt5_csv

HOLDOUT = pd.Timestamp("2026-04-29")
CASH = {"US100.cash": ("America/New_York", (9, 30), (16, 0)), "US500.cash": ("America/New_York", (9, 30), (16, 0)),
        "US30.cash": ("America/New_York", (9, 30), (16, 0)), "GER40.cash": ("Europe/Berlin", (9, 0), (17, 30)),
        "UK100.cash": ("Europe/London", (8, 0), (16, 30)), "JP225.cash": ("Asia/Tokyo", (9, 0), (15, 0))}
SLIP = 0.2


def ts_utc(dates, tz, hm):
    return [pd.Timestamp(d.year, d.month, d.day, hm[0], hm[1], tz=tz).tz_convert("UTC").tz_localize(None) for d in dates]


class Px:
    def __init__(self, m1):
        self.o, self.s = m1["open"], m1["spread_px"]
        self.idx = m1.index

    def at(self, stamps):
        pos = self.idx.searchsorted(pd.DatetimeIndex(stamps))
        pos = np.clip(pos, 0, len(self.idx) - 1)
        ok = (self.idx[pos] - pd.DatetimeIndex(stamps)) <= pd.Timedelta("5min")
        return np.where(ok, self.o.to_numpy()[pos], np.nan), np.where(ok, self.s.to_numpy()[pos], np.nan)


def trade_pnl(entry, es, exit_, xs, d, swap_bps=0.0):
    """Rendement net (fraction). Long : achat ask = bid+spread, vente bid. Short : vente bid, rachat ask."""
    ent = np.where(d > 0, entry + es, entry)
    ext = np.where(d > 0, exit_, exit_ + xs)
    return d * (ext - ent) / entry - 2 * SLIP / 1e4 - swap_bps / 1e4


def run(name, m1, spec):
    tz, op, cl = spec
    px = Px(m1)
    days = pd.bdate_range(m1.index[0].normalize() + pd.Timedelta(days=40), m1.index[-1].normalize() - pd.Timedelta(days=2))
    T = lambda hm, shift=0: ts_utc(days + pd.Timedelta(days=shift), tz, hm)
    add = lambda hm, mins: ((hm[0] * 60 + hm[1] + mins) // 60, (hm[0] * 60 + hm[1] + mins) % 60)
    p_open, s_open = px.at(T(op))
    p_o30, s_o30 = px.at(T(add(op, 30)))
    p_close, s_close = px.at(T(cl))
    p_c30, s_c30 = px.at(T(add(cl, -30)))
    p_noon, s_noon = px.at(T((12, 0)) if tz != "Asia/Tokyo" else T((12, 30)))
    prev_close = pd.Series(p_close).shift(1).to_numpy()
    prev_close_next = pd.Series(p_close).shift(-1).to_numpy()
    p_open_next, s_open_next = np.r_[p_open[1:], np.nan], np.r_[s_open[1:], np.nan]
    swap = json.load(open(f"{DATA}/{name}_spec.json"))
    sw_bps = lambda side: abs(swap["swap_long" if side > 0 else "swap_short"]) * swap["point"] / np.nanmedian(p_close) * 1e4 if swap["swap_mode"] == 1 else 0.0
    sw_long, sw_short = sw_bps(1) if swap["swap_long"] < 0 else -sw_bps(1), sw_bps(-1) if swap["swap_short"] < 0 else -sw_bps(-1)
    ret_ovn = p_o30 / prev_close - 1                       # clôture veille -> +30 min
    ret_o30 = p_o30 / p_open - 1
    gap = p_open / prev_close - 1
    gsd = pd.Series(gap).rolling(60).std().shift(1).to_numpy()
    d_close = pd.Series(p_close, index=days)
    tr50 = (d_close / d_close.shift(50) - 1).shift(0).to_numpy()
    S = {}
    S["imom_ovn"] = (np.sign(ret_ovn), p_c30, s_c30, p_close, s_close, 0.0, 0.0)
    S["imom_open"] = (np.sign(ret_o30), p_c30, s_c30, p_close, s_close, 0.0, 0.0)
    S["ovn_long"] = (np.ones(len(days)), p_close, s_close, p_open_next, s_open_next, sw_long, sw_long)
    big = np.abs(gap) > gsd
    S["gap_fade"] = (np.where(big, -np.sign(gap), 0), p_o30, s_o30, p_noon, s_noon, 0.0, 0.0)
    S["gap_go"] = (np.where(big, np.sign(gap), 0), p_o30, s_o30, p_noon, s_noon, 0.0, 0.0)
    S["open30"] = (np.sign(ret_o30), p_o30, s_o30, p_noon, s_noon, 0.0, 0.0)
    S["long_only"] = (np.ones(len(days)), p_open, s_open, p_close, s_close, 0.0, 0.0)
    out = {}
    for k, (d, pe, se, px_, sx, swl, sws) in S.items():
        d = np.nan_to_num(d)
        sw = np.where(d > 0, swl, sws)
        r = trade_pnl(pe, se, px_, sx, d, sw)
        m = (d != 0) & np.isfinite(r)
        out[k] = pd.Series(np.where(m, r, np.nan), index=days)
    # tendance journalière : tenue de clôture à clôture, un jour sur deux de swap compté par nuit
    d = np.sign(np.nan_to_num(tr50))
    sw = np.where(d > 0, sw_long, sw_short)
    r = trade_pnl(p_close, s_close, prev_close_next, np.r_[s_close[1:], np.nan], d, sw)
    out["trend_d"] = pd.Series(np.where((d != 0) & np.isfinite(r), r, np.nan), index=days)
    return out


if __name__ == "__main__":
    DATA = sys.argv[1] if len(sys.argv) > 1 else "mt5_export"
    rows, daily = [], {}
    for name, spec in CASH.items():
        m1 = load_mt5_csv(f"{DATA}/{name}_M1.csv")
        m1 = m1[m1.index < HOLDOUT]
        res = run(name, m1, spec)
        for k, s in res.items():
            x = s.dropna()
            if len(x) < 100: continue
            full = s.fillna(0.0)
            tstat = x.mean() / x.std(ddof=1) * np.sqrt(len(x))
            rows.append(dict(strat=k, idx=name, n=len(x), net_bps=round(x.mean() * 1e4, 2), hit=round((x > 0).mean(), 3),
                             t=round(tstat, 2), sharpe=round(common.sharpe(full.to_numpy()), 2),
                             y1=round(x[x.index < "2024-07-01"].mean() * 1e4, 1), y2=round(x[(x.index >= "2024-07-01") & (x.index < "2025-10-01")].mean() * 1e4, 1),
                             y3=round(x[x.index >= "2025-10-01"].mean() * 1e4, 1)))
            daily[(k, name)] = full.to_numpy()
    T = pd.DataFrame(rows).sort_values(["strat", "idx"])
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    sr = np.array([v.mean() / v.std(ddof=1) for v in daily.values() if v.std() > 0])
    best = T.sort_values("t", ascending=False).iloc[0]
    d = daily[(best.strat, best.idx)]
    print(f"\nEssais : {len(sr)} (+144 antérieurs) | meilleur t={best.t} ({best.strat}, {best.idx}) | DSR={common.dsr(d, sr, len(sr) + 144):.4f}")
    T.to_csv("results_anomalies.csv", index=False)
