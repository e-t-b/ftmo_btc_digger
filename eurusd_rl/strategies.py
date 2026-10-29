"""Bibliothèque de stratégies explicites causales (étape B). Version : voir LIB_VERSION.

Chaque stratégie est évaluée à la clôture de chaque barre M5 i (instant T = fin de la barre, heure serveur) et
n'utilise que des barres dont la fin est <= T. Sortie par stratégie (tableaux de longueur n_M5) :
  dir      : -1 / 0 / +1 (0 = inactive)
  stop_px  : niveau d'invalidation (prix BID) ; NaN si inactive
  strength : force normalisée dans [0, 1]
Contexte commun : vol_rank (rang de l'ATR H1 sur ~1 an de barres passées), rr_room (place jusqu'au prochain niveau
opposé, en multiples de la distance au stop).
Paramètres fixés a priori (pré-enregistrés) : aucune optimisation sur les données.

Liste : trend_pullback, squeeze_breakout, range_meanrev, liquidity_sweep, mtf_momentum, vol_session_filter (filtre,
sans direction), vwap_reclaim, session_level_reject, orb, break_retest, fvg_reaction, fib_confluence.
"""
from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd

from .market import atr
from .timeframes import aggregate, last_closed

LIB_VERSION = "B1-2026-10-29"
NAMES = ["trend_pullback", "squeeze_breakout", "range_meanrev", "liquidity_sweep", "mtf_momentum",
         "vol_session_filter", "vwap_reclaim", "session_level_reject", "orb", "break_retest", "fvg_reaction",
         "fib_confluence"]
DIRECTIONAL = [n for n in NAMES if n != "vol_session_filter"]


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def adx(b: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = b["high"].to_numpy(), b["low"].to_numpy(), b["close"].to_numpy()
    up, dn = np.r_[0, h[1:] - h[:-1]], np.r_[0, l[:-1] - l[1:]]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    w = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    atr_ = w(tr)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi, mdi = 100 * w(pdm) / atr_, 100 * w(mdm) / atr_
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return w(np.nan_to_num(dx))


@nb.njit(cache=True)
def _swings(h, l, k):
    """Swings fractals d'ordre k. Renvoie, pour chaque barre t, les deux derniers swing highs/lows CONFIRMÉS à la
    clôture de t (un swing en j est confirmé à la clôture de j + k)."""
    n = len(h)
    sh1 = np.full(n, np.nan); sh2 = np.full(n, np.nan); sl1 = np.full(n, np.nan); sl2 = np.full(n, np.nan)
    a1 = np.nan; a2 = np.nan; b1 = np.nan; b2 = np.nan
    for t in range(n):
        j = t - k
        if j >= k:
            is_h = True
            is_l = True
            for q in range(j - k, j + k + 1):
                if q == j:
                    continue
                if (q < j and h[q] >= h[j]) or (q > j and h[q] > h[j]):
                    is_h = False
                if (q < j and l[q] <= l[j]) or (q > j and l[q] < l[j]):
                    is_l = False
            if is_h:
                a2 = a1; a1 = h[j]
            if is_l:
                b2 = b1; b1 = l[j]
        sh1[t] = a1; sh2[t] = a2; sl1[t] = b1; sl2[t] = b2
    return sh1, sh2, sl1, sl2


@nb.njit(cache=True)
def _fvg_track(h, l, c, end_idx_for_m5, m5_low, m5_high, m5_close, trend, max_age):
    """FVG haussier M15 : low[k] > high[k-2], confirmé à la clôture de k. Réaction : une barre M5 entre dans le gap
    (low <= haut du gap) et clôture au-dessus du haut du gap, tendance H1 haussière. Symétrique en baissier.
    Un gap est invalidé si une clôture M15 traverse son bas (haussier) / son haut (baissier).
    end_idx_for_m5[i] = indice de la dernière M15 close à la fin de la barre M5 i."""
    n5 = len(m5_close)
    out_dir = np.zeros(n5, np.int64)
    out_stop = np.full(n5, np.nan)
    # état : dernier FVG haussier et baissier valides (indice de confirmation, bas, haut)
    bu_k = -1; bu_lo = 0.0; bu_hi = 0.0; be_k = -1; be_lo = 0.0; be_hi = 0.0
    last_k = -1
    for i in range(n5):
        k = end_idx_for_m5[i]
        while last_k < k:
            last_k += 1
            q = last_k
            if q >= 2:
                if l[q] > h[q - 2]:
                    bu_k = q; bu_lo = h[q - 2]; bu_hi = l[q]
                if h[q] < l[q - 2]:
                    be_k = q; be_lo = h[q]; be_hi = l[q - 2]
            if bu_k >= 0 and c[q] < bu_lo:
                bu_k = -1
            if be_k >= 0 and c[q] > be_hi:
                be_k = -1
        if bu_k >= 0 and k - bu_k > max_age:
            bu_k = -1
        if be_k >= 0 and k - be_k > max_age:
            be_k = -1
        if bu_k >= 0 and trend[i] > 0 and m5_low[i] <= bu_hi and m5_close[i] > bu_hi:
            out_dir[i] = 1; out_stop[i] = bu_lo
        elif be_k >= 0 and trend[i] < 0 and m5_high[i] >= be_lo and m5_close[i] < be_lo:
            out_dir[i] = -1; out_stop[i] = be_hi
    return out_dir, out_stop


@nb.njit(cache=True)
def _break_retest(c5, h5, l5, lvl_hi, lvl_lo, atr15, window):
    """Cassure puis retest : une clôture M5 au-dessus du niveau (plus haut de la veille), puis dans les `window`
    barres suivantes un retour à moins de 0,25 ATR M15 du niveau et une clôture M5 au-dessus -> achat."""
    n = len(c5)
    d = np.zeros(n, np.int64); st = np.full(n, np.nan)
    up_at = -10**9; up_lvl = 0.0; dn_at = -10**9; dn_lvl = 0.0
    for i in range(1, n):
        a = atr15[i]
        if not (a > 0):
            continue
        if lvl_hi[i] == lvl_hi[i] and c5[i - 1] <= lvl_hi[i] < c5[i]:
            up_at = i; up_lvl = lvl_hi[i]
            continue
        if lvl_lo[i] == lvl_lo[i] and c5[i - 1] >= lvl_lo[i] > c5[i]:
            dn_at = i; dn_lvl = lvl_lo[i]
            continue
        if 2 <= i - up_at <= window and l5[i] <= up_lvl + 0.25 * a and c5[i] > up_lvl and c5[i] > c5[i - 1]:
            d[i] = 1; st[i] = up_lvl - 0.5 * a; up_at = -10**9
        elif 2 <= i - dn_at <= window and h5[i] >= dn_lvl - 0.25 * a and c5[i] < dn_lvl and c5[i] < c5[i - 1]:
            d[i] = -1; st[i] = dn_lvl + 0.5 * a; dn_at = -10**9
    return d, st


def _local_minutes(t_utc_end: pd.DatetimeIndex, tz: str) -> np.ndarray:
    loc = t_utc_end.tz_localize("UTC").tz_convert(tz)
    return (loc.hour * 60 + loc.minute).to_numpy(), loc.normalize().tz_localize(None).values.astype("datetime64[D]").astype(np.int64)


def _opening_range(m5: pd.DataFrame, tz: str, open_min: int, or_min: int):
    """Plage d'ouverture (heure locale DST-correcte) : plus haut/bas des barres M5 entre open et open + or_min.
    Connue seulement à partir de la clôture de la dernière barre de la plage."""
    t_end = pd.DatetimeIndex(m5["t_utc_end"])
    lm, lday = _local_minutes(t_end, tz)
    lm_start = lm - 5
    inside = (lm_start >= open_min) & (lm <= open_min + or_min)
    df = pd.DataFrame({"day": lday, "h": np.where(inside, m5["high"], np.nan), "l": np.where(inside, m5["low"], np.nan),
                       "n": inside.astype(int)})
    g = df.groupby("day")
    hi = g["h"].max(); lo = g["l"].min(); cnt = g["n"].sum()
    or_hi = hi.reindex(lday).to_numpy(); or_lo = lo.reindex(lday).to_numpy(); n_in = cnt.reindex(lday).to_numpy()
    ready = (lm >= open_min + or_min) & (n_in >= or_min // 5 - 1)
    return np.where(ready, or_hi, np.nan), np.where(ready, or_lo, np.nan), lm


def compute(m1: pd.DataFrame) -> dict:
    """Signaux de toutes les stratégies, indexés sur les barres M5. Renvoie aussi `m5_end` pour l'alignement."""
    m5 = aggregate(m1, "5min")
    end = m5["end"].to_numpy()
    o, h, l, c = (m5[k].to_numpy() for k in ("open", "high", "low", "close"))
    tv = m5["tick_volume"].to_numpy(float)
    n = len(m5)
    HT = {tf: aggregate(m1, r) for tf, r in (("M15", "15min"), ("H1", "1h"), ("H4", "4h"), ("D1", "1D"))}
    ix = {tf: np.maximum(last_closed(b["end"].to_numpy(), end), 0) for tf, b in HT.items()}
    ok = {tf: last_closed(b["end"].to_numpy(), end) >= 1 for tf, b in HT.items()}
    at = lambda tf, arr: arr[ix[tf]]
    atr5 = atr(m5, 14)
    atr15 = at("M15", atr(HT["M15"], 14))
    atrh1_raw = atr(HT["H1"], 14)
    atrh1 = at("H1", atrh1_raw)
    atrd = at("D1", atr(HT["D1"], 14))
    vol_rank = at("H1", pd.Series(atrh1_raw).rolling(6000, min_periods=500).rank(pct=True).to_numpy())

    h1c = HT["H1"]["close"].to_numpy()
    e20h1, e50h1 = ema(h1c, 20), ema(h1c, 50)
    trend_h1 = np.where((e20h1 > e50h1) & (e50h1 > np.r_[e50h1[:3], e50h1[:-3]]), 1,
                        np.where((e20h1 < e50h1) & (e50h1 < np.r_[e50h1[:3], e50h1[:-3]]), -1, 0))
    trend = at("H1", trend_h1) * ok["H1"]
    pdh, pdl = at("D1", HT["D1"]["high"].to_numpy()), at("D1", HT["D1"]["low"].to_numpy())
    pdh = np.where(ok["D1"], pdh, np.nan); pdl = np.where(ok["D1"], pdl, np.nan)

    t_utc_end = pd.DatetimeIndex(m5["t_utc_end"])
    ny_min = ((pd.DatetimeIndex(end).hour * 60 + pd.DatetimeIndex(end).minute).to_numpy() - 420) % 1440   # fin de barre, heure NY
    day5 = pd.DatetimeIndex(m5["start"]).normalize().values.astype("datetime64[D]").astype(np.int64)
    new_day = np.r_[True, day5[1:] != day5[:-1]]

    S = {}

    def put(name, d, stop, strength=None):
        d = np.asarray(d, np.int64)
        stop = np.where(d != 0, stop, np.nan)
        bad = (d != 0) & ~((d == 1) & (stop < c) | (d == -1) & (stop > c))      # invalidation du mauvais côté
        d = np.where(bad, 0, d)
        stop = np.where(d != 0, stop, np.nan)
        s = np.zeros(n) if strength is None else np.clip(np.nan_to_num(strength), 0, 1)
        S[name] = {"dir": d, "stop_px": stop, "strength": np.where(d != 0, s, 0.0)}

    pc = np.r_[c[0], c[:-1]]
    lo6 = pd.Series(l).rolling(6).min().to_numpy(); hi6 = pd.Series(h).rolling(6).max().to_numpy()
    lo3 = pd.Series(l).rolling(3).min().to_numpy(); hi3 = pd.Series(h).rolling(3).max().to_numpy()

    # 1. Pullback dans une tendance H1 : contact de l'EMA20 M5 puis clôture de reprise
    e20m5 = ema(c, 20)
    pb_long = (trend > 0) & (lo3 <= e20m5) & (c > e20m5) & (c > o) & (c > pc)
    pb_short = (trend < 0) & (hi3 >= e20m5) & (c < e20m5) & (c < o) & (c < pc)
    put("trend_pullback", np.where(pb_long, 1, np.where(pb_short, -1, 0)),
        np.where(pb_long, lo6 - 0.2 * atr5, hi6 + 0.2 * atr5), np.abs(c - e20m5) / atr5 / 2)

    # 2. Cassure après compression : largeur de Bollinger M15 dans le quintile bas (300 barres passées)
    m15c = HT["M15"]["close"].to_numpy()
    bw = pd.Series(m15c).rolling(20).std().to_numpy() * 4 / m15c
    bw_rank = pd.Series(bw).rolling(300, min_periods=100).rank(pct=True).to_numpy()
    m15h, m15l = HT["M15"]["high"].to_numpy(), HT["M15"]["low"].to_numpy()
    rng_hi = at("M15", pd.Series(m15h).rolling(12).max().to_numpy())
    rng_lo = at("M15", pd.Series(m15l).rolling(12).min().to_numpy())
    squeeze = at("M15", bw_rank) <= 0.2
    sq_long = squeeze & (pc <= rng_hi) & (c > rng_hi)
    sq_short = squeeze & (pc >= rng_lo) & (c < rng_lo)
    mid = (rng_hi + rng_lo) / 2
    put("squeeze_breakout", np.where(sq_long, 1, np.where(sq_short, -1, 0)), mid, 1 - at("M15", bw_rank))

    # 3. Retour à la moyenne en range : ADX H1 < 20, dernière M15 sous/sur la bande de Bollinger, barre M5 de rejet
    adx_h1 = at("H1", adx(HT["H1"]))
    ma15 = pd.Series(m15c).rolling(20).mean().to_numpy(); sd15 = pd.Series(m15c).rolling(20).std().to_numpy()
    below = at("M15", m15c < ma15 - 2 * sd15); above = at("M15", m15c > ma15 + 2 * sd15)
    rm_long = (adx_h1 < 20) & below & (c > o) & (c > pc)
    rm_short = (adx_h1 < 20) & above & (c < o) & (c < pc)
    put("range_meanrev", np.where(rm_long, 1, np.where(rm_short, -1, 0)),
        np.where(rm_long, lo6 - 0.3 * atr15, hi6 + 0.3 * atr15), (20 - adx_h1) / 20)

    # 4. Rejet après balayage de liquidité au plus haut / plus bas de la veille
    sw_long = (l < pdl) & (c > pdl) & (c > o)
    sw_short = (h > pdh) & (c < pdh) & (c < o)
    put("liquidity_sweep", np.where(sw_long, 1, np.where(sw_short, -1, 0)),
        np.where(sw_long, l - 0.2 * atr5, h + 0.2 * atr5), np.abs(c - np.where(sw_long, pdl, pdh)) / atr15)

    # 5. Momentum multi-unités : M15, H1, H4 au-dessus/en dessous de leur EMA20, D1 dans le même sens, nouveau plus haut 1 h
    def above_ema(tf, n_=20):
        cc = HT[tf]["close"].to_numpy()
        return at(tf, np.sign(cc - ema(cc, n_)))
    d1c = HT["D1"]["close"].to_numpy()
    d1ret = at("D1", np.sign(d1c - np.r_[d1c[0], d1c[:-1]]))
    align = above_ema("M15") + above_ema("H1") + above_ema("H4") + d1ret
    hi12 = pd.Series(h).rolling(12).max().shift(1).to_numpy(); lo12 = pd.Series(l).rolling(12).min().shift(1).to_numpy()
    mm_long = (align == 4) & (c > hi12); mm_short = (align == -4) & (c < lo12)
    put("mtf_momentum", np.where(mm_long, 1, np.where(mm_short, -1, 0)), np.where(mm_long, c - atr15, c + atr15),
        np.full(n, 1.0))

    # 6. Filtre de volatilité et de session (pas de direction) : 02:00-12:00 NY et volatilité H1 entre 30 et 90 %
    S["vol_session_filter"] = {"active": (ny_min >= 120) & (ny_min <= 720) & (vol_rank >= 0.3) & (vol_rank <= 0.9)}

    # 7. Reclaim / rejet du VWAP journalier (journée serveur), dans le sens de la tendance H1
    tp = (h + l + c) / 3
    gid = np.cumsum(new_day)
    cum_pv = pd.Series(tp * tv).groupby(gid).cumsum().to_numpy(); cum_v = pd.Series(tv).groupby(gid).cumsum().to_numpy()
    vwap = cum_pv / np.maximum(cum_v, 1e-9)
    pvwap = np.r_[np.nan, vwap[:-1]]
    bars_in_day = pd.Series(np.ones(n)).groupby(gid).cumsum().to_numpy()
    vw_long = (trend > 0) & (pc < pvwap) & (c > vwap) & (bars_in_day >= 12)
    vw_short = (trend < 0) & (pc > pvwap) & (c < vwap) & (bars_in_day >= 12)
    put("vwap_reclaim", np.where(vw_long, 1, np.where(vw_short, -1, 0)),
        np.where(vw_long, lo3 - 0.2 * atr5, hi3 + 0.2 * atr5), np.abs(c - vwap) / atr5)

    # 8. Rejet d'un niveau de session : plus haut / plus bas de la session asiatique (00:00-07:00 UTC du jour)
    um, uday = _local_minutes(t_utc_end, "UTC")
    in_asia = (um - 5 >= 0) & (um <= 420)
    adf = pd.DataFrame({"d": uday, "h": np.where(in_asia, h, np.nan), "l": np.where(in_asia, l, np.nan)})
    ah = adf.groupby("d")["h"].max().reindex(uday).to_numpy(); al = adf.groupby("d")["l"].min().reindex(uday).to_numpy()
    after = um > 420
    ah, al = np.where(after, ah, np.nan), np.where(after, al, np.nan)
    sr_short = (h > ah) & (c < ah) & (c < o) & ((h - np.maximum(o, c)) > np.abs(c - o))
    sr_long = (l < al) & (c > al) & (c > o) & ((np.minimum(o, c) - l) > np.abs(c - o))
    put("session_level_reject", np.where(sr_long, 1, np.where(sr_short, -1, 0)),
        np.where(sr_long, l - 0.2 * atr5, h + 0.2 * atr5), np.full(n, 1.0))

    # 9. Opening Range Breakout : Londres 08:00 et New York 09:30 (heure locale), plage de 30 min, 2 h pour casser
    d_orb = np.zeros(n, np.int64); st_orb = np.full(n, np.nan)
    for tz, om in (("Europe/London", 8 * 60), ("America/New_York", 9 * 60 + 30)):
        oh, ol, lm = _opening_range(m5, tz, om, 30)
        win = (lm > om + 30) & (lm <= om + 150)
        up = win & (pc <= oh) & (c > oh)
        dn = win & (pc >= ol) & (c < ol)
        free = d_orb == 0
        d_orb = np.where(free & up, 1, np.where(free & dn, -1, d_orb))
        st_orb = np.where(free & (up | dn), (oh + ol) / 2, st_orb)
    # une seule cassure par plage et par sens : on ne garde que le premier signal du jour dans chaque sens
    first = pd.DataFrame({"g": gid, "d": d_orb}).groupby(["g", "d"]).cumcount().to_numpy() == 0
    d_orb = np.where(first, d_orb, 0)
    put("orb", d_orb, st_orb, np.full(n, 1.0))

    # 10. Cassure-retest du plus haut / plus bas de la veille
    d_br, st_br = _break_retest(c, h, l, pdh, pdl, atr15, 24)
    put("break_retest", d_br, st_br, np.full(n, 1.0))

    # 11. Réaction sur FVG M15 confirmé, dans le sens de la tendance H1 (gap valable 16 barres M15)
    k15 = last_closed(HT["M15"]["end"].to_numpy(), end)
    d_fvg, st_fvg = _fvg_track(m15h, m15l, m15c, k15.astype(np.int64), l, h, c, trend.astype(np.int64), 16)
    put("fvg_reaction", d_fvg, st_fvg - np.sign(d_fvg) * 0.1 * np.nan_to_num(atr5), np.full(n, 1.0))

    # 12. Confluence Fibonacci + structure + tendance : swings H1 confirmés (k = 3), structure HH/HL (LH/LL),
    #     prix dans la zone 50-61,8 % de la dernière jambe, clôture M5 de reprise ; stop au-delà de 78,6 %
    sh1, sh2, sl1, sl2 = _swings(HT["H1"]["high"].to_numpy(), HT["H1"]["low"].to_numpy(), 3)
    sh1, sh2, sl1, sl2 = (at("H1", x) for x in (sh1, sh2, sl1, sl2))
    up_struct = (sh1 > sh2) & (sl1 > sl2) & (trend > 0)
    dn_struct = (sh1 < sh2) & (sl1 < sl2) & (trend < 0)
    leg = sh1 - sl1
    f50u, f618u, f786u = sh1 - 0.5 * leg, sh1 - 0.618 * leg, sh1 - 0.786 * leg
    f50d, f618d, f786d = sl1 + 0.5 * leg, sl1 + 0.618 * leg, sl1 + 0.786 * leg
    fb_long = up_struct & (leg > atrh1) & (l <= f50u) & (l >= f786u) & (c > o) & (c > pc) & (c < sh1)
    fb_short = dn_struct & (leg > atrh1) & (h >= f50d) & (h <= f786d) & (c < o) & (c < pc) & (c > sl1)
    put("fib_confluence", np.where(fb_long, 1, np.where(fb_short, -1, 0)), np.where(fb_long, f786u, f786d),
        leg / atrd)

    # contexte commun : place jusqu'au prochain niveau opposé (veille), en multiples de la distance au stop
    rr = {}
    for name in DIRECTIONAL:
        d, st = S[name]["dir"], S[name]["stop_px"]
        dist = np.abs(c - st)
        room = np.where(d > 0, pdh - c, np.where(d < 0, c - pdl, np.nan))
        S[name]["rr_room"] = np.where((d != 0) & (room > 0), room / np.maximum(dist, 1e-9), 0.0)
        S[name]["vol_rank"] = np.where(d != 0, vol_rank, np.nan)
    return {"m5_end": end, "close": c, "signals": S, "vol_rank": vol_rank, "version": LIB_VERSION}


def to_decisions(lib: dict, mk) -> dict:
    """Aligne les signaux M5 sur la grille de décision : la décision d lit la barre M5 qui se termine à T_d."""
    i = mk.seq_idx["M5"]
    assert np.array_equal(lib["m5_end"][i], mk.dec_t_server), "grille M5 incohérente"
    out = {}
    for name, s in lib["signals"].items():
        out[name] = {k: v[i] for k, v in s.items()}
    out["_close"] = lib["close"][i]
    return out
