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

LIB_VERSION = "B2-2026-10-29"   # B1 + échelles H1/H4 ; sorties M5 identiques à B1
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


def _opening_range(m5: pd.DataFrame, tz: str, open_min: int, or_min: int, bar_min: int = 5):
    """Plage d'ouverture (heure locale DST-correcte) : plus haut/bas des barres M5 entre open et open + or_min.
    Connue seulement à partir de la clôture de la dernière barre de la plage."""
    t_end = pd.DatetimeIndex(m5["t_utc_end"])
    lm, lday = _local_minutes(t_end, tz)
    lm_start = lm - bar_min
    inside = (lm_start >= open_min) & (lm <= open_min + or_min)
    df = pd.DataFrame({"day": lday, "h": np.where(inside, m5["high"], np.nan), "l": np.where(inside, m5["low"], np.nan),
                       "n": inside.astype(int)})
    g = df.groupby("day")
    hi = g["h"].max(); lo = g["l"].min(); cnt = g["n"].sum()
    or_hi = hi.reindex(lday).to_numpy(); or_lo = lo.reindex(lday).to_numpy(); n_in = cnt.reindex(lday).to_numpy()
    ready = (lm >= open_min + or_min) & (n_in >= max(or_min // bar_min - 1, 1))
    return np.where(ready, or_hi, np.nan), np.where(ready, or_lo, np.nan), lm


SCALES = {
    # E : barres d'entrée ; M : structure ; T : tendance ; H : unité haute ; L : niveaux de référence (veille / semaine)
    "M5": dict(E="5min", M="15min", T="1h", H="4h", L="1D", H_ema=20, rank_win=6000, orb_len=30, intraday=True),
    "H1": dict(E="1h", M="4h", T="1D", H="1W", L="1D", H_ema=20, rank_win=250, orb_len=60, intraday=True),
    "H4": dict(E="4h", M="1D", T="1W", H="1W", L="1W", H_ema=50, rank_win=52, orb_len=None, intraday=False),
}
INTRADAY_ONLY = ["vwap_reclaim", "session_level_reject", "orb"]


def names_for(scale: str) -> list[str]:
    return [n for n in DIRECTIONAL if SCALES[scale]["intraday"] or n not in INTRADAY_ONLY]


def compute(m1: pd.DataFrame, scale: str = "M5") -> dict:
    """Signaux de toutes les stratégies, indexés sur les barres d'entrée de l'échelle `scale`."""
    P = SCALES[scale]
    eb = aggregate(m1, P["E"])
    end = eb["end"].to_numpy()
    o, h, l, c = (eb[k].to_numpy() for k in ("open", "high", "low", "close"))
    tv = eb["tick_volume"].to_numpy(float)
    n = len(eb)
    HT = {}
    for role in ("M", "T", "H", "L"):
        if P[role] not in HT:
            HT[P[role]] = aggregate(m1, P[role])
    B = {role: HT[P[role]] for role in ("M", "T", "H", "L")}
    ix = {role: np.maximum(last_closed(b["end"].to_numpy(), end), 0) for role, b in B.items()}
    ok = {role: last_closed(b["end"].to_numpy(), end) >= 1 for role, b in B.items()}
    at = lambda role, arr: arr[ix[role]]
    atrE = atr(eb, 14)
    atrM = at("M", atr(B["M"], 14))
    atrT_raw = atr(B["T"], 14)
    atrT = at("T", atrT_raw)
    atrL = at("L", atr(B["L"], 14))
    vol_rank = at("T", pd.Series(atrT_raw).rolling(P["rank_win"], min_periods=P["rank_win"] // 12).rank(pct=True).to_numpy())

    tc = B["T"]["close"].to_numpy()
    e20T, e50T = ema(tc, 20), ema(tc, 50)
    lag = np.r_[e50T[:3], e50T[:-3]]
    trend_T = np.where((e20T > e50T) & (e50T > lag), 1, np.where((e20T < e50T) & (e50T < lag), -1, 0))
    trend = at("T", trend_T) * ok["T"]
    pdh = np.where(ok["L"], at("L", B["L"]["high"].to_numpy()), np.nan)
    pdl = np.where(ok["L"], at("L", B["L"]["low"].to_numpy()), np.nan)

    t_utc_end = pd.DatetimeIndex(eb["t_utc_end"])
    ny_min = ((pd.DatetimeIndex(end).hour * 60 + pd.DatetimeIndex(end).minute).to_numpy() - 420) % 1440
    dayE = pd.DatetimeIndex(eb["start"]).normalize().values.astype("datetime64[D]").astype(np.int64)
    new_day = np.r_[True, dayE[1:] != dayE[:-1]]
    gid = np.cumsum(new_day)

    S = {}

    def put(name, d, stop, strength=None):
        d = np.asarray(d, np.int64)
        stop = np.where(d != 0, stop, np.nan)
        bad = (d != 0) & ~((d == 1) & (stop < c) | (d == -1) & (stop > c))      # invalidation du mauvais côté
        d = np.where(bad, 0, d)
        stop = np.where(d != 0, stop, np.nan)
        st_ = np.zeros(n) if strength is None else np.clip(np.nan_to_num(strength), 0, 1)
        S[name] = {"dir": d, "stop_px": stop, "strength": np.where(d != 0, st_, 0.0)}

    pc = np.r_[c[0], c[:-1]]
    lo6 = pd.Series(l).rolling(6).min().to_numpy(); hi6 = pd.Series(h).rolling(6).max().to_numpy()
    lo3 = pd.Series(l).rolling(3).min().to_numpy(); hi3 = pd.Series(h).rolling(3).max().to_numpy()

    # 1. Pullback dans la tendance T : contact de l'EMA20 des barres d'entrée puis clôture de reprise
    e20E = ema(c, 20)
    pb_long = (trend > 0) & (lo3 <= e20E) & (c > e20E) & (c > o) & (c > pc)
    pb_short = (trend < 0) & (hi3 >= e20E) & (c < e20E) & (c < o) & (c < pc)
    put("trend_pullback", np.where(pb_long, 1, np.where(pb_short, -1, 0)),
        np.where(pb_long, lo6 - 0.2 * atrE, hi6 + 0.2 * atrE), np.abs(c - e20E) / atrE / 2)

    # 2. Cassure après compression : largeur de Bollinger sur M dans le quintile bas (300 barres M passées)
    mc, mh, ml = (B["M"][k].to_numpy() for k in ("close", "high", "low"))
    bw = pd.Series(mc).rolling(20).std().to_numpy() * 4 / mc
    bw_rank = pd.Series(bw).rolling(300, min_periods=100).rank(pct=True).to_numpy()
    rng_hi = at("M", pd.Series(mh).rolling(12).max().to_numpy())
    rng_lo = at("M", pd.Series(ml).rolling(12).min().to_numpy())
    squeeze = at("M", bw_rank) <= 0.2
    sq_long = squeeze & (pc <= rng_hi) & (c > rng_hi)
    sq_short = squeeze & (pc >= rng_lo) & (c < rng_lo)
    put("squeeze_breakout", np.where(sq_long, 1, np.where(sq_short, -1, 0)), (rng_hi + rng_lo) / 2, 1 - at("M", bw_rank))

    # 3. Retour à la moyenne en range : ADX(T) < 20, dernière barre M hors bande de Bollinger, barre d'entrée de rejet
    adxT = at("T", adx(B["T"]))
    maM = pd.Series(mc).rolling(20).mean().to_numpy(); sdM = pd.Series(mc).rolling(20).std().to_numpy()
    below = at("M", mc < maM - 2 * sdM); above = at("M", mc > maM + 2 * sdM)
    rm_long = (adxT < 20) & below & (c > o) & (c > pc)
    rm_short = (adxT < 20) & above & (c < o) & (c < pc)
    put("range_meanrev", np.where(rm_long, 1, np.where(rm_short, -1, 0)),
        np.where(rm_long, lo6 - 0.3 * atrM, hi6 + 0.3 * atrM), (20 - adxT) / 20)

    # 4. Rejet après balayage de liquidité au plus haut / plus bas de référence (veille ou semaine précédente)
    sw_long = (l < pdl) & (c > pdl) & (c > o)
    sw_short = (h > pdh) & (c < pdh) & (c < o)
    put("liquidity_sweep", np.where(sw_long, 1, np.where(sw_short, -1, 0)),
        np.where(sw_long, l - 0.2 * atrE, h + 0.2 * atrE), np.abs(c - np.where(sw_long, pdl, pdh)) / atrM)

    # 5. Momentum multi-unités : M, T, H au-dessus/en dessous de leur EMA, L dans le même sens, nouveau plus haut 12 barres
    def above_ema(role, n_=20):
        cc = B[role]["close"].to_numpy()
        return at(role, np.sign(cc - ema(cc, n_)))
    lc = B["L"]["close"].to_numpy()
    lret = at("L", np.sign(lc - np.r_[lc[0], lc[:-1]]))
    align = above_ema("M") + above_ema("T") + above_ema("H", P["H_ema"]) + lret
    hi12 = pd.Series(h).rolling(12).max().shift(1).to_numpy(); lo12 = pd.Series(l).rolling(12).min().shift(1).to_numpy()
    mm_long = (align == 4) & (c > hi12); mm_short = (align == -4) & (c < lo12)
    put("mtf_momentum", np.where(mm_long, 1, np.where(mm_short, -1, 0)), np.where(mm_long, c - atrM, c + atrM),
        np.full(n, 1.0))

    # 6. Filtre de volatilité et de session (pas de direction) : 02:00-12:00 NY et volatilité T entre 30 et 90 %
    S["vol_session_filter"] = {"active": (ny_min >= 120) & (ny_min <= 720) & (vol_rank >= 0.3) & (vol_rank <= 0.9)}

    if P["intraday"]:
        # 7. Reclaim / rejet du VWAP journalier (journée serveur), dans le sens de la tendance T
        tp = (h + l + c) / 3
        cum_pv = pd.Series(tp * tv).groupby(gid).cumsum().to_numpy(); cum_v = pd.Series(tv).groupby(gid).cumsum().to_numpy()
        vwap = cum_pv / np.maximum(cum_v, 1e-9)
        pvwap = np.r_[np.nan, vwap[:-1]]
        bars_min = pd.Series(np.ones(n)).groupby(gid).cumsum().to_numpy() * (pd.Timedelta(P["E"]).total_seconds() / 60)
        vw_long = (trend > 0) & (pc < pvwap) & (c > vwap) & (bars_min >= 60)
        vw_short = (trend < 0) & (pc > pvwap) & (c < vwap) & (bars_min >= 60)
        put("vwap_reclaim", np.where(vw_long, 1, np.where(vw_short, -1, 0)),
            np.where(vw_long, lo3 - 0.2 * atrE, hi3 + 0.2 * atrE), np.abs(c - vwap) / atrE)

        # 8. Rejet d'un niveau de session : plus haut / plus bas de la session asiatique (00:00-07:00 UTC du jour)
        um, uday = _local_minutes(t_utc_end, "UTC")
        e_min = int(pd.Timedelta(P["E"]).total_seconds() // 60)
        in_asia = (um - e_min >= 0) & (um <= 420)
        adf = pd.DataFrame({"d": uday, "h": np.where(in_asia, h, np.nan), "l": np.where(in_asia, l, np.nan)})
        ah = adf.groupby("d")["h"].max().reindex(uday).to_numpy(); al = adf.groupby("d")["l"].min().reindex(uday).to_numpy()
        after = um > 420
        ah, al = np.where(after, ah, np.nan), np.where(after, al, np.nan)
        sr_short = (h > ah) & (c < ah) & (c < o) & ((h - np.maximum(o, c)) > np.abs(c - o))
        sr_long = (l < al) & (c > al) & (c > o) & ((np.minimum(o, c) - l) > np.abs(c - o))
        put("session_level_reject", np.where(sr_long, 1, np.where(sr_short, -1, 0)),
            np.where(sr_long, l - 0.2 * atrE, h + 0.2 * atrE), np.full(n, 1.0))

        # 9. Opening Range Breakout : Londres 08:00 et New York 09:30 (heure locale), plage orb_len, 2 h pour casser
        d_orb = np.zeros(n, np.int64); st_orb = np.full(n, np.nan)
        L_ = P["orb_len"]
        for tz, om in (("Europe/London", 8 * 60), ("America/New_York", 9 * 60 + 30)):
            oh, ol, lm = _opening_range(eb, tz, om, L_, e_min)
            win = (lm > om + L_) & (lm <= om + L_ + 120)
            up = win & (pc <= oh) & (c > oh)
            dn = win & (pc >= ol) & (c < ol)
            free = d_orb == 0
            d_orb = np.where(free & up, 1, np.where(free & dn, -1, d_orb))
            st_orb = np.where(free & (up | dn), (oh + ol) / 2, st_orb)
        first = pd.DataFrame({"g": gid, "d": d_orb}).groupby(["g", "d"]).cumcount().to_numpy() == 0
        d_orb = np.where(first, d_orb, 0)
        put("orb", d_orb, st_orb, np.full(n, 1.0))

    # 10. Cassure-retest du plus haut / plus bas de référence
    d_br, st_br = _break_retest(c, h, l, pdh, pdl, atrM, 24)
    put("break_retest", d_br, st_br, np.full(n, 1.0))

    # 11. Réaction sur FVG de l'unité M confirmé, dans le sens de la tendance T (gap valable 16 barres M)
    kM = last_closed(B["M"]["end"].to_numpy(), end)
    d_fvg, st_fvg = _fvg_track(mh, ml, mc, kM.astype(np.int64), l, h, c, trend.astype(np.int64), 16)
    put("fvg_reaction", d_fvg, st_fvg - np.sign(d_fvg) * 0.1 * np.nan_to_num(atrE), np.full(n, 1.0))

    # 12. Confluence Fibonacci + structure + tendance : swings T confirmés (k = 3), structure HH/HL (LH/LL),
    #     prix dans la zone 50-78,6 % de la dernière jambe, clôture de reprise ; stop au-delà de 78,6 %
    sh1, sh2, sl1, sl2 = _swings(B["T"]["high"].to_numpy(), B["T"]["low"].to_numpy(), 3)
    sh1, sh2, sl1, sl2 = (at("T", x) for x in (sh1, sh2, sl1, sl2))
    up_struct = (sh1 > sh2) & (sl1 > sl2) & (trend > 0)
    dn_struct = (sh1 < sh2) & (sl1 < sl2) & (trend < 0)
    leg = sh1 - sl1
    f50u, f786u = sh1 - 0.5 * leg, sh1 - 0.786 * leg
    f50d, f786d = sl1 + 0.5 * leg, sl1 + 0.786 * leg
    fb_long = up_struct & (leg > atrT) & (l <= f50u) & (l >= f786u) & (c > o) & (c > pc) & (c < sh1)
    fb_short = dn_struct & (leg > atrT) & (h >= f50d) & (h <= f786d) & (c < o) & (c < pc) & (c > sl1)
    put("fib_confluence", np.where(fb_long, 1, np.where(fb_short, -1, 0)), np.where(fb_long, f786u, f786d),
        leg / atrL)

    for name in [k for k in S if k != "vol_session_filter"]:
        d, st = S[name]["dir"], S[name]["stop_px"]
        dist = np.abs(c - st)
        room = np.where(d > 0, pdh - c, np.where(d < 0, c - pdl, np.nan))
        S[name]["rr_room"] = np.where((d != 0) & (room > 0), room / np.maximum(dist, 1e-9), 0.0)
        S[name]["vol_rank"] = np.where(d != 0, vol_rank, np.nan)
    return {"bar_end": end, "m5_end": end, "close": c, "signals": S, "vol_rank": vol_rank,
            "version": LIB_VERSION, "scale": scale}


def to_decisions(lib: dict, mk, max_stale_min: int = 0) -> dict:
    """Aligne les signaux sur la grille de décision. Le signal de la barre d'entrée k (fin = end_k) n'est émis qu'à la
    PREMIÈRE décision T >= end_k, et seulement si T - end_k <= max_stale_min ; ensuite il est consommé."""
    T = mk.dec_t_server
    k = last_closed(lib["bar_end"], T)
    first_seen = np.r_[True, k[1:] != k[:-1]] & (k >= 0)
    stale = (T - lib["bar_end"][np.maximum(k, 0)]).astype("timedelta64[m]").astype(np.int64)
    fire = first_seen & (stale <= max_stale_min)
    kk = np.maximum(k, 0)
    out = {}
    for name, s in lib["signals"].items():
        out[name] = {}
        for key, v in s.items():
            x = v[kk]
            if key == "dir":
                x = np.where(fire, x, 0)
            elif key == "active":
                x = np.where(k >= 0, x, False)
            out[name][key] = x
    out["_close"] = mk.c[np.maximum(mk.dec_m1 - 1, 0)]       # dernier prix BID connu à la décision
    return out
