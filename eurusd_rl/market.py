"""Grille de décision causale + features de base (étape A).

Décisions : à chaque clôture M5 (heure serveur) entre entry_start et force_close (heure NY + 7 h).
À l'instant T, seules les bougies dont fin <= T sont lues. Le remplissage a lieu à l'ouverture de la
première bougie M1 qui commence à T (ou plus tard si latence).

Toutes les features sont sans échelle (ratios à l'ATR, rangs glissants passés) : aucun scaler n'est ajusté,
donc aucune fuite statistique entre périodes. Écrêtage à [-CLIP, CLIP].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import costs
from .timeframes import TFS, aggregate, last_closed

CLIP = 10.0
SEQ = {"M5": 48, "H1": 24}           # longueurs de séquence de l'étape A
BAR_FEATS = ["ret_atr", "range_atr", "body_atr", "uwick_atr", "lwick_atr", "tv_ratio"]
CTX_FEATS = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "atr_m5_d1", "atr_h1_rank", "dist_day_open",
             "dist_prev_high", "dist_prev_low", "pos_day_range", "min_to_close"]


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def atr(bars: pd.DataFrame, n: int) -> np.ndarray:
    h, l, c = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], axis=1)
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def bar_features(bars: pd.DataFrame, atr_n: int = 14, tv_n: int = 288) -> tuple[np.ndarray, np.ndarray]:
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(bars, atr_n)
    pc = np.r_[np.nan, c[:-1]]
    tv = bars["tick_volume"].to_numpy(float)
    tvm = pd.Series(tv).rolling(tv_n, min_periods=20).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.c_[(c - pc) / a, (h - l) / a, (c - o) / a, (h - np.maximum(o, c)) / a,
                  (np.minimum(o, c) - l) / a, np.log(tv / tvm)]
    f = np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), -CLIP, CLIP).astype(np.float32)
    return f, a


@dataclass
class Market:
    # M1 (simulation)
    o: np.ndarray; h: np.ndarray; l: np.ndarray; c: np.ndarray       # BID, float64
    spread: np.ndarray                                               # coût de spread, en prix
    m1_day: np.ndarray                                               # identifiant de jour serveur
    t_server_min: np.ndarray                                         # minutes epoch serveur
    # décisions
    dec_m1: np.ndarray            # indice M1 de remplissage (première M1 commençant à T)
    dec_next_m1: np.ndarray       # indice M1 de la décision suivante du même jour (ou j_force + 1)
    dec_force_m1: np.ndarray      # indice M1 de clôture forcée du jour
    dec_day: np.ndarray           # jour serveur
    dec_week: np.ndarray          # semaine serveur
    dec_can_enter: np.ndarray     # bool : dans la fenêtre d'entrée
    dec_is_last: np.ndarray       # bool : dernière décision du jour (clôture forcée appliquée)
    dec_stop_dist: np.ndarray     # distance de stop (prix) si entrée à cette décision
    dec_min_stop: np.ndarray      # distance de stop minimale admise (coûts, 3 pips)
    dec_t_utc: np.ndarray
    dec_t_server: np.ndarray
    # features
    seq_feats: dict = field(default_factory=dict)  # tf -> (n_bars, n_feat) float32
    seq_idx: dict = field(default_factory=dict)    # tf -> (n_dec,) indice de la dernière bougie close
    ctx: np.ndarray = None                         # (n_dec, n_ctx) float32
    meta: dict = field(default_factory=dict)

    def subset_mask(self, rng: list[str]) -> np.ndarray:
        t = pd.DatetimeIndex(self.dec_t_utc)
        return np.asarray((t >= pd.Timestamp(rng[0])) & (t < pd.Timestamp(rng[1]) + pd.Timedelta(days=1)))


def build_market(cfg: dict, m1: pd.DataFrame, spread_stress: float | None = None) -> Market:
    ex, sess, tr = cfg["execution"], cfg["session"], cfg["trade"]
    pt = cfg["data"]["point"]
    prof = costs.fit_profile(m1, ex["spread"]["profile_fit"], ex["spread"]["profile_quantile"])
    stress = ex["spread"]["stress_multiplier"] if spread_stress is None else spread_stress
    sp_pts = costs.spread_cost_pts(m1, prof, ex["spread"]["trust_observed_from"], ex["spread"]["floor_pts"], stress)
    m1 = m1.assign(spread_cost_pts=sp_pts)

    ts = pd.DatetimeIndex(m1["t_server"])
    t_min = ts.values.astype("datetime64[m]").astype(np.int64)
    day = (t_min // 1440).astype(np.int64)
    mod = t_min % 1440                                   # minute du jour serveur
    week = ((day + 3) // 7).astype(np.int64)             # 1970-01-01 = jeudi -> semaines lundi-dimanche

    off = 7 * 60                                         # serveur = NY + 7 h
    e0, e1, fc = _hhmm(sess["entry_start_ny"]) + off, _hhmm(sess["entry_end_ny"]) + off, _hhmm(sess["force_close_ny"]) + off
    assert fc < 1440, "la clôture forcée doit précéder la fin de journée serveur"

    bars = {tf: aggregate(m1, TFS[tf]) for tf in ("M5", "M15", "H1", "D1")}
    m5 = bars["M5"]
    T = m5["end"].to_numpy()                             # instants de décision candidats (fin des M5)
    T_min = T.astype("datetime64[m]").astype(np.int64)
    T_mod = T_min % 1440
    T_day = T_min // 1440
    keep = (T_mod >= e0) & (T_mod <= fc)
    T, T_min, T_day, T_mod = T[keep], T_min[keep], T_day[keep], T_mod[keep]
    j = np.searchsorted(t_min, T_min, side="left")       # première M1 qui commence à T ou après
    ok = (j < len(t_min))
    j = np.where(ok, j, len(t_min) - 1)
    ok &= (day[j] == T_day) & ((t_min[j] - T_min) <= 15)  # même jour et pas de trou > 15 min
    T, T_min, T_day, T_mod, j = T[ok], T_min[ok], T_day[ok], T_mod[ok], j[ok]

    # indice M1 de clôture forcée par jour : première M1 >= fc (sinon dernière M1 du jour)
    days = np.unique(T_day)
    jf_day = np.searchsorted(t_min, days * 1440 + fc, side="left")
    last_of_day = np.searchsorted(day, days, side="right") - 1
    jf_day = np.where((jf_day < len(t_min)) & (day[np.minimum(jf_day, len(day) - 1)] == days), jf_day, last_of_day)
    jf = jf_day[np.searchsorted(days, T_day)]

    # pas de décision après la clôture forcée ; une seule décision à j == jf (la première) par jour
    keep2 = (j < jf) | ((j == jf) & np.r_[True, ~((T_day[1:] == T_day[:-1]) & (j[:-1] >= jf[:-1]))])
    keep2 &= j <= jf
    T, T_min, T_day, T_mod, j, jf = T[keep2], T_min[keep2], T_day[keep2], T_mod[keep2], j[keep2], jf[keep2]
    is_last = np.r_[T_day[1:] != T_day[:-1], True]
    # borne exclusive de la simulation après la décision : décision suivante du jour, ou jf pour la dernière
    # (la clôture forcée se fait à l'OUVERTURE de jf, donc la dernière décision simule [j, jf) puis ferme à jf)
    jn = np.where(is_last, jf, np.r_[j[1:], 0])
    can_enter = (T_mod >= e0) & (T_mod < e1) & ~is_last

    # --- features
    seq_feats, seq_idx, atrs = {}, {}, {}
    for tf in ("M5", "H1"):
        f, a = bar_features(bars[tf])
        seq_feats[tf] = f
        seq_idx[tf] = last_closed(bars[tf]["end"].to_numpy(), T)
        atrs[tf] = a
    a15 = atr(bars["M15"], tr["stop_atr_period"])
    i15 = last_closed(bars["M15"]["end"].to_numpy(), T)
    aD = atr(bars["D1"], 14)
    iD = last_closed(bars["D1"]["end"].to_numpy(), T)
    D1 = bars["D1"]

    # coût aller-retour (prix) pour le stop minimal
    comm_px = 2 * ex["commission_per_lot_side"] / cfg["data"]["contract_size"]
    rt_cost = sp_pts[j] * pt + (ex["entry_slippage_pts"] + ex["exit_slippage_pts"]) * pt + comm_px
    stop = a15[np.maximum(i15, 0)] * tr["stop_atr_mult"]
    stop = np.where(i15 >= 0, stop, np.nan)
    min_stop = np.maximum(tr["min_stop_cost_mult"] * rt_cost, tr["min_stop_pts"] * pt)
    stop = np.maximum(stop, min_stop)

    # contexte
    c_prev = m1["close"].to_numpy()[np.maximum(j - 1, 0)]       # dernière clôture M1 connue (barre j-1)
    ny_mod = (T_mod - off) % 1440
    dow = pd.DatetimeIndex(T).dayofweek.to_numpy()
    ah1 = atrs["H1"][np.maximum(seq_idx["H1"], 0)]
    am5 = atrs["M5"][np.maximum(seq_idx["M5"], 0)]
    ad = aD[np.maximum(iD, 0)]
    h1_rank = pd.Series(atrs["H1"]).rolling(500, min_periods=50).rank(pct=True).to_numpy()[np.maximum(seq_idx["H1"], 0)]
    # ouverture, plus haut et plus bas du jour serveur jusqu'à T (M1 strictement avant T)
    first_of_day = np.searchsorted(day, T_day, side="left")
    day_open = m1["open"].to_numpy()[first_of_day]
    hi_cum = pd.Series(m1["high"].to_numpy()).groupby(day).cummax().to_numpy()
    lo_cum = pd.Series(m1["low"].to_numpy()).groupby(day).cummin().to_numpy()
    jm1 = np.maximum(j - 1, first_of_day)
    dh, dl = hi_cum[jm1], lo_cum[jm1]
    pdh = D1["high"].to_numpy()[np.maximum(iD, 0)]
    pdl = D1["low"].to_numpy()[np.maximum(iD, 0)]
    with np.errstate(invalid="ignore", divide="ignore"):
        ctx = np.c_[np.sin(2 * np.pi * ny_mod / 1440), np.cos(2 * np.pi * ny_mod / 1440),
                    np.sin(2 * np.pi * dow / 5), np.cos(2 * np.pi * dow / 5),
                    am5 / ad * 10, h1_rank * 2 - 1, (c_prev - day_open) / ah1,
                    (c_prev - pdh) / ad, (c_prev - pdl) / ad,
                    (c_prev - dl) / np.maximum(dh - dl, 1e-9) * 2 - 1,
                    ((fc - T_mod) / 600.0)]
    ctx = np.clip(np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0), -CLIP, CLIP).astype(np.float32)

    valid = (i15 >= 0) & (iD >= 1) & np.isfinite(stop) & (seq_idx["M5"] >= SEQ["M5"] + 300) & (seq_idx["H1"] >= SEQ["H1"] + 60)
    # une décision invalide ne peut pas entrer ; on la garde dans la grille pour la gestion des positions
    can_enter &= valid

    return Market(
        o=m1["open"].to_numpy(float), h=m1["high"].to_numpy(float), l=m1["low"].to_numpy(float),
        c=m1["close"].to_numpy(float), spread=sp_pts * pt, m1_day=day, t_server_min=t_min,
        dec_m1=j.astype(np.int64), dec_next_m1=jn.astype(np.int64), dec_force_m1=jf.astype(np.int64),
        dec_day=T_day.astype(np.int64), dec_week=((T_day + 3) // 7).astype(np.int64),
        dec_can_enter=can_enter, dec_is_last=is_last, dec_stop_dist=np.nan_to_num(stop, nan=1.0), dec_min_stop=min_stop,
        dec_t_utc=pd.DatetimeIndex(m1["t_utc"].to_numpy()[j]).values, dec_t_server=T,
        seq_feats=seq_feats, seq_idx=seq_idx, ctx=ctx,
        meta={"spread_profile": prof.tolist(), "spread_stress": stress, "n_dec": int(len(T))},
    )


def gather_seq(mk: Market, tf: str, dec_idx: np.ndarray) -> np.ndarray:
    """(B, L, F) : les L dernières bougies closes à la décision (la plus récente en dernier)."""
    L = SEQ[tf]
    last = mk.seq_idx[tf][dec_idx]
    ii = last[:, None] - np.arange(L - 1, -1, -1)[None, :]
    ii = np.maximum(ii, 0)
    return mk.seq_feats[tf][ii]
