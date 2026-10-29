"""Table de features causales pour le prédicteur (étage 1).

Chaque ligne = une décision d (instant T, clôture M5, heure serveur). Toutes les colonnes n'utilisent que des
données dont l'horodatage de fin est <= T. NaN autorisés (LightGBM les gère) quand une donnée n'existe pas.
Blocs :
  ctx_*   : contexte du marché (11 colonnes de market.py)
  m5_*    : 12 dernières bougies M5 closes x 6 features (ratios à l'ATR)
  h1_*    : 6 dernières bougies H1 closes x 6 features
  ret_*   : rendements EURUSD 15 min, 1 h, 4 h, 24 h, 120 h, en ATR H1
  osc_*   : RSI 14 sur M15/H1/D1 (barres closes), position dans le range 20 barres H1/D1
  sig_*   : état des stratégies de la bibliothèque à trois échelles (M5, H1, H4) : direction de la dernière barre
            d'entrée close (+ filtre volatilité/session, rang de volatilité)
  x_*     : autres marchés (GBPUSD, US500, GER40) : log-rendements 5, 60, 240 min jusqu'à T (x 1e4)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import strategies as S
from .leadlag import INSTR, load_close
from .market import CTX_FEATS, Market, atr
from .timeframes import aggregate, last_closed

PIP = 1e-4


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(close, prepend=close[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 - 100 / (1 + up / dn)


def price_at(t_min: np.ndarray, c: np.ndarray, T_min: np.ndarray, max_gap: int = 15) -> np.ndarray:
    """Clôture de la dernière minute terminée à T (minute < T), si elle date de moins de max_gap minutes."""
    i = np.searchsorted(t_min, T_min, side="left") - 1
    ok = (i >= 0) & (T_min - t_min[np.maximum(i, 0)] <= max_gap)
    return np.where(ok, c[np.maximum(i, 0)], np.nan)


def build(mk: Market, m1: pd.DataFrame, dec_idx: np.ndarray, with_cross: bool = True) -> pd.DataFrame:
    T = mk.dec_t_server[dec_idx]
    T_min = T.astype("datetime64[m]").astype(np.int64)
    cols = {}
    for i, n in enumerate(CTX_FEATS):
        cols[f"ctx_{n}"] = mk.ctx[dec_idx, i]
    for tf, L in (("M5", 12), ("H1", 6)):
        last = mk.seq_idx[tf][dec_idx]
        for lag in range(L):
            f = mk.seq_feats[tf][np.maximum(last - lag, 0)]
            for k in range(f.shape[1]):
                cols[f"{tf.lower()}_l{lag}_f{k}"] = f[:, k]
    # rendements multi-horizons (EURUSD), normalisés par l'ATR H1
    t_min = pd.DatetimeIndex(m1["t_server"]).values.astype("datetime64[m]").astype(np.int64)
    c = m1["close"].to_numpy()
    p0 = price_at(t_min, c, T_min)
    h1 = aggregate(m1, "1h")
    a_h1 = atr(h1, 14)[np.maximum(last_closed(h1["end"].to_numpy(), T), 0)]
    for mins in (15, 60, 240, 1440, 7200):
        pm = price_at(t_min, c, T_min - mins, max_gap=60 if mins >= 1440 else 15)
        cols[f"ret_{mins}m_atr"] = (p0 - pm) / a_h1
    # oscillateurs sur barres closes
    for tf, rule in (("m15", "15min"), ("h1", "1h"), ("d1", "1D")):
        b = aggregate(m1, rule)
        k = last_closed(b["end"].to_numpy(), T)
        ok = k >= 0
        kk = np.maximum(k, 0)
        cols[f"osc_rsi14_{tf}"] = np.where(ok, rsi(b["close"].to_numpy())[kk], np.nan)
        if tf != "m15":
            hh = pd.Series(b["high"].to_numpy()).rolling(20).max().to_numpy()[kk]
            ll = pd.Series(b["low"].to_numpy()).rolling(20).min().to_numpy()[kk]
            with np.errstate(invalid="ignore", divide="ignore"):
                cols[f"osc_pos_range20_{tf}"] = np.where(ok, (p0 - ll) / (hh - ll), np.nan)
    # bibliothèque de stratégies à trois échelles : état de la dernière barre d'entrée close
    for scale in ("M5", "H1", "H4"):
        lib = S.compute(m1, scale)
        k = last_closed(lib["bar_end"], T)
        kk = np.maximum(k, 0)
        for name, s in lib["signals"].items():
            if "dir" in s:
                cols[f"sig_{scale}_{name}"] = np.where(k >= 0, s["dir"][kk], 0).astype(np.float32)
            else:
                cols[f"sig_{scale}_{name}"] = np.where(k >= 0, s["active"][kk], False).astype(np.float32)
        cols[f"sig_{scale}_vol_rank"] = np.where(k >= 0, lib["vol_rank"][kk], np.nan)
    # autres marchés
    if with_cross:
        for name in INSTR:
            x = load_close(name)
            xt, xc = x.index.to_numpy(), x.to_numpy()
            q0 = price_at(xt, xc, T_min, max_gap=5)
            for mins in (5, 60, 240):
                qm = price_at(xt, xc, T_min - mins, max_gap=5)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cols[f"x_{name}_{mins}m"] = np.log(q0 / qm) * 1e4
    df = pd.DataFrame(cols)
    return df.astype(np.float32)
