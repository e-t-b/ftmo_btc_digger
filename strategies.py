"""Stratégies simples. Chaque stratégie est une fonction PURE de l'historique des barres COMPLÈTES :
    tgt[i]  : position cible (-1, 0, +1) après la clôture de la barre i
    stop[i] : distance de stop en prix, mesurée à la clôture de i (0 = pas de stop)
Rien ne regarde au-delà de la barre i. Les mêmes fonctions servent au backtest ET au bot live.

Règle d'entrée commune (moteur et bot) : on n'ouvre que sur un CHANGEMENT de tgt (tgt[i] != tgt[i-1]).
Après un stop ou une sortie temporelle, on n'entre donc plus tant que le signal n'a pas changé : pas
d'état caché à persister, et un redémarrage du bot ne rejoue pas d'anciens signaux."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import resample


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    tf: str
    params: tuple          # ((clé, valeur), ...) -> hashable
    trail: bool            # stop suiveur (chandelier) ou stop fixe
    max_hold: int          # sortie temporelle en barres (0 = aucune)
    warmup: int            # barres nécessaires pour reproduire exactement le signal

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def tf_min(self) -> int:
        return int(pd.Timedelta(self.tf).total_seconds() // 60)


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    c1 = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - c1).abs(), (bars["low"] - c1).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _finish(tgt: np.ndarray, stop_k: float, bars: pd.DataFrame):
    stop = (stop_k * atr(bars)).to_numpy()
    tgt = tgt.astype(int)
    tgt[np.isnan(stop)] = 0
    return tgt, np.nan_to_num(stop)


def sig_donchian(bars: pd.DataFrame, p: dict):
    n = p["n"]
    hi = bars["high"].rolling(n).max().shift(1)
    lo = bars["low"].rolling(n).min().shift(1)
    raw = np.where(bars["close"] > hi, 1.0, np.where(bars["close"] < lo, -1.0, np.nan))
    tgt = pd.Series(raw, index=bars.index).ffill().fillna(0).to_numpy()
    return _finish(tgt, p["stop_k"], bars)


def sig_ema(bars: pd.DataFrame, p: dict):
    c = bars["close"]
    ef = c.ewm(span=p["fast"], adjust=False).mean()
    es = c.ewm(span=p["slow"], adjust=False).mean()
    return _finish(np.sign((ef - es).to_numpy()), p["stop_k"], bars)


def sig_orb(bars: pd.DataFrame, p: dict, tf_min: int):
    """Cassure de la fourchette d'ouverture (heures UTC), sortie forcée en fin de session."""
    sh, rh, eh = p["start"], p["range_h"], p["end"]
    idx = bars.index
    minute = (idx.hour * 60 + idx.minute).to_numpy()
    day = idx.normalize().to_numpy()
    h, l, c = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    n = len(bars)
    tgt = np.zeros(n)
    r0, r1, end = sh * 60, (sh + rh) * 60, eh * 60
    cuts = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    for a, b in zip(cuts[:-1], cuts[1:]):
        m = minute[a:b]
        inr = (m >= r0) & (m < r1)
        if not inr.any():
            continue
        hi, lo = h[a:b][inr].max(), l[a:b][inr].min()
        cur = 0.0
        for q in range(a, b):
            mq = minute[q]
            if mq < r1:
                continue
            if mq + tf_min >= end:
                break
            if c[q] > hi:
                cur = 1.0
            elif c[q] < lo:
                cur = -1.0
            tgt[q] = cur
    return _finish(tgt, p["stop_k"], bars)


def sig_bb(bars: pd.DataFrame, p: dict):
    """Réversion Bollinger : entrée à |z| > z, sortie au retour de z à 0."""
    c = bars["close"]
    z = ((c - c.rolling(p["n"]).mean()) / c.rolling(p["n"]).std()).to_numpy()
    kz = p["z"]
    tgt = np.zeros(len(z))
    pos = 0
    for i in range(len(z)):
        zi = z[i]
        if not np.isfinite(zi):
            pos = 0
        elif pos == 0:
            pos = 1 if zi < -kz else (-1 if zi > kz else 0)
        elif pos == 1 and zi >= 0:
            pos = 0
        elif pos == -1 and zi <= 0:
            pos = 0
        tgt[i] = pos
    return _finish(tgt, p["stop_k"], bars)


def build_signals(cfg: Config, bars: pd.DataFrame):
    if cfg.family == "donchian":
        return sig_donchian(bars, cfg.p)
    if cfg.family == "ema":
        return sig_ema(bars, cfg.p)
    if cfg.family == "orb":
        return sig_orb(bars, cfg.p, cfg.tf_min)
    if cfg.family == "bb":
        return sig_bb(bars, cfg.p)
    raise ValueError(cfg.family)


def default_configs() -> list:
    """Grille volontairement petite : chaque configuration est un ESSAI compté dans le deflated Sharpe."""
    out = []
    for tf, ns in (("1h", (24, 48, 96)), ("4h", (20, 40))):
        for n in ns:
            for k in (2.0, 3.0):
                out.append(Config(f"donch_{tf}_n{n}_k{k:g}", "donchian", tf, (("n", n), ("stop_k", k)), True, 0, n + 150))
    for f, s in ((8, 32), (12, 48), (24, 96)):
        out.append(Config(f"ema_1h_{f}_{s}", "ema", "1h", (("fast", f), ("slow", s), ("stop_k", 3.0)), True, 0, 8 * s))
    for sh in (0, 8, 13):
        for rh in (1, 2):
            out.append(Config(f"orb_15m_s{sh}_r{rh}", "orb", "15min",
                              (("start", sh), ("range_h", rh), ("end", 21), ("stop_k", 2.0)), False, 0, 220))
    for n in (48, 96):
        for z in (2.0, 3.0):
            out.append(Config(f"bb_15m_n{n}_z{z:g}", "bb", "15min", (("n", n), ("z", z), ("stop_k", 3.0)), False, 48, n + 600))
    return out


def complete_bars(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Bougies de timeframe `tf` entièrement formées (la dernière, partielle, est écartée)."""
    bars = resample(m1, tf)
    end_data = m1.index[-1] + pd.Timedelta("1min")
    return bars[bars.index + pd.Timedelta(tf) <= end_data]


def latest_signal(cfg: Config, m1: pd.DataFrame):
    """Décision à la clôture de la dernière barre complète (utilisée par le bot).
    Retourne un dict (tgt, fresh, stop, bar_start, high, low, spread) ou None si l'historique est insuffisant."""
    bars = complete_bars(m1, cfg.tf)
    if len(bars) < cfg.warmup + 5:
        return None
    tgt, stop = build_signals(cfg, bars)
    last = bars.iloc[-1]
    return dict(tgt=int(tgt[-1]), fresh=bool(tgt[-1] != tgt[-2]), stop=float(stop[-1]), bar_start=bars.index[-1],
                high=float(last["high"]), low=float(last["low"]), spread=float(last["spread_px"]))
