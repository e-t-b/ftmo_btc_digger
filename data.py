"""Chargement des données 1m : Parquet Binance (spread supposé) ou CSV exporté de MT5 (spread réel).

Convention interne : DataFrame 1m indexé en UTC naïf (heure d'ouverture de la bougie), colonnes
open/high/low/close (prix BID), spread_px (écart ask-bid en unités de prix), tick_volume."""
import numpy as np
import pandas as pd
from scipy.signal import lfilter

from common import server_to_utc


def load_binance_parquet(path: str, spread_bps: float) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("open_time").sort_index()
    out = df[["open", "high", "low", "close"]].astype(float).copy()
    out["spread_px"] = out["close"] * spread_bps / 1e4
    out["tick_volume"] = df["volume"].astype(float)
    out.index.name = "time"
    return out


def load_mt5_csv(path: str, tz_mode: str = "eet_us_dst", tz_hours: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    idx = server_to_utc(pd.to_datetime(df["time"], unit="s"), tz_mode, tz_hours)
    out = pd.DataFrame({"open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float),
                        "low": df["low"].to_numpy(float), "close": df["close"].to_numpy(float),
                        "spread_px": df["spread_px"].to_numpy(float),
                        "tick_volume": df["tick_volume"].to_numpy(float)},
                       index=pd.DatetimeIndex(idx, name="time"))
    out = out[~out.index.duplicated(keep="first")].sort_index()
    med = out["spread_px"].median()
    out["spread_px"] = out["spread_px"].where(out["spread_px"] > 0, med)   # spread 0 = donnée manquante
    return out


def make_synthetic(mode: str = "null", days: int = 900, seed: int = 11, spread_bps: float = 2.0) -> pd.DataFrame:
    """1m synthétique pour valider la mécanique (pas pour juger une stratégie).
    null    : marche aléatoire -> aucune stratégie ne doit gagner net.
    trend   : composante de tendance persistante (demi-vie ~12 h) -> les suiveurs de tendance doivent la voir.
    meanrev : bruit de retour à la moyenne (demi-vie ~70 min) -> la réversion doit la voir."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    idx = pd.date_range("2021-01-01", periods=n, freq="min", name="time")
    sigma = 0.0006 * np.exp(lfilter([1], [1, -0.9995], rng.normal(size=n) * 0.01))
    r = sigma * rng.normal(size=n)
    x = np.zeros(n)
    if mode == "trend":
        r = r + 0.03 * sigma * lfilter([1], [1, -0.999], rng.normal(size=n) * 0.02) * 1.0
    elif mode == "meanrev":
        x = lfilter([1], [1, -0.99], sigma * 0.35 * rng.normal(size=n))
    lp = np.log(30000.0) + np.cumsum(r) + x
    close = np.exp(lp)
    open_ = np.r_[close[0], close[:-1]]
    w = sigma * 0.5
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(size=n)) * w)
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(size=n)) * w)
    vol = np.exp(rng.normal(size=n) * 0.5) * 10
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "spread_px": close * spread_bps / 1e4, "tick_volume": vol}, index=idx)
