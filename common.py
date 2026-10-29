"""Utilitaires communs : resampling, conversion heure serveur MT5 -> UTC, métriques."""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
       "spread_px": "mean", "tick_volume": "sum"}


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Bougies [t, t+rule) indexées sur l'heure d'OUVERTURE. Identique en backtest et en live."""
    out = df.resample(rule, label="left", closed="left").agg(AGG)
    return out.dropna(subset=["close"])


def server_to_utc(idx, mode: str = "eet_us_dst", fixed_hours: int = 0) -> pd.DatetimeIndex:
    """Heure serveur MT5 (naïve) -> UTC naïf.
    'eet_us_dst' : convention courante des courtiers MT5 (GMT+2 l'hiver, GMT+3 pendant l'heure d'été US).
    'fixed'      : décalage constant `fixed_hours`."""
    idx = pd.DatetimeIndex(idx)
    if mode == "fixed":
        return idx - pd.Timedelta(hours=fixed_hours)
    days = pd.DatetimeIndex(idx.normalize().unique())
    noon = (days + pd.Timedelta(hours=12)).tz_localize("UTC").tz_convert("America/New_York")
    flag = pd.Series([t.dst() != pd.Timedelta(0) for t in noon], index=days)
    offs = 2 + flag.reindex(idx.normalize()).to_numpy().astype(int)
    return idx - pd.to_timedelta(offs, unit="h")


# ----------------------------------------------------------------------------- métriques
def daily_returns(tr: pd.DataFrame, start, end) -> np.ndarray:
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    if tr.empty:
        return np.zeros(len(days))
    s = tr["net"].groupby(tr["exit_time"].dt.normalize()).sum()
    return s.reindex(days, fill_value=0.0).to_numpy()


def sharpe(d: np.ndarray) -> float:
    s = d.std(ddof=1)
    return float(d.mean() / s * np.sqrt(365)) if s > 0 else 0.0


def psr(d: np.ndarray, sr0_daily: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado) sur rendements journaliers."""
    n, s = len(d), d.std(ddof=1)
    if n < 30 or s == 0:
        return float("nan")
    sr = d.mean() / s
    g3, g4 = skew(d), kurtosis(d, fisher=False)
    den = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr0_daily) * np.sqrt(n - 1) / den))


def dsr(d: np.ndarray, trials_sr_daily: np.ndarray, n_trials: int) -> float:
    """Deflated Sharpe Ratio : PSR contre le Sharpe maximal attendu par hasard sur n_trials essais."""
    g = 0.5772156649
    v = float(np.var(trials_sr_daily, ddof=1))
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    return psr(d, sr0)


def summarize(tr: pd.DataFrame, start, end, extra_cost_bps_rt: float = 0.0) -> dict:
    if tr.empty:
        return dict(n_trades=0)
    t = tr.assign(net=tr["net"] - extra_cost_bps_rt / 1e4)
    d = daily_returns(t, start, end)
    cum = np.cumsum(d)
    return dict(n_trades=len(t), trades_per_day=round(len(t) / len(d), 2),
                hit_rate=round(float((t["net"] > 0).mean()), 3),
                gross_bps=round(float(t["gross"].mean() * 1e4), 2),
                net_bps=round(float(t["net"].mean() * 1e4), 2),
                net_total_pct=round(float(t["net"].sum() * 100), 1),
                sharpe=round(sharpe(d), 2), max_dd_pct=round(float((cum - np.maximum.accumulate(cum)).min() * 100), 1),
                psr=round(psr(d), 4))
