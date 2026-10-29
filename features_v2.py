"""Bloc de features v2 (pré-enregistré avant tout résultat) :
  1. indicateurs techniques classiques sur 15m / 1h / 4h : RSI, MACD, Bollinger %b, stochastique, ADX/DI, CCI, distance MA50/MA200
  2. microstructure issue des bougies M1 : skew, kurtosis, efficacité du chemin, déséquilibre de tick-volume signé, range moyen
     (fenêtres 1 / 4 / 16 barres de 15 min)
  3. flux inter-actifs : rendements BTCUSD (sentiment de risque, marché 24h/24) à 15m / 1h / 4h
  4. calendrier : 1er vendredi du mois (jour NFP) et sa fenêtre, fin de mois, jour du mois
Toutes les features sont calculées à la clôture de la barre de décision (aucune information future)."""
import numpy as np
import pandas as pd

from strategies import complete_bars

BAR = pd.Timedelta("15min")


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _adx(b, n=14):
    h, l, c = b["high"], b["low"], b["close"]
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=b.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=b.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi - mdi


def indicators(bars, p):
    c = bars["close"]
    f = pd.DataFrame(index=bars.index)
    sd = c.diff().rolling(50).std()
    f[f"{p}_rsi14"] = _rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f[f"{p}_macd_h"] = (macd - macd.ewm(span=9, adjust=False).mean()) / sd
    f[f"{p}_bb_b"] = (c - c.rolling(20).mean()) / (2 * c.rolling(20).std())
    lo, hi = bars["low"].rolling(14).min(), bars["high"].rolling(14).max()
    f[f"{p}_stoch"] = (c - lo) / (hi - lo).replace(0, np.nan)
    f[f"{p}_adx"], f[f"{p}_di_diff"] = _adx(bars)
    tp = (bars["high"] + bars["low"] + c) / 3
    sma = tp.rolling(20).mean()
    f[f"{p}_cci"] = (tp - sma) / (0.015 * (tp - sma).abs().rolling(20).mean())
    for n in (50, 200):
        f[f"{p}_ma{n}_dist"] = (c - c.rolling(n).mean()) / sd
    return f


def micro(m1, idx):
    r = np.log(m1["close"]).diff()
    g = pd.DataFrame({"r": r, "r2": r ** 2, "r3": r ** 3, "r4": r ** 4, "ar": r.abs(),
                      "sv": m1["tick_volume"] * np.sign(m1["close"] - m1["open"]), "v": m1["tick_volume"],
                      "rng": (m1["high"] - m1["low"]) / m1["close"]})
    s = g.resample("15min", label="left", closed="left").sum(min_count=1)
    s["n"] = m1["close"].resample("15min", label="left", closed="left").count()
    s = s.reindex(idx)
    out = pd.DataFrame(index=idx)
    for w, nm in ((1, "1"), (4, "4"), (16, "16")):
        a = s.rolling(w, min_periods=w).sum()
        n = a["n"].replace(0, np.nan)
        v2 = a["r2"] / n
        out[f"mic{nm}_skew"] = (a["r3"] / n) / v2.pow(1.5)
        out[f"mic{nm}_kurt"] = (a["r4"] / n) / v2.pow(2)
        out[f"mic{nm}_eff"] = a["r"].abs() / a["ar"].replace(0, np.nan)
        out[f"mic{nm}_imb"] = a["sv"] / a["v"].replace(0, np.nan)
        out[f"mic{nm}_rng"] = a["rng"] / n
    return out


def cross_btc(m1_btc, idx):
    dec = idx + BAR
    parts = []
    for rule, pre, lags in (("15min", "b15", (1, 4, 12)), ("1h", "bh1", (1, 4)), ("4h", "bh4", (1,))):
        lc = np.log(complete_bars(m1_btc, rule)["close"])
        f = pd.DataFrame({f"{pre}_ret{n}": lc.diff(n) for n in lags})
        if rule == "15min":
            f["b15_rv20"] = lc.diff().rolling(20).std()
        f.index = f.index + pd.Timedelta(rule)
        parts.append(f.reindex(dec, method="ffill").set_axis(idx))
    return pd.concat(parts, axis=1)


def calendar(idx):
    dec = idx + BAR
    dow, dom = dec.dayofweek.to_numpy(), dec.day.to_numpy()
    mins = (dec.hour * 60 + dec.minute).to_numpy()
    out = pd.DataFrame(index=idx)
    out["cal_first_friday"] = ((dow == 4) & (dom <= 7)).astype(float)
    out["cal_nfp_window"] = ((dow == 4) & (dom <= 7) & (mins >= 12 * 60 + 30) & (mins <= 14 * 60 + 30)).astype(float)
    out["cal_dom"] = dom.astype(float)
    out["cal_month_end"] = (dec.days_in_month.to_numpy() - dom <= 2).astype(float)
    return out


def extra_features(m1, base, m1_btc=None):
    idx = base.index
    dec = idx + BAR
    parts = []
    for rule, pre in (("15min", "i15"), ("1h", "ih1"), ("4h", "ih4")):
        bars = base if rule == "15min" else complete_bars(m1, rule)
        f = indicators(bars, pre)
        f.index = f.index + pd.Timedelta(rule)
        parts.append(f.reindex(dec, method="ffill").set_axis(idx))
    parts.append(micro(m1, idx))
    if m1_btc is not None:
        parts.append(cross_btc(m1_btc, idx))
    parts.append(calendar(idx))
    return pd.concat(parts, axis=1)
