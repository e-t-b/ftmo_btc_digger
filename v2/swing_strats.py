"""Stratégies swing pré-enregistrées (docs/protocole_swing.md). Paramètres fixés à l'avance, standards publiés.

T1 Donchian 55/20 (Turtle système 2) : entrée sur clôture au-delà du plus haut/bas des 55 jours précédents,
   stop initial 2 × ATR20, sortie sur clôture au-delà du plus bas/haut des 20 jours précédents.
T2 Donchian 20/10 (Turtle système 1) : idem 20/10.
T3 Tendance MA 50/200 + cassure 20 j : long si SMA50 > SMA200 et clôture > plus haut 20 j (short symétrique),
   stop 3 × ATR20 suiveur (relevé à chaque clôture), sortie si le croisement s'inverse.
M1 RSI(2) indices, longs seulement : clôture > SMA200 et RSI2 < 10 → long ; sortie clôture > SMA5 ;
   stop catastrophe 2,5 × ATR20 ; sortie forcée après 10 jours."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import swing as S


def _ind(D):
    if "_ind" in D:
        return D["_ind"]
    c = pd.Series(D["c"])
    h, l = pd.Series(D["h"]), pd.Series(D["l"])
    tr = pd.Series(D["tr"])
    I = dict(
        atr20=tr.rolling(20).mean().to_numpy(),
        hh55=h.rolling(55).max().shift(1).to_numpy(), ll55=l.rolling(55).min().shift(1).to_numpy(),
        hh20=h.rolling(20).max().shift(1).to_numpy(), ll20=l.rolling(20).min().shift(1).to_numpy(),
        hh10=h.rolling(10).max().shift(1).to_numpy(), ll10=l.rolling(10).min().shift(1).to_numpy(),
        sma5=c.rolling(5).mean().to_numpy(), sma50=c.rolling(50).mean().to_numpy(), sma200=c.rolling(200).mean().to_numpy(),
    )
    d = c.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    rs = up.ewm(alpha=1 / 2, adjust=False).mean() / dn.ewm(alpha=1 / 2, adjust=False).mean()
    I["rsi2"] = (100 - 100 / (1 + rs)).to_numpy()
    D["_ind"] = I
    return I


def donchian(n_in, n_out, k_stop=2.0):
    def sig(D, i):
        I = _ind(D)
        hh, ll = I[f"hh{n_in}"][i], I[f"ll{n_in}"][i]
        if np.isnan(hh) or np.isnan(I["atr20"][i]):
            return 0
        return 1 if D["c"][i] > hh else (-1 if D["c"][i] < ll else 0)

    def ex(D, i, t):
        I = _ind(D)
        return D["c"][i] < I[f"ll{n_out}"][i] if t.d == 1 else D["c"][i] > I[f"hh{n_out}"][i]

    def stop(D, i, d, e):
        return e - d * k_stop * _ind(D)["atr20"][i]
    return sig, ex, stop


def ma_trend(k_stop=3.0):
    def sig(D, i):
        I = _ind(D)
        if np.isnan(I["sma200"][i]) or np.isnan(I["hh20"][i]):
            return 0
        if I["sma50"][i] > I["sma200"][i] and D["c"][i] > I["hh20"][i]:
            return 1
        if I["sma50"][i] < I["sma200"][i] and D["c"][i] < I["ll20"][i]:
            return -1
        return 0

    def ex(D, i, t):
        I = _ind(D)
        if (t.d == 1 and I["sma50"][i] < I["sma200"][i]) or (t.d == -1 and I["sma50"][i] > I["sma200"][i]):
            return True
        ns = D["c"][i] - t.d * k_stop * I["atr20"][i]
        t.stop = max(t.stop, ns) if t.d == 1 else min(t.stop, ns)
        return False

    def stop(D, i, d, e):
        return e - d * k_stop * _ind(D)["atr20"][i]
    return sig, ex, stop


def rsi2(k_stop=2.5, max_days=10):
    def sig(D, i):
        I = _ind(D)
        if np.isnan(I["sma200"][i]):
            return 0
        return 1 if (D["c"][i] > I["sma200"][i] and I["rsi2"][i] < 10) else 0

    def ex(D, i, t):
        return D["c"][i] > _ind(D)["sma5"][i] or (i + 1 - t.i_entry) >= max_days

    def stop(D, i, d, e):
        return e - d * k_stop * _ind(D)["atr20"][i]
    return sig, ex, stop


STRATS = {"T1": donchian(55, 20), "T2": donchian(20, 10), "T3": ma_trend(), "M1": rsi2()}


def run_symbol(D, sid, cost_mult=1.0, swap_mult=1.0):
    sig, ex, st = STRATS[sid]
    tr = S.simulate(D, sig, ex, st, cost_mult=cost_mult, swap_mult=swap_mult)
    df = S.to_frame(tr, D)
    df["sid"] = sid
    return df
