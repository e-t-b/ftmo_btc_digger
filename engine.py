"""Moteur de backtest événementiel, modèle d'exécution MT5.

Les prix OHLC sont des prix BID (comme les barres MT5). Ask = bid + spread_px.
  long  : entrée au ASK, sortie au BID ; le stop se déclenche quand le BID touche le niveau.
  short : entrée au BID, sortie au ASK ; le stop se déclenche quand l'ASK touche le niveau.
Décision à la clôture de la barre i, exécution à l'OUVERTURE de la barre i+1.
Dans une même barre : entrée à l'open, puis test du stop sur toute la barre (hypothèse défavorable),
puis mise à jour du stop suiveur (effective à partir de la barre suivante).
Coûts : spread, slippage (bps par côté), commission (bps par côté), swap (bps par franchissement du rollover)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAY_NS = 86_400 * 10 ** 9


@dataclass
class Costs:
    spread_mult: float = 1.0
    commission_bps: float = 0.0        # par côté, sur le notionnel
    slip_bps: float = 0.5              # par côté
    swap_long_bps_day: float = 0.0     # coût (positif) par franchissement du rollover
    swap_short_bps_day: float = 0.0
    rollover_utc_hour: int = 21        # ~ minuit heure serveur (GMT+2/+3)


def run(bars: pd.DataFrame, tgt: np.ndarray, stop: np.ndarray, costs: Costs,
        trail: bool, max_hold: int, start: int = 0) -> pd.DataFrame:
    o, h, l = (bars[x].to_numpy(float) for x in ("open", "high", "low"))
    sp = bars["spread_px"].to_numpy(float) * costs.spread_mult
    t = bars.index.to_numpy("datetime64[ns]").astype("int64")
    slip, com = costs.slip_bps / 1e4, costs.commission_bps / 1e4
    off = costs.rollover_utc_hour * 3600 * 10 ** 9
    n = len(o)
    pos, ent_j, e_bid, e_px, stop_px = 0, -1, 0.0, 0.0, np.nan
    rows = []

    def record(j, x_bid, x_px, reason):
        gross = pos * (x_bid - e_bid) / e_bid
        raw = pos * (x_px - e_px) / e_px
        n_roll = (t[j] - off) // DAY_NS - (t[ent_j] - off) // DAY_NS
        swap = n_roll * (costs.swap_long_bps_day if pos == 1 else costs.swap_short_bps_day) / 1e4
        rows.append((t[ent_j], t[j], pos, gross, raw - 2 * com - swap, j - ent_j, reason))

    for j in range(max(start, 2), n):
        i = j - 1
        ti, fresh = int(tgt[i]), tgt[i] != tgt[i - 1]
        if pos != 0 and (ti != pos or (max_hold and j - ent_j >= max_hold)):
            if pos == 1:
                record(j, o[j], o[j] * (1 - slip), "signal" if ti != pos else "time")
            else:
                record(j, o[j], (o[j] + sp[j]) * (1 + slip), "signal" if ti != pos else "time")
            pos = 0
        if pos == 0 and ti != 0 and fresh:
            pos, ent_j, e_bid = ti, j, o[j]
            e_px = (o[j] + sp[j]) * (1 + slip) if pos == 1 else o[j] * (1 - slip)
            sd = stop[i]
            stop_px = (e_px - sd if pos == 1 else e_px + sd) if sd > 0 else np.nan
        if pos != 0 and not np.isnan(stop_px):
            if pos == 1 and l[j] <= stop_px:
                xb = min(o[j], stop_px)
                record(j, xb, xb * (1 - slip), "stop")
                pos = 0
            elif pos == -1 and h[j] + sp[j] >= stop_px:
                xa = max(o[j] + sp[j], stop_px)
                record(j, xa - sp[j], xa * (1 + slip), "stop")
                pos = 0
        if pos != 0 and trail and stop[j] > 0:
            if pos == 1:
                new = h[j] - stop[j]
                stop_px = new if np.isnan(stop_px) else max(stop_px, new)
            else:
                new = l[j] + sp[j] + stop[j]
                stop_px = new if np.isnan(stop_px) else min(stop_px, new)

    df = pd.DataFrame(rows, columns=["entry_time", "exit_time", "side", "gross", "net", "bars", "reason"])
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df
