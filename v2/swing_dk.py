"""Barres D1 Dukascopy (indices 2013 → 2021-08) au format des exports MT5, pour le développement swing des indices.
Jour = jour serveur FTMO (UTC+2, +3 en heure d'été US), comme les barres D1 MT5. Spec et spread : ceux du symbole FTMO."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import server_offset_hours
from .dukascopy import DK_DIR
from .swing import D1_DIR

MAP = {"DK_US500": "US500.cash", "DK_US100": "US100.cash", "DK_GER40": "GER40.cash",
       "DK_US30": "US30.cash", "DK_UK100": "UK100.cash", "DK_JP225": "JP225.cash"}


def build():
    for dk, ft in MAP.items():
        z = np.load(DK_DIR / f"{dk}.npz")
        t = z["t_utc"].astype(np.int64)
        ts = t + 2 * 3600
        ts = t + server_offset_hours(ts) * 3600
        df = pd.DataFrame(dict(day=ts // 86400 * 86400, open=z["open"], high=z["high"], low=z["low"], close=z["close"],
                               v=z["vol"]))
        g = df.groupby("day").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                  close=("close", "last"), tick_volume=("v", "size")).reset_index()
        g = g[g["tick_volume"] >= 60]                                      # jours quasi vides écartés
        ftd = pd.read_csv(D1_DIR / f"{ft}_D1.csv")
        frac = float(np.median(ftd["spread_px"].to_numpy()[-750:] / ftd["close"].to_numpy()[-750:]))
        g["spread_px"] = frac * g["close"]
        g = g.rename(columns={"day": "time"})
        g.to_csv(D1_DIR / f"{dk}_D1.csv", index=False)
        spec = json.loads((D1_DIR / f"{ft}_spec.json").read_text())
        spec.update(symbol=dk, source="dukascopy", first_bar=str(pd.Timestamp(int(g["time"].iloc[0]), unit="s").date()))
        (D1_DIR / f"{dk}_spec.json").write_text(json.dumps(spec, indent=1))
        print(dk, spec["first_bar"], "->", pd.Timestamp(int(g["time"].iloc[-1]), unit="s").date(), len(g), "jours")


if __name__ == "__main__":
    build()
