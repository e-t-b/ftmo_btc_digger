"""Recherche d'une seconde stratégie décorrélée de S26-B3 (entraînement uniquement : Dukascopy → 2021-08, FTMO 2021-09 → 2024-06).
Usage : python -m v2.combo"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from . import dukascopy as DK
from . import lot4, s26x
from .catalogue_run import prepared
from .jalon_a import tstat

IDX = {"dukascopy": ("DK_US500", "DK_US100", "DK_GER40"), "ftmo": ("US500.cash", "US100.cash", "GER40.cash")}
WIN = {"dukascopy": (0, DK.END), "ftmo": (s26x.FTMO_START, s26x.FTMO_END)}
CANDS = {  # stratégie : (fonction, configuration)
    "S2 volatility breakout": ("S2", dict(k=0.3, s=0.5, filt=1)),
    "S3 cassure veille": ("S3", None), "S4 Donchian H1": ("S4", None), "S5 VWAP": ("S5", None),
    "S6 Bollinger range": ("S6", None), "S7 Turtle Soup": ("S7", None), "S8 IBS": ("S8", None),
    "S9 retournement fin de séance": ("S9", {"fixed": 1}), "S10 avant Fed": ("S10", {}),
    "S11 tournant de mois": ("S11", {}), "S16 ORB 5 min": ("S16", {}), "S17 Noise Area": ("S17", {}),
    "S21 stretch Crabel": ("S21", {}), "S23 ORB NR7": ("S23", {}), "S24 après Fed": ("S24", {}),
    "S25 veille férié": ("S25", {}), "S27 réintégration OPR": ("S27", {}),
}


def daily(tr, key="r_net"):
    d = pd.to_datetime(tr["entry_t"], unit="s").dt.tz_localize("UTC").dt.tz_convert("Europe/Prague").dt.date
    return tr.groupby(d)[key].sum()


def main():
    s26x._register_dk()
    cal = DK.calendars_dk()
    lot4._CAL = cal
    res = {}
    base = {}
    Ps = {}
    for src, syms in IDX.items():
        Ps[src] = {s: (K.prepare(s) if src == "dukascopy" else prepared(s)) for s in syms}
        a, b = WIN[src]
        parts = []
        for s, P in Ps[src].items():
            r = K.run(P, "S26x", {}, setups=s26x.setups(P, hours=(1, 2, 3)))["trades"]
            parts.append(r[(r.entry_t >= a) & (r.entry_t < b)])
        base[src] = pd.concat(parts)
    for name, (sid, cfg) in CANDS.items():
        row = {}
        for src, syms in IDX.items():
            a, b = WIN[src]
            parts = []
            for s, P in Ps[src].items():
                if sid == "S24" and "GER40" in s:
                    continue
                c = cfg if cfg is not None else K.grid(sid)[0]
                r = DK.runner(P, sid, c, cal)["trades"]
                parts.append(r[(r.entry_t >= a) & (r.entry_t < b)])
            tr = pd.concat(parts)
            x = tr["r_net"].to_numpy()
            d1, d2 = daily(base[src]), daily(tr)
            j = d1.index.union(d2.index)
            corr = float(np.corrcoef(d1.reindex(j, fill_value=0), d2.reindex(j, fill_value=0))[0, 1]) if len(x) > 5 else np.nan
            # quand S26 perd sa journée, que fait le candidat ?
            bad = d1[d1 < 0].index
            on_bad = float(d2.reindex(bad).dropna().mean()) if len(d2.reindex(bad).dropna()) else np.nan
            row[src] = dict(n=int(len(x)), R=round(float(x.mean()), 4) if len(x) else None, t=round(tstat(x), 2) if len(x) > 2 else None,
                            corr_S26=round(corr, 3), R_jours_perdants_S26=round(on_bad, 3))
        ok = all((row[s]["R"] or -1) > 0 for s in IDX) and all((row[s]["corr_S26"] if row[s]["corr_S26"] == row[s]["corr_S26"] else 1) < 0.15 for s in IDX)
        row["retenu"] = bool(ok)
        res[name] = row
        print(f"{name:32s} DK n={row['dukascopy']['n']:5d} R={row['dukascopy']['R']} corr={row['dukascopy']['corr_S26']} | "
              f"FTMO n={row['ftmo']['n']:5d} R={row['ftmo']['R']} corr={row['ftmo']['corr_S26']} | jours où S26 perd : "
              f"{row['dukascopy']['R_jours_perdants_S26']} / {row['ftmo']['R_jours_perdants_S26']}  {'<- RETENU' if ok else ''}", flush=True)
    Path("reports_v2/combo_screen.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
