"""Agrège les runs ml_fx.py : tableau comparatif + deflated Sharpe (tous les essais comptés).

Usage : python summarize_grid.py results_ml"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common

roots = [Path(x) for x in (sys.argv[1:] or ["results_ml"])]
rows, series = [], {}
for rp in [p for root in roots for p in sorted(glob.glob(str(root / "report_*.json")))]:
    root = Path(rp).parent
    r = json.load(open(rp))
    tag = Path(rp).stem.replace("report_", "")
    start, end = pd.Timestamp(r["oos"][0]), pd.Timestamp(r["oos"][1])
    for thr, s in r["thresholds"].items():
        tf = root / f"trades_{tag}_thr{float(thr):g}.csv"
        tr = pd.read_csv(tf, parse_dates=["entry_time", "exit_time"]) if tf.exists() else pd.DataFrame()
        d = common.daily_returns(tr, start, end) if len(tr) else np.zeros((end.normalize() - start.normalize()).days + 1)
        key = (root.name, r["pair"], r["k"], r["horizon"], float(thr))
        series[key] = d
        rows.append(dict(jeu=root.name, pair=r["pair"], k=r["k"], H=r["horizon"], seuil_R=float(thr), n_trades=s.get("n_trades", 0),
                         par_jour=s.get("trades_per_day"), brut_bps=s.get("gross_bps"), net_bps=s.get("net_bps"),
                         sharpe=s.get("sharpe"), psr=s.get("psr"), ic_bps=r["ic_bps"],
                         top_decile_brut=r["decile_gross_bps"].get("9"), bas_decile_brut=r["decile_gross_bps"].get("0")))
T = pd.DataFrame(rows).sort_values(["jeu", "pair", "k", "seuil_R"])
active = {k: v for k, v in series.items() if (v != 0).sum() >= 10}
daily_sr = np.array([v.mean() / v.std(ddof=1) for v in active.values() if v.std() > 0])
best_key = max(active, key=lambda k: active[k].mean() / max(active[k].std(ddof=1), 1e-12)) if active else None
pd.set_option("display.width", 220)
print(T.to_string(index=False))
print(f"\nEssais comptés (≥ 10 jours actifs) : {len(daily_sr)}")
if best_key is not None:
    d = active[best_key]
    print(f"Meilleur essai par Sharpe : {best_key} | Sharpe annualisé {common.sharpe(d):.2f} | "
          f"DSR (corrigé des {len(daily_sr)} essais) = {common.dsr(d, daily_sr, len(daily_sr)):.4f}")
