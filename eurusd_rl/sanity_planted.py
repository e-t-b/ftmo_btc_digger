"""Diagnostic d'apprentissabilité : la chaîne PPO apprend-elle un signal connu ?

On part de vrais prix EURUSD (2023-07 -> 2024-09). Chaque jour, une dérive de `drift_pips_h` pips/heure est ajoutée
pendant la fenêtre de trading, dans un sens tiré au hasard ; ce sens est fourni à l'agent dans une colonne de contexte
(dow_sin, remplacée). Une politique qui suit l'indice doit gagner nettement. Si PPO n'y parvient pas, le problème
vient de la chaîne d'apprentissage, pas de l'absence d'avantage dans les données réelles.
Ce module ne touche ni à la validation ni au test.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import pandas as pd

from .data import ROOT, load_config, load_m1
from .evaluate import metrics, period_bounds, run_policy
from .market import build_market, CTX_FEATS
from .train import train

CUE_COL = CTX_FEATS.index("dow_sin")


def planted_market(cfg, drift_pips_h: float, seed: int = 0):
    m1 = load_m1(cfg)
    m1 = m1[(m1["t_utc"] >= "2023-06-01") & (m1["t_utc"] < "2024-10-01")].reset_index(drop=True)
    ts = pd.DatetimeIndex(m1["t_server"])
    day = (ts.values.astype("datetime64[D]").astype(np.int64))
    mod = ts.hour * 60 + ts.minute
    rng = np.random.default_rng(seed)
    udays = np.unique(day)
    sign = pd.Series(rng.choice([-1.0, 1.0], len(udays)), index=udays)
    s = sign.reindex(day).to_numpy()
    start = 9 * 60                                     # 02:00 NY
    elapsed_h = np.clip((mod - start) / 60.0, 0, None)
    shift = s * drift_pips_h * 1e-4 * elapsed_h        # dérive cumulée depuis 02:00 NY (nulle avant)
    shift_open = s * drift_pips_h * 1e-4 * np.clip((mod - start - 1) / 60.0, 0, None)
    m1 = m1.copy()
    m1["open"] = m1["open"] + shift_open
    m1["close"] = m1["close"] + shift
    m1["high"] = np.maximum(m1["high"] + shift, np.maximum(m1["open"], m1["close"]))
    m1["low"] = np.minimum(m1["low"] + shift_open, np.minimum(m1["open"], m1["close"]))
    c = copy.deepcopy(cfg)
    c["splits"]["train"] = ["2023-07-03", "2024-06-30"]
    c["splits"]["calibration"] = ["2024-07-01", "2024-09-30"]
    c["execution"]["spread"]["profile_fit"] = ["2023-06-01", "2023-12-31"]
    mk = build_market(c, m1)
    dd = mk.dec_day
    mk.ctx[:, CUE_COL] = sign.reindex(dd).to_numpy().astype(np.float32)   # indice = sens de la dérive du jour
    return c, mk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=float, default=4.0)
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--config", default="A2.yaml")
    ap.add_argument("--run", default="runs/sanity_planted")
    a = ap.parse_args()
    cfg = load_config(a.config)
    c, mk = planted_market(cfg, a.drift)
    c["ppo"]["eval_every"] = 20
    train(c, ROOT / a.run, 1, a.updates, mk=mk)
    lo, hi = period_bounds(mk, c["splits"]["calibration"])
    # référence oracle : suivre l'indice à la première décision du jour (règle explicite)
    print(json.dumps({"drift_pips_h": a.drift}, indent=0))


if __name__ == "__main__":
    main()
