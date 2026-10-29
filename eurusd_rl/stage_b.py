"""Étape B, partie 1 : mesure de chaque stratégie seule, SANS apprentissage.

Porte de passage pré-enregistrée (fixée avant toute mesure, le 2026-10-29) — une variante est retenue si, sur
l'ENTRAÎNEMENT (2015-01 -> 2025-06, comptes mensuels) :
  - au moins 300 trades ;
  - espérance nette moyenne >= +0,05 R et t-stat (moyenne / erreur type) >= 3,0
    (seuil relevé pour ~33 variantes testées : correction de type Bonferroni à 5 %) ;
  - au moins 65 % des années civiles avec une espérance positive ;
  - espérance positive avec coûts stressés (spread x 1,5, glissements x 2) ;
puis, seulement pour les variantes retenues, espérance positive sur la CALIBRATION (2025-07 -> 2025-12).
La validation 2026 et le test protégé ne sont pas utilisés ici.
Variantes : TP 4R, TP 2R, TP 4R + filtre volatilité/session (11 x 3 = 33 essais).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategies as S
from .data import load_config, load_m1
from .evaluate import metrics, period_bounds
from .market import build_market
from .rules import per_year, run_rule, stop_distances

GATE = {"min_trades": 300, "min_mean_R": 0.05, "min_t": 3.0, "min_year_share": 0.65}
OUT = Path(__file__).parent / "reports"


def tstat(r: np.ndarray) -> float:
    return float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))) if len(r) > 2 and r.std() > 0 else 0.0


def summary(tr, st, cnt, mk, cfg):
    m = metrics(tr, st, cnt, mk, cfg)
    r = tr["R"].to_numpy() if len(tr) else np.zeros(0)
    yrs = per_year(tr)
    pos_years = np.mean([v["mean_R"] > 0 for v in yrs.values()]) if yrs else 0.0
    return {"n_trades": m.get("n_trades", 0), "trades_per_day": m.get("trades_per_day"), "win_rate": m.get("win_rate"),
            "mean_R": m.get("mean_R"), "t_stat": round(tstat(r), 2), "profit_factor": m.get("profit_factor"),
            "total_R": m.get("total_R"), "worst_month_pct": None, "violations": m.get("violations"),
            "pos_year_share": round(float(pos_years), 3), "by_year": yrs, "exit_reasons": m.get("exit_reasons"),
            "buy_sell": m.get("buy_sell")}


def main():
    t0 = time.time()
    cfg = load_config()
    m1 = load_m1(cfg)
    mk = build_market(cfg, m1)
    cfg_s = copy.deepcopy(cfg)
    cfg_s["execution"]["spread"]["stress_multiplier"] = 1.5
    cfg_s["execution"]["entry_slippage_pts"] *= 2; cfg_s["execution"]["exit_slippage_pts"] *= 2
    cfg_s["execution"]["stop_slippage_pts"] *= 2
    mk_s = build_market(cfg_s, m1)
    lib = S.compute(m1)
    D = S.to_decisions(lib, mk)
    filt = D["vol_session_filter"]["active"]
    tlo, thi = period_bounds(mk, cfg["splits"]["train"])
    clo, chi = period_bounds(mk, cfg["splits"]["calibration"])
    res = {"library_version": lib["version"], "gate": GATE, "variants": {}}
    for name in S.DIRECTIONAL:
        for var, tp, use_f in (("tp4", 4.0, False), ("tp2", 2.0, False), ("tp4_filtre", 4.0, True)):
            c = copy.deepcopy(cfg); c["trade"]["take_profit_r"] = tp
            cs = copy.deepcopy(cfg_s); cs["trade"]["take_profit_r"] = tp
            dirs = D[name]["dir"] * (filt if use_f else 1)
            d_ok, sd = stop_distances(dirs, D[name]["stop_px"], D["_close"], mk.dec_min_stop)
            tr, st, cnt = run_rule(mk, c, tlo, thi, d_ok, sd)
            train = summary(tr, st, cnt, mk, c)
            d_ok_s, sd_s = stop_distances(dirs, D[name]["stop_px"], D["_close"], mk_s.dec_min_stop)
            tr_s, st_s, cnt_s = run_rule(mk_s, cs, tlo, thi, d_ok_s, sd_s)
            stress = summary(tr_s, st_s, cnt_s, mk_s, cs)
            passed = (train["n_trades"] >= GATE["min_trades"] and (train["mean_R"] or -9) >= GATE["min_mean_R"]
                      and train["t_stat"] >= GATE["min_t"] and train["pos_year_share"] >= GATE["min_year_share"]
                      and (stress["mean_R"] or -9) > 0)
            rec = {"train": train, "train_stress": {k: stress[k] for k in ("n_trades", "mean_R", "t_stat", "profit_factor")},
                   "pass_train_gate": bool(passed)}
            if passed:
                trc, stc, cntc = run_rule(mk, c, clo, chi, d_ok, sd)
                cal = summary(trc, stc, cntc, mk, c)
                rec["calibration"] = cal
                rec["pass_calibration"] = bool((cal["mean_R"] or -9) > 0)
            res["variants"][f"{name}/{var}"] = rec
            print(f"{name:22s} {var:11s} n={train['n_trades']:5d} R/trade={train['mean_R']} t={train['t_stat']} "
                  f"PF={train['profit_factor']} années+={train['pos_year_share']} stress={stress['mean_R']} "
                  f"porte={'OUI' if passed else 'non'}" + (f" calib={rec['calibration']['mean_R']}" if passed else ""),
                  flush=True)
    res["elapsed_s"] = round(time.time() - t0)
    (OUT / "stage_b_strategies.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def gross():
    """Diagnostic : espérance BRUTE (aucun coût) de chaque stratégie, TP 4R, entraînement uniquement."""
    cfg = load_config()
    m1 = load_m1(cfg)
    c0 = copy.deepcopy(cfg)
    ex = c0["execution"]
    ex["commission_per_lot_side"] = 0
    ex["entry_slippage_pts"] = ex["exit_slippage_pts"] = ex["stop_slippage_pts"] = 0
    ex["spread"]["stress_multiplier"] = 0.0
    mk = build_market(c0, m1)
    D = S.to_decisions(S.compute(m1), mk)
    lo, hi = period_bounds(mk, cfg["splits"]["train"])
    out = {}
    for name in S.DIRECTIONAL:
        d, sd = stop_distances(D[name]["dir"], D[name]["stop_px"], D["_close"], mk.dec_min_stop)
        tr, st, cnt = run_rule(mk, c0, lo, hi, d, sd)
        r = tr["R"].to_numpy()
        out[name] = {"n_trades": int(len(r)), "gross_mean_R": round(float(r.mean()), 4), "t_stat": round(tstat(r), 2),
                     "stop_pips_median": round(float(np.median(tr["stop_dist"]) * 1e4), 1), "by_year": per_year(tr)}
        print(name, out[name]["n_trades"], out[name]["gross_mean_R"], out[name]["t_stat"], flush=True)
    (OUT / "stage_b_gross.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    import sys
    gross() if "--gross" in sys.argv else main()
