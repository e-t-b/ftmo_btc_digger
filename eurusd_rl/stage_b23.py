"""Étape B2 (intraday, échelle H1) et B3 (swing, échelle H4) : stratégies seules, SANS apprentissage.

Protocole pré-enregistré le 2026-10-29, avant toute mesure :
  B2 : entrée H1 (signal émis à la clôture H1, sans retard), structure H4, tendance D1, niveaux de la veille ;
       intraday (clôture forcée 16:45 NY) ; stop plafonné à 60 pips ; comptes mensuels.
  B3 : entrée H4 (signal émis au plus 60 min après la clôture H4), structure D1, tendance W1, niveaux de la semaine
       précédente ; positions sur plusieurs jours (compte Swing), swaps FTMO, week-ends ; détention max 10 jours de
       bourse ; stop plafonné à 150 pips ; comptes trimestriels. ORB, VWAP et range asiatique exclus (intraday).
  Variantes : TP 4R et TP 2R. Mesures sur l'ENTRAÎNEMENT : net, coûts stressés, brut (sans coûts).
  Porte : >= 300 trades (B2) / >= 150 trades (B3) ; espérance nette >= +0,05 R et t >= 3,2 (~70 essais cumulés depuis
  l'étape B) ; >= 65 % d'années positives ; espérance positive sous coûts stressés. Calibration seulement si porte.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import strategies as S
from .data import load_config, load_m1
from .evaluate import period_bounds
from .market import build_market
from .rules import per_year, run_rule, stop_distances
from .stage_b import summary, tstat

OUT = Path(__file__).parent / "reports"
PROTO = {
    "B2": dict(scale="H1", overnight=False, stale=0, max_stop=60.0, max_hold=None, seg="monthly", min_trades=300),
    "B3": dict(scale="H4", overnight=True, stale=60, max_stop=150.0, max_hold=10, seg="quarterly", min_trades=150),
}
GATE = {"min_mean_R": 0.05, "min_t": 3.2, "min_year_share": 0.65}


def _cfgs(base, P):
    c = copy.deepcopy(base)
    c["session"]["overnight"] = P["overnight"]
    cs = copy.deepcopy(c)
    cs["execution"]["spread"]["stress_multiplier"] = 1.5
    for k in ("entry_slippage_pts", "exit_slippage_pts", "stop_slippage_pts"):
        cs["execution"][k] *= 2
    c0 = copy.deepcopy(c)
    c0["execution"]["commission_per_lot_side"] = 0
    for k in ("entry_slippage_pts", "exit_slippage_pts", "stop_slippage_pts"):
        c0["execution"][k] = 0
    c0["execution"]["spread"]["stress_multiplier"] = 0.0
    c0["data"]["swap_long_pts"] = c0["data"]["swap_short_pts"] = 0.0
    return c, cs, c0


def run(proto: str):
    t0 = time.time()
    P = PROTO[proto]
    base = load_config()
    m1 = load_m1(base)
    c, cs, c0 = _cfgs(base, P)
    mk, mk_s, mk_0 = build_market(c, m1), build_market(cs, m1), build_market(c0, m1)
    lib = S.compute(m1, P["scale"])
    D = S.to_decisions(lib, mk, P["stale"])
    tlo, thi = period_bounds(mk, base["splits"]["train"])
    clo, chi = period_bounds(mk, base["splits"]["calibration"])
    res = {"protocol": proto, "params": P, "gate": GATE, "library_version": lib["version"], "variants": {}}
    for name in S.names_for(P["scale"]):
        dirs, stop = D[name]["dir"], D[name]["stop_px"]
        for var, tp in (("tp4", 4.0), ("tp2", 2.0)):
            cc, ccs, cc0 = (copy.deepcopy(x) for x in (c, cs, c0))
            for x in (cc, ccs, cc0):
                x["trade"]["take_profit_r"] = tp
            runs = {}
            for tag, mkx, cx in (("net", mk, cc), ("stress", mk_s, ccs), ("gross", mk_0, cc0)):
                d_ok, sd = stop_distances(dirs, stop, D["_close"], mkx.dec_min_stop, P["max_stop"])
                tr, st, cnt = run_rule(mkx, cx, tlo, thi, d_ok, sd, P["seg"], P["max_hold"])
                runs[tag] = summary(tr, st, cnt, mkx, cx)
                if tag == "net":
                    runs[tag]["stop_pips_median"] = round(float(np.median(tr["stop_dist"]) * 1e4), 1) if len(tr) else None
                    runs[tag]["hold_days_median"] = (round(float(np.median(mk.dec_day[tr["exit_dec"]] - mk.dec_day[tr["entry_dec"]])), 1)
                                                     if len(tr) else None)
            net = runs["net"]
            passed = (net["n_trades"] >= P["min_trades"] and (net["mean_R"] or -9) >= GATE["min_mean_R"]
                      and net["t_stat"] >= GATE["min_t"] and net["pos_year_share"] >= GATE["min_year_share"]
                      and (runs["stress"]["mean_R"] or -9) > 0)
            rec = {"train": net, "train_stress": {k: runs["stress"][k] for k in ("n_trades", "mean_R", "t_stat", "profit_factor")},
                   "train_gross": {k: runs["gross"][k] for k in ("n_trades", "mean_R", "t_stat", "pos_year_share")},
                   "pass_train_gate": bool(passed)}
            if passed:
                d_ok, sd = stop_distances(dirs, stop, D["_close"], mk.dec_min_stop, P["max_stop"])
                trc, stc, cntc = run_rule(mk, cc, clo, chi, d_ok, sd, P["seg"], P["max_hold"])
                rec["calibration"] = summary(trc, stc, cntc, mk, cc)
                rec["pass_calibration"] = bool((rec["calibration"]["mean_R"] or -9) > 0)
            res["variants"][f"{name}/{var}"] = rec
            print(f"{proto} {name:20s} {var} n={net['n_trades']:5d} net={net['mean_R']} t={net['t_stat']} "
                  f"PF={net['profit_factor']} ans+={net['pos_year_share']} stress={runs['stress']['mean_R']} "
                  f"brut={runs['gross']['mean_R']} (t {runs['gross']['t_stat']}) stop={net.get('stop_pips_median')}p "
                  f"durée={net.get('hold_days_median')}j porte={'OUI' if passed else 'non'}"
                  + (f" calib={rec['calibration']['mean_R']}" if passed else ""), flush=True)
    res["elapsed_s"] = round(time.time() - t0)
    (OUT / f"stage_{proto.lower()}_strategies.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["B2", "B3"]:
        run(p)
