"""Rapport d'un run : best vs latest, calibration, validation par blocs, stress de coûts, références.

Usage : python -m eurusd_rl.report --run runs/A_baseline_s1
Aucune donnée du test protégé n'est chargée (elle n'existe pas dans le fichier de données).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .data import ROOT, load_config, load_m1
from .evaluate import metrics, period_bounds, regimes_table, run_policy
from .market import build_market
from .model import ActorCritic
from .train import load_ckpt

KEYS = ["n_trades", "trades_per_day", "win_rate", "profit_factor", "mean_R", "total_R", "return_pct", "max_drawdown_pct",
        "worst_day_pct", "worst_5d_pct", "worst_10d_pct", "days_without_trade_share", "top5_days_share_of_net",
        "violations", "blocked_entries", "invalid_actions", "robust_score"]


class Uniform(torch.nn.Module):
    """Référence : action uniforme parmi les actions valides (échantillonnée)."""
    def forward(self, obs):
        z = torch.zeros(obs["mask"].shape)
        return z.masked_fill(~obs["mask"], -1e9), torch.zeros(len(z))


def load_policy(path: Path, cfg: dict) -> ActorCritic:
    P = cfg["ppo"]
    m = ActorCritic(P["timeframes"], P["hidden"], P["fuse"], wait_bias=P.get("wait_bias_init", 0.0))
    m.load_state_dict(load_ckpt(path)["model"])
    m.eval()
    return m


def evaluate_all(policy, mk, cfg, regimes, deterministic=True, fixed=None, seed=0):
    out = {}
    for name in ("calibration", "validation"):
        lo, hi = period_bounds(mk, cfg["splits"][name])
        tr, st, cnt = run_policy(policy, mk, cfg, lo, hi, "continuous", deterministic, seed, fixed)
        out[name] = metrics(tr, st, cnt, mk, cfg, regimes)
        if name == "validation":
            out["validation_blocks"] = []
            for b in cfg["splits"]["validation_blocks"]:
                lo, hi = period_bounds(mk, b)
                tr, st, cnt = run_policy(policy, mk, cfg, lo, hi, "continuous", deterministic, seed, fixed)
                mb = metrics(tr, st, cnt, mk, cfg)
                out["validation_blocks"].append({"block": b, **{k: mb.get(k) for k in KEYS}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--references", action="store_true")
    a = ap.parse_args()
    run = ROOT / a.run
    meta = json.loads((run / "config.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    m1 = load_m1(cfg)
    mk = build_market(cfg, m1)
    cfg_s = copy.deepcopy(cfg); cfg_s["execution"]["spread"]["stress_multiplier"] = 1.5
    cfg_s["execution"]["stop_slippage_pts"] *= 2; cfg_s["execution"]["entry_slippage_pts"] *= 2
    mk_s = build_market(cfg_s, m1)
    reg = regimes_table(m1)
    res = {}
    for ck in ("best_validation", "latest"):
        pol = load_policy(run / f"{ck}.pt", cfg)
        r = evaluate_all(pol, mk, cfg, reg)
        lo, hi = period_bounds(mk_s, cfg["splits"]["validation"])
        tr, st, cnt = run_policy(pol, mk_s, cfg_s, lo, hi)
        r["validation_cost_stress"] = {k: metrics(tr, st, cnt, mk_s, cfg_s).get(k) for k in KEYS}
        r["checkpoint_update"] = int(load_ckpt(run / f"{ck}.pt")["update"])
        res[ck] = r
    if a.references:
        res["ref_wait"] = evaluate_all(None, mk, cfg, reg, fixed=0)
        res["ref_uniform"] = evaluate_all(Uniform(), mk, cfg, reg, deterministic=False, seed=11)
    (run / "evaluation.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    for k, r in res.items():
        print(k, "| calib", {x: r["calibration"].get(x) for x in ("n_trades", "total_R", "max_drawdown_pct", "robust_score")},
              "| valid", {x: r["validation"].get(x) for x in ("n_trades", "total_R", "max_drawdown_pct", "robust_score")})


if __name__ == "__main__":
    main()
