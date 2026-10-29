"""Entraînement PPO (étape A) avec checkpoints sûrs, reprise exacte et sélection sur la calibration.

Usage : python -m eurusd_rl.train --run runs/A_baseline_s1 --seed 1 --updates 400
        python -m eurusd_rl.train --run runs/A_baseline_s1 --resume      (reprend latest.pt)
Le test protégé n'existe pas dans les données chargées ; la validation n'est utilisée que par report.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fnn
import yaml

from . import env as E
from .data import ROOT, load_config, load_m1
from .evaluate import metrics, period_bounds, run_policy
from .market import build_market
from .model import ActorCritic, to_torch

LIB_VERSION = "none (étape A)"


def set_determinism(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def _rng_state(rng: np.random.Generator) -> dict:
    s = rng.bit_generator.state
    return {"bit_generator": s["bit_generator"], "state": int(s["state"]["state"]), "inc": int(s["state"]["inc"]),
            "has_uint32": int(s["has_uint32"]), "uinteger": int(s["uinteger"])}


def _set_rng_state(rng: np.random.Generator, d: dict):
    rng.bit_generator.state = {"bit_generator": d["bit_generator"], "state": {"state": d["state"], "inc": d["inc"]},
                               "has_uint32": d["has_uint32"], "uinteger": d["uinteger"]}


def save_ckpt(path: Path, model, opt, update: int, venv: E.VecEnv, extra: dict):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "update": update,
                "torch_rng": torch.get_rng_state(), "env_rng": _rng_state(venv.rng),
                "env_F": torch.from_numpy(venv.F.copy()), "env_I": torch.from_numpy(venv.I.copy()),
                "extra": extra}, tmp)
    tmp.replace(path)


def load_ckpt(path: Path) -> dict:
    # chargement restreint : aucun code arbitraire désérialisé
    return torch.load(path, map_location="cpu", weights_only=True)


class Buffer:
    def __init__(self, T, N):
        self.T, self.N = T, N
        self.obs = []
        self.act = np.zeros((T, N), np.int64); self.logp = np.zeros((T, N), np.float32)
        self.val = np.zeros((T, N), np.float32); self.rew = np.zeros((T, N), np.float32)
        self.done = np.zeros((T, N), np.float32)


def gae(buf: Buffer, last_val: np.ndarray, gamma: float, lam: float):
    adv = np.zeros_like(buf.rew)
    last = 0.0
    for t in reversed(range(buf.T)):
        nv = last_val if t == buf.T - 1 else buf.val[t + 1]
        nonterm = 1.0 - buf.done[t]
        delta = buf.rew[t] + gamma * nv * nonterm - buf.val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + buf.val


def train(cfg: dict, run: Path, seed: int, updates: int, resume: bool = False, mk=None, log=print) -> dict:
    run.mkdir(parents=True, exist_ok=True)
    P = cfg["ppo"]
    set_determinism(seed)
    if mk is None:
        mk = build_market(cfg, load_m1(cfg))
    lo, hi = period_bounds(mk, cfg["splits"]["train"])
    clo, chi = period_bounds(mk, cfg["splits"]["calibration"])
    venv = E.VecEnv(mk, cfg, P["n_env"], lo, hi, episode_days=P["episode_days"], seed=seed)
    model = ActorCritic(P["timeframes"], P["hidden"], P["fuse"], wait_bias=P.get("wait_bias_init", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=P["lr"], eps=1e-5)
    start, best = 0, -np.inf
    hist_path = run / "validation_history.jsonl"
    if resume:
        ck = load_ckpt(run / "latest.pt")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optim"])
        torch.set_rng_state(ck["torch_rng"]); _set_rng_state(venv.rng, ck["env_rng"])
        venv.F[:] = ck["env_F"].numpy(); venv.I[:] = ck["env_I"].numpy()
        start, best = int(ck["update"]), float(ck["extra"]["best_score"])
    else:
        meta = {"seed": seed, "config": cfg, "data_sha256": cfg["data"]["sha256"], "strategy_library": LIB_VERSION,
                "n_params": sum(p.numel() for p in model.parameters()),
                "train_decisions": [int(lo), int(hi)], "calibration_decisions": [int(clo), int(chi)],
                "selection": "best_validation.pt = meilleur score robuste sur la période de CALIBRATION"}
        (run / "config.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        hist_path.write_text("", encoding="utf-8")
    T, N = P["n_steps"], P["n_env"]
    progress = []
    t0 = time.time()
    for u in range(start, updates):
        buf = Buffer(T, N)
        ep_ret, act_count = [], np.zeros(4)
        model.eval()
        with torch.no_grad():
            for t in range(T):
                o = venv.obs()
                ot = to_torch(o)
                logits, v = model(ot)
                a = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)
                buf.obs.append(o)
                buf.act[t] = a.numpy(); buf.val[t] = v.numpy()
                buf.logp[t] = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1).numpy()
                r, d, viol, _ = venv.step(buf.act[t])
                buf.rew[t] = r; buf.done[t] = d
                np.add.at(act_count, buf.act[t], 1)
            last_v = model(to_torch(venv.obs()))[1].numpy()
        adv, ret = gae(buf, last_v, P["gamma"], P["gae_lambda"])
        # aplatissement
        keys = buf.obs[0].keys()
        flat = {k: np.concatenate([o[k] for o in buf.obs]) for k in keys}
        A = buf.act.reshape(-1); LP = buf.logp.reshape(-1); ADV = adv.reshape(-1); RET = ret.reshape(-1)
        # seuls les pas avec un vrai choix (>= 2 actions valides) entrent dans la perte de politique
        CHOICE = flat["mask"].sum(1) >= 2 if P.get("policy_loss_on_choices_only", False) else np.ones(len(A), bool)
        ADV = (ADV - ADV[CHOICE].mean()) / (ADV[CHOICE].std() + 1e-8)
        model.train()
        n = len(A)
        g = np.random.default_rng(seed * 100003 + u)
        stats = []
        for _ in range(P["epochs"]):
            perm = g.permutation(n)
            for i in range(0, n, P["minibatch"]):
                idx = perm[i:i + P["minibatch"]]
                ob = to_torch({k: v[idx] for k, v in flat.items()})
                logits, v = model(ob)
                lp_all = torch.log_softmax(logits, -1)
                lp = lp_all.gather(1, torch.from_numpy(A[idx])[:, None]).squeeze(1)
                ratio = torch.exp(lp - torch.from_numpy(LP[idx]))
                adv_t = torch.from_numpy(ADV[idx]).float()
                ch = torch.from_numpy(CHOICE[idx])
                surr = torch.min(ratio * adv_t, torch.clamp(ratio, 1 - P["clip"], 1 + P["clip"]) * adv_t)
                pl = -(surr * ch).sum() / ch.sum().clamp(min=1)
                vl = Fnn.mse_loss(v, torch.from_numpy(RET[idx]).float())
                p_all = lp_all.exp()
                ent = (-(p_all * torch.where(p_all > 0, lp_all, torch.zeros_like(lp_all))).sum(-1) * ch).sum() / ch.sum().clamp(min=1)
                loss = pl + P["value_coef"] * vl - P["entropy_coef"] * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), P["max_grad_norm"])
                opt.step()
                with torch.no_grad():
                    kl = (((ratio - 1) - torch.log(ratio)) * ch).sum().item() / max(ch.sum().item(), 1)
                    cf = (((ratio - 1).abs() > P["clip"]).float() * ch).sum().item() / max(ch.sum().item(), 1)
                stats.append((pl.item(), vl.item(), ent.item(), kl, cf))
        s = np.mean(stats, 0)
        rec = {"update": u + 1, "steps": (u + 1) * T * N, "reward_per_step": float(buf.rew.mean()),
               "reward_sum_R": float(buf.rew.sum()), "policy_loss": s[0], "value_loss": s[1], "entropy": s[2],
               "approx_kl": s[3], "clip_frac": s[4], "actions_share": (act_count / act_count.sum()).round(4).tolist(),
               "choice_share": float(CHOICE.mean()), "elapsed_s": round(time.time() - t0, 1)}
        progress.append(rec)
        with open(run / "progress.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if (u + 1) % P["eval_every"] == 0 or u + 1 == updates:
            model.eval()
            tr, st, cnt = run_policy(model, mk, cfg, clo, chi, mode="monthly")
            m = metrics(tr, st, cnt, mk, cfg)
            ev = {"update": u + 1, "period": "calibration", **{k: m.get(k) for k in (
                "n_trades", "total_R", "mean_R", "win_rate", "profit_factor", "max_drawdown_pct", "violations",
                "trades_per_day", "robust_score", "days_without_trade_share")}}
            with open(hist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
            log(f"[maj {u + 1}] calib : trades {ev['n_trades']} | R total {ev['total_R']} | score {ev['robust_score']} | "
                f"actions {rec['actions_share']} | ent {s[2]:.3f}")
            if (m.get("n_trades", 0) or 0) > 0 and m["robust_score"] > best:
                best = m["robust_score"]
                save_ckpt(run / "best_validation.pt", model, opt, u + 1, venv, {"best_score": best, "calib": ev})
        if (u + 1) % P["checkpoint_every"] == 0 or u + 1 == updates:
            save_ckpt(run / "latest.pt", model, opt, u + 1, venv, {"best_score": best})
    return {"best_score": best, "updates": updates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    train(cfg, ROOT / a.run, a.seed, a.updates, a.resume)


if __name__ == "__main__":
    main()
