"""Évaluation déterministe d'une politique sur une période + métriques obligatoires (section 12)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import env as E
from .market import Market, CTX_FEATS
from .model import to_torch

LOG_COLS = ["entry_dec", "exit_dec", "exit_m1", "side", "entry", "exit", "lots", "stop_dist", "reason",
            "balance", "net_usd", "R"]
REASONS = {1: "stop", 2: "tp", 3: "close", 4: "forced", 5: "violation"}


def period_bounds(mk: Market, rng: list[str]) -> tuple[int, int]:
    idx = np.flatnonzero(mk.subset_mask(rng))
    return int(idx[0]), int(idx[-1]) + 1


def segments(mk: Market, lo: int, hi: int, mode: str) -> list[tuple[int, int]]:
    if mode == "continuous":
        return [(lo, hi)]
    t = pd.DatetimeIndex(mk.dec_t_utc[lo:hi]).to_period("M").asi8
    cut = lo + np.flatnonzero(np.r_[True, t[1:] != t[:-1]])
    return list(zip(cut, np.r_[cut[1:], hi]))


@torch.no_grad()
def run_policy(policy, mk: Market, cfg: dict, lo: int, hi: int, mode: str = "continuous",
               deterministic: bool = True, seed: int = 0, fixed_action: int | None = None):
    """Rejoue la politique chronologiquement. Retourne (trades, steps, compteurs)."""
    segs = segments(mk, lo, hi, mode)
    venvs = [E.VecEnv(mk, cfg, 1, a, b, sequential=True, log_capacity=20000, seed=seed) for a, b in segs]
    g = torch.Generator().manual_seed(seed)
    rows = []
    active = list(range(len(venvs)))
    while active:
        obs = [venvs[i].obs() for i in active]
        batch = {k: np.concatenate([o[k] for o in obs]) for k in obs[0]}
        if fixed_action is not None:
            acts = np.where(batch["mask"][:, fixed_action], fixed_action, 0)
        else:
            logits, _ = policy(to_torch(batch))
            if deterministic:
                acts = logits.argmax(-1).numpy()
            else:
                acts = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(-1).numpy()
        nxt = []
        for n, i in enumerate(active):
            v = venvs[i]
            d = int(v.I[0, E.I_PTR])
            r, done, viol, info = v.step(acts[n:n + 1])
            rows.append((d, i, int(acts[n]), float(info["equity"][0]), int(viol[0])))
            if not done[0] and v.I[0, E.I_PTR] < v.I[0, E.I_END]:
                nxt.append(i)
        active = nxt
    trades = []
    for i, v in enumerate(venvs):
        L = pd.DataFrame(v.log[0, :v.nlog[0]], columns=LOG_COLS)
        L["segment"] = i
        trades.append(L)
    tr = pd.concat(trades, ignore_index=True)
    for c in ("entry_dec", "exit_dec", "exit_m1", "side", "reason", "segment"):
        tr[c] = tr[c].astype(np.int64)
    tr["entry_time_utc"] = pd.DatetimeIndex(mk.dec_t_utc[tr["entry_dec"].to_numpy()])
    tr["exit_time_server"] = pd.to_datetime(mk.t_server_min[tr["exit_m1"].to_numpy()] * 60, unit="s")
    tr["reason"] = tr["reason"].map(REASONS)
    st = pd.DataFrame(rows, columns=["dec", "segment", "action", "equity", "viol"]).sort_values(["segment", "dec"])
    st["day"] = mk.dec_day[st["dec"].to_numpy()]
    cnt = {"blocked_entries": int(sum(v.I[0, E.I_NBLOCK] for v in venvs)),
           "invalid_actions": int(sum(v.I[0, E.I_NINVALID] for v in venvs)),
           "violations": int(sum(v.I[0, E.I_VIOL] for v in venvs)),
           "n_segments": len(venvs), "mode": mode}
    return tr, st, cnt


# ---------------------------------------------------------------------------------------------- métriques
def _daily(st: pd.DataFrame, init: float) -> pd.DataFrame:
    """PnL quotidien (en $) par segment ; l'équité de fin de jour est l'équité de la dernière décision du jour."""
    out = []
    for _, s in st.groupby("segment"):
        g = s.groupby("day")
        eq_end = g["equity"].last()
        eq_min = g["equity"].min()
        start = eq_end.shift(1).fillna(init)
        # après une violation le compte est réinitialisé : la base du jour suivant reste l'équité de fin de jour
        out.append(pd.DataFrame({"pnl": eq_end - start, "min_dd": eq_min - start, "eq": eq_end}))
    d = pd.concat(out)
    d.index = pd.to_datetime(d.index.to_numpy() * 86400, unit="s")
    return d.sort_index()


def challenge_mc(daily_pnl_pct: np.ndarray, daily_min_pct: np.ndarray, target=0.10, dloss=0.05, tloss=0.10,
                 horizon=60, block=5, n=20000, seed=7) -> dict:
    if len(daily_pnl_pct) < block * 2:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(horizon / block))
    st = rng.integers(0, len(daily_pnl_pct) - block, size=(n, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :horizon]
    p, m = daily_pnl_pct[idx], daily_min_pct[idx]
    cum = np.cumsum(p, 1)
    prev = np.c_[np.zeros(n), cum[:, :-1]]
    hit_t = cum >= target
    hit_l = (m <= -dloss) | (prev + m <= -tloss) | (cum <= -tloss)
    tt = np.where(hit_t.any(1), hit_t.argmax(1), horizon + 1)
    tl = np.where(hit_l.any(1), hit_l.argmax(1), horizon + 1)
    passed, failed = tt < tl, tl < tt
    return {"p_pass": round(float(passed.mean()), 4), "p_fail": round(float(failed.mean()), 4),
            "p_open": round(float(1 - passed.mean() - failed.mean()), 4),
            "median_days_to_pass": float(np.median(tt[passed]) + 1) if passed.any() else None,
            "horizon_days": horizon}


def metrics(tr: pd.DataFrame, st: pd.DataFrame, cnt: dict, mk: Market, cfg: dict, regimes: pd.DataFrame | None = None) -> dict:
    init = cfg["challenge"]["initial_capital"]
    R = cfg["internal_risk"]["risk_per_trade"] * init
    d = _daily(st, init)
    eq = st["equity"].to_numpy()
    # drawdown sur l'équité de chaque décision, par segment (compte réinitialisé entre segments)
    mdd = 0.0
    for _, s in st.groupby("segment"):
        e = s["equity"].to_numpy()
        peak = np.maximum.accumulate(np.r_[init, e])[1:]
        mdd = min(mdd, float((e - peak).min()))
    n = len(tr)
    m = {"n_days": int(len(d)), "pnl_usd": round(float(tr["net_usd"].sum()), 2) if n else 0.0}
    m["return_pct"] = round(m["pnl_usd"] / init * 100, 2)
    m["n_trades"] = n
    m["trades_per_day"] = round(n / max(len(d), 1), 3)
    if n:
        w, lsr = tr[tr["net_usd"] > 0], tr[tr["net_usd"] <= 0]
        m.update(win_rate=round(len(w) / n, 4), avg_win_usd=round(float(w["net_usd"].mean()), 2) if len(w) else 0.0,
                 avg_loss_usd=round(float(lsr["net_usd"].mean()), 2) if len(lsr) else 0.0,
                 profit_factor=round(float(w["net_usd"].sum() / -lsr["net_usd"].sum()), 3) if lsr["net_usd"].sum() < 0 else None,
                 expectancy_usd=round(float(tr["net_usd"].mean()), 2), mean_R=round(float(tr["R"].mean()), 4),
                 total_R=round(float(tr["R"].sum()), 2))
        lose = (tr["net_usd"] <= 0).to_numpy().astype(int)
        streak = 0
        cur = 0
        for x in lose:
            cur = cur + 1 if x else 0
            streak = max(streak, cur)
        m["worst_losing_streak_trades"] = int(streak)
        m["buy_sell"] = {s: {"n": int((tr["side"] == v).sum()), "pnl_usd": round(float(tr.loc[tr["side"] == v, "net_usd"].sum()), 2)}
                         for s, v in (("BUY", 1), ("SELL", -1))}
        ny = tr["entry_time_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        hh = ny.dt.hour
        sess = np.select([hh < 8, hh < 12], ["londres_02-08NY", "chevauchement_08-12NY"], "ny_12-16NY")
        m["by_session"] = tr.groupby(sess)["net_usd"].agg(["count", "sum"]).round(2).to_dict(orient="index")
        m["by_quarter"] = tr.groupby(tr["entry_time_utc"].dt.to_period("Q").astype(str))["net_usd"].sum().round(2).to_dict()
        m["exit_reasons"] = tr["reason"].value_counts().to_dict()
        if regimes is not None:
            day = tr["entry_time_utc"].dt.normalize()
            rg = regimes.reindex(day.values)
            for col in rg.columns:
                m[f"by_{col}"] = tr.groupby(rg[col].to_numpy())["net_usd"].agg(["count", "sum"]).round(2).to_dict(orient="index")
    m["max_drawdown_pct"] = round(mdd / init * 100, 2)
    m["worst_day_pct"] = round(float(d["pnl"].min() / init * 100), 2) if len(d) else 0.0
    m["worst_intraday_pct"] = round(float(d["min_dd"].min() / init * 100), 2) if len(d) else 0.0
    m["worst_5d_pct"] = round(float(d["pnl"].rolling(5).sum().min() / init * 100), 2) if len(d) >= 5 else None
    m["worst_10d_pct"] = round(float(d["pnl"].rolling(10).sum().min() / init * 100), 2) if len(d) >= 10 else None
    trade_days = set(tr["exit_time_server"].dt.normalize()) if n else set()
    m["profitable_days_share"] = round(float((d["pnl"] > 0).mean()), 4) if len(d) else 0.0
    m["days_without_trade_share"] = round(1 - len(trade_days) / max(len(d), 1), 4)
    pos = d["pnl"].clip(lower=0)
    net = d["pnl"].sum()
    m["top5_days_share_of_net"] = round(float(d["pnl"].nlargest(5).sum() / net), 3) if net > 0 else None
    m["best_day_share_of_gross_gains"] = round(float(pos.max() / pos.sum()), 3) if pos.sum() > 0 else None
    cum = d["pnl"].cumsum() / init
    hit = np.flatnonzero(cum.to_numpy() >= cfg["challenge"]["profit_target_phase1"])
    m["day_target_reached"] = int(hit[0] + 1) if len(hit) else None
    m.update(cnt)
    m["challenge_mc"] = challenge_mc((d["pnl"] / init).to_numpy(), (d["min_dd"] / init).to_numpy(),
                                     target=cfg["challenge"]["profit_target_phase1"], dloss=cfg["challenge"]["max_daily_loss"],
                                     tloss=cfg["challenge"]["max_total_loss"])
    tR = m.get("total_R", 0.0)
    m["robust_score"] = round(tR / max(-mdd / R, 4.0), 3)
    return m


def regimes_table(m1: pd.DataFrame) -> pd.DataFrame:
    """Régimes causaux connus au début de chaque jour UTC : tendance (rendement 20 j) et volatilité (rang ATR 20 j sur 1 an)."""
    d = m1.set_index("t_utc")["close"].resample("1D").last().dropna()
    hi = m1.set_index("t_utc")["high"].resample("1D").max().reindex(d.index)
    lo = m1.set_index("t_utc")["low"].resample("1D").min().reindex(d.index)
    ret20 = d / d.shift(20) - 1
    rng20 = ((hi - lo) / d).rolling(20).mean()
    vr = rng20.rolling(250, min_periods=60).rank(pct=True)
    trend = np.select([ret20 > 0.015, ret20 < -0.015], ["haussier", "baissier"], "range")
    vol = np.select([vr > 2 / 3, vr < 1 / 3], ["vol_forte", "vol_faible"], "vol_moyenne")
    t = pd.DataFrame({"trend": trend, "vol": vol}, index=d.index).shift(1)    # connu à l'ouverture du jour suivant
    return t.dropna()
