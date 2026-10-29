"""Backtest de stratégies intraday simples sur BTCUSD avec coûts MT5, sélection walk-forward honnête.

Exemples :
  python backtest.py --data data/btcusdt_1m.parquet --assume-spread-bps 2          # historique Binance, spread supposé
  python backtest.py --data mt5_export/BTCUSD_M1.csv                              # export de votre courtier (spread réel)
  python backtest.py --synthetic null|trend|meanrev                               # test de mécanique

Ce qui est rapporté comme performance = UNIQUEMENT la série walk-forward hors échantillon : à chaque
trimestre, on choisit la configuration au meilleur Sharpe NET sur le passé strict, et on la trade sur le
trimestre suivant (flat si le meilleur Sharpe passé est <= 0). Le tableau par configuration est biaisé
(in-sample) et n'est là que pour diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import common
from data import load_binance_parquet, load_mt5_csv, make_synthetic
from engine import Costs, run
from strategies import build_signals, complete_bars, default_configs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def daily_matrix(all_trades: dict, days: pd.DatetimeIndex) -> np.ndarray:
    D = np.zeros((len(all_trades), len(days)))
    for r, tr in enumerate(all_trades.values()):
        if len(tr):
            s = tr["net"].groupby(tr["exit_time"].dt.normalize()).sum().reindex(days, fill_value=0.0)
            D[r] = s.to_numpy()
    return D


def walk_forward(all_trades: dict, D: np.ndarray, days: pd.DatetimeIndex, a):
    names = list(all_trades)
    S1 = np.c_[np.zeros(len(D)), np.cumsum(D, axis=1)]
    S2 = np.c_[np.zeros(len(D)), np.cumsum(D ** 2, axis=1)]
    exits = {k: v["exit_time"].to_numpy() for k, v in all_trades.items()}
    ts = (days[0] + pd.DateOffset(months=a.min_train_months)).to_period("M").to_timestamp()
    picks, parts = [], []
    while ts < days[-1]:
        te = ts + pd.DateOffset(months=a.test_months)
        k = int((ts - days[0]).days)
        mean = S1[:, k] / k
        var = (S2[:, k] - S1[:, k] ** 2 / k) / (k - 1)
        score = np.where(var > 0, mean / np.sqrt(np.maximum(var, 1e-18)), -np.inf)
        ntr = np.array([np.searchsorted(exits[nm], np.datetime64(ts)) for nm in names])
        score = np.where(ntr >= a.min_trades, score, -np.inf)
        best = int(np.argmax(score))
        if np.isfinite(score[best]) and score[best] > 0:
            tr = all_trades[names[best]]
            parts.append(tr[(tr["exit_time"] >= ts) & (tr["exit_time"] < te)].assign(config=names[best]))
            picks.append((str(ts.date()), names[best], round(float(score[best] * np.sqrt(365)), 2)))
        else:
            picks.append((str(ts.date()), "FLAT", None))
        ts = te
    oos = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "gross", "net", "bars", "reason", "config"])
    return oos, picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", choices=["null", "trend", "meanrev"])
    ap.add_argument("--assume-spread-bps", type=float, default=2.0, help="spread supposé pour un Parquet Binance")
    ap.add_argument("--tz-mode", default="eet_us_dst", choices=["eet_us_dst", "fixed"])
    ap.add_argument("--tz-hours", type=int, default=0)
    ap.add_argument("--spread-mult", type=float, default=1.0)
    ap.add_argument("--commission-bps", type=float, default=0.0, help="par côté (0 si le courtier ne facture que le spread)")
    ap.add_argument("--slip-bps", type=float, default=0.5)
    ap.add_argument("--swap-long-bps-day", type=float, default=3.0)
    ap.add_argument("--swap-short-bps-day", type=float, default=3.0)
    ap.add_argument("--min-train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.synthetic:
        m1 = make_synthetic(a.synthetic, days=900)
        a.min_train_months = min(a.min_train_months, 8)
    elif a.data and a.data.endswith(".csv"):
        m1 = load_mt5_csv(a.data, a.tz_mode, a.tz_hours)
    elif a.data:
        m1 = load_binance_parquet(a.data, a.assume_spread_bps)
    else:
        ap.error("--data ou --synthetic requis")
    start, end = m1.index[0], m1.index[-1]
    print(f"{len(m1):,} bougies 1m | {start} -> {end} | spread médian {m1['spread_px'].median() / m1['close'].median() * 1e4:.2f} bps")

    costs = Costs(a.spread_mult, a.commission_bps, a.slip_bps, a.swap_long_bps_day, a.swap_short_bps_day)
    cfgs = default_configs()
    cache, all_trades, rows = {}, {}, []
    for cfg in cfgs:
        if cfg.tf not in cache:
            cache[cfg.tf] = complete_bars(m1, cfg.tf)
        bars = cache[cfg.tf]
        tgt, stop = build_signals(cfg, bars)
        tgt[:cfg.warmup] = 0
        tr = run(bars, tgt, stop, costs, cfg.trail, cfg.max_hold, start=cfg.warmup)
        all_trades[cfg.name] = tr
        s = common.summarize(tr, start, end)
        s.update(config=cfg.name,
                 long_pct=round(float(tr.loc[tr.side == 1, "net"].sum() * 100), 1) if len(tr) else 0.0,
                 short_pct=round(float(tr.loc[tr.side == -1, "net"].sum() * 100), 1) if len(tr) else 0.0)
        rows.append(s)
        print(f"  {cfg.name:<22} trades={s.get('n_trades', 0):>6}  net_bps={s.get('net_bps', float('nan')):>7}  sharpe={s.get('sharpe', float('nan')):>6}")
    table = pd.DataFrame(rows).set_index("config").sort_values("sharpe", ascending=False)

    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    D = daily_matrix(all_trades, days)
    oos, picks = walk_forward(all_trades, D, days, a)
    oos_start = pd.Timestamp(picks[0][0])
    d_oos = common.daily_returns(oos, oos_start, end) if len(oos) else np.zeros((end.normalize() - oos_start).days + 1)
    k0 = int((oos_start - days[0]).days)
    tr_sr = D[:, k0:].mean(axis=1) / np.maximum(D[:, k0:].std(axis=1, ddof=1), 1e-12)

    report = {"oos_start": str(oos_start.date()), "n_configs_tried": len(cfgs),
              "walk_forward": common.summarize(oos, oos_start, end),
              "walk_forward_cost_stress": {f"+{x}bps_rt": common.summarize(oos, oos_start, end, x) for x in (2, 5, 10)},
              "dsr": round(common.dsr(d_oos, tr_sr, len(cfgs)), 4) if len(oos) and d_oos.std() > 0 else None,
              "months_flat": sum(1 for p in picks if p[1] == "FLAT") * a.test_months,
              "selections": picks}
    if len(oos):
        yr = oos.groupby(oos["exit_time"].dt.year)["net"].sum() * 100
        report["net_pct_by_year"] = {int(k): round(float(v), 1) for k, v in yr.items()}
        report["net_pct_by_side"] = {"long": round(float(oos.loc[oos.side == 1, "net"].sum() * 100), 1),
                                     "short": round(float(oos.loc[oos.side == -1, "net"].sum() * 100), 1)}
        report["exit_reasons"] = oos["reason"].value_counts().to_dict()
    cl = m1["close"].resample("1D").last().dropna()
    bh = np.log(cl).diff().dropna()
    bh = bh[bh.index >= oos_start]
    report["buy_and_hold"] = {"return_pct": round(float((cl.iloc[-1] / cl[cl.index >= oos_start].iloc[0] - 1) * 100), 1),
                              "sharpe": round(common.sharpe(bh.to_numpy()), 2)}
    w = report["walk_forward"]
    ok = (w.get("n_trades", 0) >= 100 and w.get("sharpe", 0) > 1.0 and (report["dsr"] or 0) > 0.95
          and report["walk_forward_cost_stress"]["+5bps_rt"].get("net_bps", -1) > 0)
    report["verdict_heuristique"] = ("CANDIDAT démo : critères réunis (à confirmer en forward-test)" if ok
                                     else "AUCUN EDGE DÉMONTRÉ hors échantillon : ne pas déployer")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "configs_in_sample.csv")
    oos.to_csv(out / "oos_trades.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    if len(oos):
        eq = oos.sort_values("exit_time")
        plt.figure(figsize=(10, 4))
        plt.plot(eq["exit_time"], (eq["net"].cumsum() * 100).to_numpy(), label="walk-forward net")
        plt.plot(eq["exit_time"], (eq["gross"].cumsum() * 100).to_numpy(), label="brut", alpha=.5)
        plt.ylabel("PnL cumulé (% du notionnel, additif)"), plt.legend(), plt.grid(alpha=.3), plt.tight_layout()
        plt.savefig(out / "equity.png", dpi=130)
    print("\nConfigs (in-sample, biaisé) :")
    print(table[["n_trades", "trades_per_day", "gross_bps", "net_bps", "sharpe", "long_pct", "short_pct"]].to_string())
    print("\nRAPPORT WALK-FORWARD :")
    print(json.dumps({k: v for k, v in report.items() if k != "selections"}, indent=2, default=str))


if __name__ == "__main__":
    main()
