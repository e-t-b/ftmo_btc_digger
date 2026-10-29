"""Simulation Monte Carlo du challenge FTMO 2-Step (Phase 1 : +10 %) à partir de trades hors échantillon.

Règles simulées (à confirmer dans votre espace FTMO) : objectif +10 %, perte totale maximale 10 % (statique, base = capital
initial), perte journalière maximale 5 %. Les rendements par trade (sur notionnel) sont multipliés par un levier L
(notionnel = L × équité). Bootstrap par blocs de jours pour conserver un peu d'autocorrélation.
Limites : l'historique hors échantillon est court ; la perte journalière est mesurée sur la somme des trades du jour
(la vraie règle regarde aussi l'équité flottante intrajournalière, donc plus stricte).

Usage : python ftmo_sim.py results_ml/trades_EURUSD_k2_H16_thr0.1.csv 2024-02-01 2026-10-29"""
import sys

import numpy as np
import pandas as pd

import common


def challenge_mc(daily, leverage, n=20000, horizon=250, block=5, target=0.10, maxloss=0.10, dloss=0.05, seed=3):
    rng = np.random.default_rng(seed)
    d = daily * leverage
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(d) - block, size=(n, nb))
    paths = np.stack([np.concatenate([d[s:s + block] for s in row]) for row in starts])[:, :horizon]
    eq = np.cumsum(paths, axis=1)
    hit_t = eq >= target
    hit_l = (eq <= -maxloss) | (paths <= -dloss)
    t_t = np.where(hit_t.any(1), hit_t.argmax(1), horizon + 1)
    t_l = np.where(hit_l.any(1), hit_l.argmax(1), horizon + 1)
    passed, failed = t_t < t_l, t_l < t_t
    return float(passed.mean()), float(failed.mean()), (float(np.median(t_t[passed])) if passed.any() else float("nan"))


if __name__ == "__main__":
    tr = pd.read_csv(sys.argv[1], parse_dates=["entry_time", "exit_time"])
    d = common.daily_returns(tr, sys.argv[2], sys.argv[3])
    print(f"{len(tr)} trades | {len(d)} jours | Sharpe annualisé {common.sharpe(d):.2f} | rendement moyen/jour {d.mean() * 1e4:.2f} bps (notionnel)")
    print("levier | P(réussite) | P(échec) | jours médians")
    for L in (5, 10, 20, 30):
        p, f, m = challenge_mc(d, L)
        print(f"{L:>5}x | {p * 100:>9.1f}% | {f * 100:>6.1f}% | {m:>6.0f}")
