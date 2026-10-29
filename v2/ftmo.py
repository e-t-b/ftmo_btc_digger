"""Simulateur des règles FTMO 2-Step et Monte Carlo du challenge (protocole §6-7).

Entrée : journal quotidien du portefeuille hors échantillon, en R au risque de base (0,5 % du solde initial) :
R = résultat de la journée, worst_R = pire equity intrajournalière (réalisé + flottant) depuis le début du jour FTMO.
Chaque journée est rejouée avec le risque du jour (0,5 % ou 0,25 % selon l'equity), ce qui suppose que
résultat et pire flottant sont proportionnels au risque engagé. Le stop interne à -2 % (= -4 R au risque de base)
est déjà inclus dans les journées ; à risque réduit il coupe donc plus tôt que nécessaire (prudent)."""
from __future__ import annotations

import numpy as np
from numba import njit

from . import config as C


@njit(cache=True)
def _phase_nb(days_R, days_worst, days_n, target, rng_idx, horizon, acc, halt_below, reduce_scale, reduce_below,
              base, daily_loss, max_loss, min_days):
    eq = acc
    tdays = 0
    reached = False
    for step in range(min(horizon, rng_idx.shape[0])):
        d = rng_idx[step]
        start = eq
        if eq < halt_below * acc:
            return -1, step
        if reached:
            scale = 0.0
        elif eq < reduce_below * acc:
            scale = reduce_scale
        else:
            scale = 1.0
        unit = base * acc * scale
        worst = start + unit * days_worst[d]
        if start - worst >= daily_loss * acc - 1e-9 or worst <= (1 - max_loss) * acc + 1e-9:
            return -1, step + 1
        eq = start + unit * days_R[d]
        if days_n[d] > 0:
            tdays += 1
        if eq >= acc * (1 + target) - 1e-9:
            reached = True
        if reached and tdays >= min_days:
            return 1, step + 1
    return 0, horizon


def _phase(days_R, days_worst, days_n, target, rng_idx, horizon):
    """Rejoue une phase. Renvoie (statut, nb de jours) ; statut 1 = réussie, -1 = échouée, 0 = non conclue."""
    return _phase_nb(np.asarray(days_R, float), np.asarray(days_worst, float), np.asarray(days_n, np.int64),
                     float(target), np.asarray(rng_idx, np.int64), int(horizon), C.ACCOUNT, C.RISK["halt_below"],
                     C.RISK["reduced"] / C.RISK["base"], C.RISK["reduce_below"], C.RISK["base"],
                     C.FTMO["daily_loss"], C.FTMO["max_loss"], C.FTMO["min_days"])


def monte_carlo(day_R: np.ndarray, day_worst: np.ndarray, day_n: np.ndarray, n_paths: int = 20_000,
                horizon: int = 250, block: int = 5, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    nd = len(day_R)
    nb = int(np.ceil(horizon / block))
    res = np.zeros(n_paths, np.int64)
    d1 = np.zeros(n_paths)
    d2 = np.zeros(n_paths)
    for p in range(n_paths):
        starts = rng.integers(0, nd - block + 1, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
        s1, n1 = _phase(day_R, day_worst, day_n, C.FTMO["target1"], idx, horizon)
        if s1 != 1:
            res[p] = s1 * 1 if s1 == -1 else 0
            d1[p] = n1
            continue
        starts = rng.integers(0, nd - block + 1, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()
        s2, n2 = _phase(day_R, day_worst, day_n, C.FTMO["target2"], idx, horizon)
        d1[p], d2[p] = n1, n2
        res[p] = 2 if s2 == 1 else (-2 if s2 == -1 else 0)
    ok = res == 2
    return dict(p_pass=float(ok.mean()), p_fail_phase1=float((res == -1).mean()), p_fail_phase2=float((res == -2).mean()),
                p_unresolved=float((res == 0).mean()),
                median_days_phase1=float(np.median(d1[ok])) if ok.any() else float("nan"),
                median_days_total=float(np.median(d1[ok] + d2[ok])) if ok.any() else float("nan"))
