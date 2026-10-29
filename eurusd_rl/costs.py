"""Modèle de spread conservateur.

Constat (rapport qualité) : le spread de barre MT5 n'est pas homogène dans le temps
(2015-2019 : 7-8 pts ; 2020-2021 : 50 pts constants ; 2022+ : ~2 pts ; 2025-2026 : nombreux zéros) et,
en M1, il ne représente probablement pas le pire spread de la minute. On utilise donc :
  coût[i] = stress x max( profil_p75[quart d'heure serveur], observé[i] si date >= trust_from, plancher )
Le profil est estimé uniquement sur une fenêtre incluse dans l'entraînement (2022-2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 96   # quarts d'heure serveur


def qh_bin(t_server: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    t = pd.DatetimeIndex(t_server)
    return (t.hour * 4 + t.minute // 15).to_numpy()


def fit_profile(m1: pd.DataFrame, fit_range: list[str], q: float) -> np.ndarray:
    t = pd.DatetimeIndex(m1["t_utc"])
    a, b = pd.Timestamp(fit_range[0]), pd.Timestamp(fit_range[1]) + pd.Timedelta(days=1)
    sel = (t >= a) & (t < b)
    s = pd.Series(m1["spread_obs_pts"].to_numpy()[sel]).groupby(qh_bin(m1["t_server"][sel])).quantile(q)
    prof = s.reindex(range(N_BINS))
    if prof.isna().any():    # quart d'heure sans donnée (00:00-00:05 serveur) : valeur maximale du profil
        prof = prof.fillna(prof.max())
    return prof.to_numpy(float)


def spread_cost_pts(m1: pd.DataFrame, profile: np.ndarray, trust_from: str, floor_pts: float,
                    stress: float = 1.0) -> np.ndarray:
    prof = profile[qh_bin(m1["t_server"])]
    obs = m1["spread_obs_pts"].to_numpy(float)
    trusted = pd.DatetimeIndex(m1["t_utc"]) >= pd.Timestamp(trust_from)
    obs = np.where(trusted, obs, 0.0)
    return stress * np.maximum(np.maximum(prof, obs), floor_pts)
