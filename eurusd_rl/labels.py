"""Étiquettes « triple barrière » alignées sur l'environnement d'exécution.

Pour une décision d et un sens (achat / vente), on simule EXACTEMENT le trade que l'environnement prendrait :
entrée à l'ouverture de la barre M1 de remplissage (ask + glissement à l'achat, bid - glissement à la vente),
stop à `sd` (distance de stop de la décision), cible à `tp_r x sd`, stop prioritaire si les deux sont touchés
dans la même barre, gaps à l'ouverture (stop : sortie à l'ouverture - glissement ; cible : sortie à la cible),
sinon clôture forcée à l'ouverture de la barre de clôture du jour. Coûts : spread de coût, glissements,
commission aller-retour. Résultat en R (multiple de la distance de stop), arrondi de lot ignoré.
"""
from __future__ import annotations

import numba as nb
import numpy as np

REASON_STOP, REASON_TP, REASON_FORCE = 1, 2, 4


@nb.njit(cache=True)
def label_one(o, h, l, sp, j, jf, side, sd, tp_r, slip_in, slip_out, slip_stop, comm_px):
    if side == 1:
        entry = o[j] + sp[j] + slip_in
    else:
        entry = o[j] - slip_in
    stop = entry - side * sd
    tp = entry + side * tp_r * sd
    exit_px = np.nan
    reason = 0
    k = j
    for k in range(j, jf):
        if side == 1:
            if k != j and o[k] <= stop:
                exit_px = o[k] - slip_stop; reason = REASON_STOP; break
            if k != j and o[k] >= tp:
                exit_px = tp; reason = REASON_TP; break
            if l[k] <= stop:
                exit_px = stop - slip_stop; reason = REASON_STOP; break
            if h[k] >= tp:
                exit_px = tp; reason = REASON_TP; break
        else:
            ao, ah, al = o[k] + sp[k], h[k] + sp[k], l[k] + sp[k]
            if k != j and ao >= stop:
                exit_px = ao + slip_stop; reason = REASON_STOP; break
            if k != j and ao <= tp:
                exit_px = tp; reason = REASON_TP; break
            if ah >= stop:
                exit_px = stop + slip_stop; reason = REASON_STOP; break
            if al <= tp:
                exit_px = tp; reason = REASON_TP; break
    if reason == 0:
        k = jf
        exit_px = o[jf] - slip_out if side == 1 else o[jf] + sp[jf] + slip_out
        reason = REASON_FORCE
    r = (side * (exit_px - entry) - comm_px) / sd
    return r, reason, k


@nb.njit(cache=True)
def label_many(o, h, l, sp, js, jfs, sds, side, tp_r, slip_in, slip_out, slip_stop, comm_px):
    n = len(js)
    R = np.empty(n); why = np.empty(n, np.int64); kk = np.empty(n, np.int64)
    for i in range(n):
        R[i], why[i], kk[i] = label_one(o, h, l, sp, js[i], jfs[i], side, sds[i], tp_r, slip_in, slip_out, slip_stop, comm_px)
    return R, why, kk


def make_labels(mk, cfg: dict, dec_idx: np.ndarray, tp_r: float) -> dict:
    ex, pt = cfg["execution"], cfg["data"]["point"]
    comm_px = 2 * ex["commission_per_lot_side"] / cfg["data"]["contract_size"]
    lat = int(ex["latency_bars_m1"])
    j = np.minimum(mk.dec_m1[dec_idx] + lat, mk.dec_force_m1[dec_idx])
    args = (mk.o, mk.h, mk.l, mk.spread, j, mk.dec_force_m1[dec_idx], mk.dec_stop_dist[dec_idx])
    common = (tp_r, ex["entry_slippage_pts"] * pt, ex["exit_slippage_pts"] * pt, ex["stop_slippage_pts"] * pt, comm_px)
    rl, wl, kl = label_many(*args, 1, *common)
    rs, ws, ks = label_many(*args, -1, *common)
    return {"R_long": rl, "R_short": rs, "why_long": wl, "why_short": ws, "exit_m1_long": kl, "exit_m1_short": ks}
