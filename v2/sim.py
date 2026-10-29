"""Simulation des trades d'un instrument pour une configuration (protocole §3).

Une minute M1 = une barre BID (o, h, l, c) + coût de spread sc (ask = bid + sc).
Sortie : un trade au plus par session, avec son R net (coûts inclus) et sa trajectoire minute par minute
du pire R latent (low BID pour un long, high ASK pour un short), utilisée par le portefeuille."""
from __future__ import annotations

import numpy as np
from numba import njit

from . import config as C

REASONS = np.array(["stop", "gap", "trail", "tp", "time", "eod"])
STOP, GAP, TRAIL, TP, TIME, EOD = 0, 1, 2, 3, 4, 5


@njit(cache=True)
def _simulate(t, o, h, l, c, sc, atr15, comm, slip, sslip,
              s_open, s_close, s_iopen, s_trend, s_atr,
              N, lo, hi, kk, tp_r, entry_window, min_cost_mult, coverage):
    nS = s_open.shape[0]
    n = t.shape[0]
    # sorties
    T_sess = np.empty(nS, np.int64)
    T_dir = np.empty(nS, np.int64)
    T_ent_t = np.empty(nS, np.int64)
    T_ex_t = np.empty(nS, np.int64)
    T_ent = np.empty(nS)
    T_R = np.empty(nS)
    T_rnet = np.empty(nS)
    T_rgross = np.empty(nS)
    T_why = np.empty(nS, np.int64)
    T_cost_r = np.empty(nS)
    T_p0 = np.empty(nS, np.int64)
    T_pn = np.empty(nS, np.int64)
    cap = nS * 1500 + 1024
    P_t = np.empty(cap, np.int64)
    P_u = np.empty(cap)
    nt = 0
    npth = 0
    for s in range(nS):
        op = s_open[s]
        cl = s_close[s]
        tr = s_trend[s]
        if tr == 0.0 or not (s_atr[s] > 0.0):
            continue
        i0 = s_iopen[s]
        or_end = op + N * 60
        # range d'ouverture
        H = -1e300
        L = 1e300
        cnt = 0
        k = i0
        while k < n and t[k] < or_end:
            if h[k] > H:
                H = h[k]
            if l[k] < L:
                L = l[k]
            cnt += 1
            k += 1
        if cnt < coverage * N or not (H > L):
            continue
        ratio = (H - L) / s_atr[s]
        if ratio < lo or ratio > hi:
            continue
        # recherche du signal : clôture M5 (alignée sur l'ouverture) au-delà de l'OR dans le sens de la tendance
        last_sig = or_end + entry_window * 60
        sig_k = -1
        while k < n - 1 and t[k] + 60 <= last_sig and t[k] + 60 < cl:
            if (t[k] + 60 - op) % 300 == 0:
                if tr > 0 and c[k] > H:
                    sig_k = k
                    break
                if tr < 0 and c[k] < L:
                    sig_k = k
                    break
            k += 1
        if sig_k < 0:
            continue
        j = sig_k + 1
        if j >= n or t[j] - t[sig_k] > 300 or t[j] >= cl:
            continue
        d = 1 if tr > 0 else -1
        if d == 1:
            entry = o[j] + sc[j] + slip[j]
            stop = L
            R = entry - stop
        else:
            entry = o[j] - slip[j]
            stop = H
            R = stop - entry
        crt = 2.0 * comm[j]
        est_cost = sc[j] + 2.0 * slip[j] + crt
        if R <= 0.0 or R < min_cost_mult * est_cost:
            continue
        tp = entry + d * tp_r * R if tp_r > 0 else 0.0
        hh = -1e300
        ll = 1e300
        ex = 0.0
        why = EOD
        kx = j
        p0 = npth
        done = False
        kq = j
        while kq < n:
            if t[kq] >= cl:
                # sortie temporelle à l'ouverture de la première minute >= heure de clôture
                if d == 1:
                    ex = o[kq] - slip[kq]
                else:
                    ex = o[kq] + sc[kq] + slip[kq]
                why = TIME
                kx = kq
                done = True
            else:
                bh = h[kq]
                bl = l[kq]
                ah = h[kq] + sc[kq]
                al = l[kq] + sc[kq]
                if d == 1:
                    if bl <= stop:
                        if kq > j and o[kq] <= stop:
                            ex = o[kq] - sslip[kq]
                            why = GAP
                        else:
                            ex = stop - sslip[kq]
                            why = STOP if stop <= L + 1e-12 else TRAIL
                        kx = kq
                        done = True
                    elif tp_r > 0 and bh >= tp:
                        ex = tp
                        why = TP
                        kx = kq
                        done = True
                else:
                    if ah >= stop:
                        if kq > j and o[kq] + sc[kq] >= stop:
                            ex = o[kq] + sc[kq] + sslip[kq]
                            why = GAP
                        else:
                            ex = stop + sslip[kq]
                            why = STOP if stop >= H - 1e-12 else TRAIL
                        kx = kq
                        done = True
                    elif tp_r > 0 and al <= tp:
                        ex = tp
                        why = TP
                        kx = kq
                        done = True
            # trajectoire du pire R latent (coûts aller-retour inclus)
            if npth >= cap:
                break
            if done:
                P_t[npth] = t[kq]
                P_u[npth] = (d * (ex - entry) - crt) / R
                npth += 1
                break
            if d == 1:
                P_u[npth] = (l[kq] - entry - crt) / R
            else:
                P_u[npth] = (entry - (h[kq] + sc[kq]) - crt) / R
            P_t[npth] = t[kq]
            npth += 1
            # mise à jour du trailing à la clôture M5 (alignée sur l'ouverture de session)
            if h[kq] > hh:
                hh = h[kq]
            if l[kq] < ll:
                ll = l[kq]
            if tp_r <= 0 and (t[kq] + 60 - op) % 300 == 0 and atr15[kq] > 0:
                if d == 1:
                    ns = hh - kk * atr15[kq]
                    if ns > stop:
                        stop = ns
                else:
                    ns = ll + kk * atr15[kq]
                    if ns < stop:
                        stop = ns
            kq += 1
        if not done:
            # fin des données : sortie à la dernière clôture
            kx = n - 1
            ex = c[kx] if d == 1 else c[kx] + sc[kx]
            why = EOD
            P_t[npth] = t[kx]
            P_u[npth] = (d * (ex - entry) - crt) / R
            npth += 1
        T_sess[nt] = s
        T_dir[nt] = d
        T_ent_t[nt] = t[j]
        T_ex_t[nt] = t[kx]
        T_ent[nt] = entry
        T_R[nt] = R
        T_rnet[nt] = (d * (ex - entry) - crt) / R
        T_rgross[nt] = d * (ex - entry) / R
        T_why[nt] = why
        T_cost_r[nt] = est_cost / R
        T_p0[nt] = p0
        T_pn[nt] = npth - p0
        nt += 1
    return (T_sess[:nt], T_dir[:nt], T_ent_t[:nt], T_ex_t[:nt], T_ent[:nt], T_R[:nt], T_rnet[:nt],
            T_rgross[:nt], T_why[:nt], T_cost_r[:nt], T_p0[:nt], T_pn[:nt], P_t[:npth].copy(), P_u[:npth].copy())


def run(P: dict, cfg: dict, tp_r: float = 0.0, spread_mult: float = 1.0, slip_mult: float = 1.0) -> dict:
    s = P["sess"]
    out = _simulate(P["t"], P["o"], P["h"], P["l"], P["c"], P["sc"] * spread_mult, P["atr15"], P["comm"],
                    P["slip"] * slip_mult, P["sslip"] * slip_mult,
                    s["open"].to_numpy(np.int64), s["close"].to_numpy(np.int64), s["i_open"].to_numpy(np.int64),
                    s[f"trend{cfg['L']}"].to_numpy(float), s["atr_d1"].to_numpy(float),
                    int(cfg["N"]), float(cfg["lo"]), float(cfg["hi"]), float(cfg["k"]), float(tp_r),
                    C.ENTRY_WINDOW_MIN, C.MIN_STOP_COST_MULT, C.OR_MIN_COVERAGE)
    keys = ("sess", "dir", "entry_t", "exit_t", "entry", "R", "r_net", "r_gross", "why", "cost_r", "p0", "pn")
    tr = {k: v for k, v in zip(keys, out[:12])}
    tr["symbol"] = P["symbol"]
    return dict(trades=tr, path_t=out[12], path_u=out[13])
