"""Lot 2 : stratégies publiées S16-S20 (docs/protocole_lot2.md)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

from . import catalogue as K
from . import config as C

LOT2 = {"S16": ("US100.cash", "US500.cash", "GER40.cash"), "S17": ("US100.cash", "US500.cash", "GER40.cash"),
        "S18": ("USDJPY",), "S19": ("USDJPY",), "S20": ("BTCUSD",)}


def _atr_d1(P):
    if "_atrD" not in P:
        t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
        D = K._bars(t, h, l, c, o, 86400)
        atr = pd.Series(K._tr(D["h"], D["l"], D["c"])).rolling(14, min_periods=14).mean().to_numpy()
        P["_atrD"] = (D["end"], atr)
    return P["_atrD"]


def atr_at(P, ts):
    end, atr = _atr_d1(P)
    k = np.searchsorted(end, ts, side="right") - 1
    return float(atr[k]) if k >= 0 else np.nan


def _pack(rows):
    cols = ("day", "j", "d", "stop", "tp", "exit_t", "trail")
    if not rows:
        return {k: np.zeros(0, np.int64 if k in ("day", "j", "d", "exit_t") else float) for k in cols}
    rows = sorted(rows, key=lambda r: r[1])
    return {k: np.asarray(v, dtype=np.int64 if k in ("day", "j", "d", "exit_t") else float) for k, v in zip(cols, zip(*rows))}


def _entry(P, j, d):
    return P["o"][j] + P["sc"][j] + P["slip"][j] if d == 1 else P["o"][j] - P["slip"][j]


def setups_s16(P):
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    rows = []
    for day, r in enumerate(P["days"].to_dict("records")):
        ws, fc = int(r["ws"]), int(r["fc"])
        i0, i1 = np.searchsorted(t, ws), np.searchsorted(t, ws + 300)
        if i1 - i0 < 4 or t[i0] > ws + 60:
            continue
        O, Cl, H, L = o[i0], c[i1 - 1], h[i0:i1].max(), l[i0:i1].min()
        if Cl == O:
            continue
        d = 1 if Cl > O else -1
        j = K._first_at(t, ws + 300, 120)
        if j is None:
            continue
        e = _entry(P, j, d)
        stop = L if d == 1 else H
        R = d * (e - stop)
        if R <= 0:
            continue
        rows.append((day, j, d, stop, e + d * 10 * R, fc, 0.0))
    return _pack(rows)


def _gotobi_dates(start, end):
    out = []
    for m in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        last = (m + pd.offsets.MonthEnd(0)).day
        for dd in (5, 10, 15, 20, 25, 30):
            x = pd.Timestamp(m.year, m.month, min(dd, last))
            while x.dayofweek >= 5:
                x -= pd.Timedelta(days=1)
            out.append(x)
    return sorted(set(out))


def setups_gotobi(P, long_before: bool):
    t = P["t"]
    rows = []
    t0 = pd.Timestamp(int(t[0]), unit="s")
    t1 = pd.Timestamp(int(t[-1]), unit="s")
    for d0 in _gotobi_dates(t0, t1):
        if long_before:
            a, b, d = K._ts(d0, "Asia/Tokyo", 8, 0), K._ts(d0, "Asia/Tokyo", 9, 55), 1
        else:
            a, b, d = K._ts(d0, "Asia/Tokyo", 9, 56), K._ts(d0, "Asia/Tokyo", 12, 0), -1
        j = K._first_at(t, a, 300)
        atr = atr_at(P, a)
        if j is None or not (atr > 0) or np.searchsorted(t, b) - np.searchsorted(t, a) < 0.8 * (b - a) / 60:
            continue
        e = _entry(P, j, d)
        rows.append((-1, j, d, e - d * 0.25 * atr, 0.0, b, 0.0))
    return _pack(rows)


def setups_s20(P):
    t = P["t"]
    rows = []
    for d0 in pd.date_range(pd.Timestamp(int(t[0]), unit="s").normalize(), pd.Timestamp(int(t[-1]), unit="s").normalize()):
        a = int(d0.value // 10**9) + 21 * 3600
        j = K._first_at(t, a, 300)
        atr = atr_at(P, a)
        # validité des données : au moins 80 % des minutes de la fenêtre présentes (comme le filtre des journées)
        if j is None or not (atr > 0) or np.searchsorted(t, a + 7200) - np.searchsorted(t, a) < 96:
            continue
        e = _entry(P, j, 1)
        rows.append((-1, j, 1, e - 0.5 * atr, 0.0, a + 2 * 3600, 0.0))
    return _pack(rows)


# ----------------------------------------------------------------------------- S17 Noise Area
def noise_inputs(P):
    """Par minute M1 de séance : σ de la minute, bases des bornes, VWAP, indice de séance ; le reste à NaN."""
    t, o, h, l, c, tv = P["t"], P["o"], P["h"], P["l"], P["c"], P["tv"]
    D = P["days"]
    n = len(t)
    sig = np.full(n, np.nan); ub0 = np.full(n, np.nan); lb0 = np.full(n, np.nan); vw = np.full(n, np.nan)
    sess = np.full(n, -1, np.int64)
    M = int((D["we"] - D["ws"]).max() // 60)
    A = np.full((len(D), M), np.nan)
    info = []
    for i, r in enumerate(D.to_dict("records")):
        ws, we = int(r["ws"]), int(r["we"])
        i0, i1 = np.searchsorted(t, ws), np.searchsorted(t, we)
        if i1 - i0 < 30 or t[i0] > ws + 60:
            info.append(None)
            continue
        O = o[i0]
        m = ((t[i0:i1] - ws) // 60).astype(np.int64)
        row = np.full(M, np.nan)
        row[m] = np.abs(c[i0:i1] / O - 1)
        row = pd.Series(row).ffill().to_numpy()
        A[i] = row
        info.append((i0, i1, O, m))
    sigma = pd.DataFrame(A).rolling(14, min_periods=14).mean().shift(1).to_numpy()   # 14 séances précédentes
    for i, inf in enumerate(info):
        if inf is None:
            continue
        i0, i1, O, m = inf
        pc = D.at[i, "pcs"]
        if not (pc > 0):
            continue
        s = sigma[i, m]
        sig[i0:i1] = s
        ub0[i0:i1] = max(O, pc)
        lb0[i0:i1] = min(O, pc)
        tp_ = (h[i0:i1] + l[i0:i1] + c[i0:i1]) / 3
        w = np.maximum(tv[i0:i1], 1.0)
        vw[i0:i1] = np.cumsum(tp_ * w) / np.cumsum(w)
        sess[i0:i1] = i
    return sig, ub0, lb0, vw, sess


@njit(cache=True)
def _noise_sim(t, o, h, l, c, sc, comm, slip, sslip, sig, ub0, lb0, vw, sess, ws_arr, fc_arr, min_cost_mult):
    n = t.shape[0]
    cap = n // 2 + 1024
    T_ent_t = np.zeros(cap // 50 + 1024, np.int64)
    m_cap = T_ent_t.shape[0]
    T_ex_t = np.zeros(m_cap, np.int64); T_ent = np.zeros(m_cap); T_R = np.zeros(m_cap); T_rnet = np.zeros(m_cap)
    T_rgross = np.zeros(m_cap); T_why = np.zeros(m_cap, np.int64); T_cost = np.zeros(m_cap); T_d = np.zeros(m_cap, np.int64)
    T_day = np.zeros(m_cap, np.int64); T_p0 = np.zeros(m_cap, np.int64); T_pn = np.zeros(m_cap, np.int64)
    P_t = np.empty(cap, np.int64); P_u = np.empty(cap)
    nt = 0
    npth = 0
    pos = 0
    entry = 0.0; stop = 0.0; R = 0.0; crt = 0.0; est = 0.0; p0 = 0; ej = 0
    pending = 0          # 0 rien, sinon sens de l'ordre à exécuter à l'ouverture suivante (2 = sortie seule)
    for k in range(n):
        s = sess[k]
        if s < 0:
            if pos != 0:
                # fin de séance inattendue : sortie à la dernière clôture connue
                pos = 0
            pending = 0
            continue
        fc = fc_arr[s]
        # 1) exécution des ordres décidés à la clôture précédente
        if pending != 0 and k > 0 and sess[k - 1] == s:
            if pos != 0:
                ex = o[k] - slip[k] if pos == 1 else o[k] + sc[k] + slip[k]
                if npth < cap and nt < m_cap:
                    P_t[npth] = t[k]; P_u[npth] = (pos * (ex - entry) - crt) / R; npth += 1
                    T_ex_t[nt] = t[k]; T_rnet[nt] = (pos * (ex - entry) - crt) / R; T_rgross[nt] = pos * (ex - entry) / R
                    T_why[nt] = 4; T_pn[nt] = npth - p0; nt += 1
                pos = 0
            if pending == 1 or pending == -1:
                d = pending
                e = o[k] + sc[k] + slip[k] if d == 1 else o[k] - slip[k]
                RR = sig[k] * (ub0[k] if d == 1 else lb0[k])
                cr = 2.0 * comm[k]
                es = sc[k] + 2.0 * slip[k] + cr
                if RR > 0 and RR >= min_cost_mult * es and t[k] < fc and nt < m_cap:
                    pos = d; entry = e; R = RR; stop = e - d * RR; crt = cr; est = es; p0 = npth; ej = k
                    T_ent_t[nt] = t[k]; T_ent[nt] = e; T_R[nt] = RR; T_cost[nt] = es / RR; T_d[nt] = d; T_day[nt] = s
                    T_p0[nt] = p0
            pending = 0
        # 2) gestion de la position sur la minute k
        if pos != 0:
            done = False
            if t[k] >= fc:
                ex = o[k] - slip[k] if pos == 1 else o[k] + sc[k] + slip[k]; why = 4; done = True
            elif pos == 1 and l[k] <= stop:
                ex = (o[k] if (k > ej and o[k] <= stop) else stop) - sslip[k]; why = 0; done = True
            elif pos == -1 and h[k] + sc[k] >= stop:
                ex = (o[k] + sc[k] if (k > ej and o[k] + sc[k] >= stop) else stop) + sslip[k]; why = 0; done = True
            if done:
                if npth < cap:
                    P_t[npth] = t[k]; P_u[npth] = (pos * (ex - entry) - crt) / R; npth += 1
                T_ex_t[nt] = t[k]; T_rnet[nt] = (pos * (ex - entry) - crt) / R; T_rgross[nt] = pos * (ex - entry) / R
                T_why[nt] = why; T_pn[nt] = npth - p0; nt += 1
                pos = 0
                continue
            if npth < cap:
                P_t[npth] = t[k]
                P_u[npth] = ((l[k] - entry) - crt) / R if pos == 1 else ((entry - (h[k] + sc[k])) - crt) / R
                npth += 1
        if t[k] >= fc:
            continue
        # 3) décision aux clôtures de :00 et :30, à partir de ouverture + 30 min
        ws = ws_arr[s]
        if (t[k] + 60 - ws) % 1800 == 0 and t[k] + 60 - ws >= 1800 and sig[k] == sig[k]:
            ub = ub0[k] * (1 + sig[k])
            lb = lb0[k] * (1 - sig[k])
            if pos == 0:
                if c[k] > ub:
                    pending = 1
                elif c[k] < lb:
                    pending = -1
            elif pos == 1:
                if c[k] < max(ub, vw[k]):
                    pending = -1 if c[k] < lb else 2
            else:
                if c[k] > min(lb, vw[k]):
                    pending = 1 if c[k] > ub else 2
    return (T_ent_t[:nt], T_ex_t[:nt], T_ent[:nt], T_R[:nt], T_rnet[:nt], T_rgross[:nt], T_why[:nt], T_cost[:nt],
            T_d[:nt], T_day[:nt], T_p0[:nt], T_pn[:nt], P_t[:npth].copy(), P_u[:npth].copy())


def run_s17(P):
    sig, ub0, lb0, vw, sess = noise_inputs(P)
    D = P["days"]
    out = _noise_sim(P["t"], P["o"], P["h"], P["l"], P["c"], P["sc"], P["comm"], P["slip"], P["sslip"], sig, ub0, lb0,
                     vw, sess, D["ws"].to_numpy(np.int64), D["fc"].to_numpy(np.int64), C.MIN_STOP_COST_MULT)
    keys = ("entry_t", "exit_t", "entry", "R", "r_net", "r_gross", "why", "cost_r", "dir", "day", "p0", "pn")
    df = pd.DataFrame({k: v for k, v in zip(keys, out[:12])})
    df["symbol"] = P["symbol"]
    return dict(trades=df, path_t=out[12], path_u=out[13])


def run(P, sid):
    if sid == "S17":
        return run_s17(P)
    S = {"S16": setups_s16, "S18": lambda P: setups_gotobi(P, True), "S19": lambda P: setups_gotobi(P, False),
         "S20": setups_s20}[sid](P)
    return K.run(P, sid, {}, setups=S)
