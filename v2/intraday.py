"""Hypothèses intraday H1-H3 (docs/protocole_jalon_A_intraday.md §2).

Les setups (barre d'entrée, sens, stop, objectif, heure de sortie) sont calculés en Python à partir des seules
données disponibles à l'instant de la décision ; la gestion minute par minute est faite par `_manage` (numba),
avec les mêmes conventions de coûts que sim.py (BID, ASK = BID + coût de spread, glissements, commission)."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from numba import njit

from . import config as C
from .prep import d1_features, spread_cost
from .data import load

STOP, GAP, TP, TIME, EOD = 0, 1, 3, 4, 5
REASONS = np.array(["stop", "gap", "trail", "tp", "time", "eod"])


def grid(h: str) -> list[dict]:
    H = C.HYP[h]
    return [dict(zip(H["keys"], v)) for v in product(*(H["values"][k] for k in H["keys"]))]


def tag(c: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in c.items())


def neighbours(h: str, c: dict) -> list[dict]:
    H = C.HYP[h]
    out = []
    for k in H["keys"]:
        for v in H["values"][k]:
            if v != c[k]:
                n = dict(c)
                n[k] = v
                out.append(n)
    return out


def _local_ts(d, tz, hh, mm) -> int:
    return int(pd.Timestamp(d.year, d.month, d.day, hh, mm, tz=tz).tz_convert("UTC").value // 10**9)


def prepare(symbol: str) -> dict:
    ic = C.INSTRUMENTS[symbol]
    m = load(symbol)
    sc, prof = spread_cost(m, ic)
    tz, oh, om, ch, cm = C.CASH[symbol]
    t = m["t"]
    d1 = d1_features(m, (20,))
    rows = []
    t0 = pd.Timestamp(int(t[0]), unit="s").normalize()
    t1 = pd.Timestamp(int(t[-1]), unit="s").normalize()
    for d in pd.date_range(t0, t1, freq="D"):
        if d.dayofweek >= 5:
            continue
        rows.append((_local_ts(d, tz, oh, om), _local_ts(d, tz, ch, cm), d))
    s = pd.DataFrame(rows, columns=["open", "close", "date"])
    s = s[(s["open"] > t[0]) & (s["close"] < t[-1])].reset_index(drop=True)

    def last_price_before(ts_end: int, max_age: int = 900) -> float:
        k = np.searchsorted(t, ts_end, side="left") - 1
        if k < 0 or t[k] < ts_end - max_age:
            return np.nan
        return float(m["c"][k])

    # clôture de référence de la veille
    pc = np.full(len(s), np.nan)
    if symbol == "BTCUSD":
        for i, r in s.iterrows():
            prev = r["date"] - pd.Timedelta(days=1)
            pc[i] = last_price_before(_local_ts(prev, tz, ch, cm))
    else:
        last_valid = np.nan
        for i, r in s.iterrows():
            pc[i] = last_valid                      # dernière séance précédente ayant des données
            v = last_price_before(int(r["close"]))
            if not np.isnan(v):
                last_valid = v
    s["pc"] = pc
    od = s["open"].to_numpy(np.int64) // 86400
    j = np.searchsorted(d1["days"], od, side="left") - 1
    s["atr_d1"] = np.where(j >= 0, d1["atr"][np.maximum(j, 0)], np.nan)
    s["i_open"] = np.searchsorted(t, s["open"].to_numpy(np.int64), side="left")
    price = m["c"]
    comm = np.full(len(price), ic["comm_usd_lot_side"] / 100_000) + ic["comm_bps_side"] * 1e-4 * price
    if "slip_bps" in ic:
        slip, sslip = ic["slip_bps"] * 1e-4 * price, ic["stop_slip_bps"] * 1e-4 * price
    else:
        slip, sslip = np.full(len(price), ic["slip_px"]), np.full(len(price), ic["stop_slip_px"])
    return dict(symbol=symbol, t=t, o=m["o"], h=m["h"], l=m["l"], c=m["c"], sc=sc, profile=prof, comm=comm,
                slip=slip, sslip=sslip, sess=s, spec=m["spec"])


def setups(P: dict, h: str, cfg: dict) -> dict:
    """Renvoie les setups (une ligne par séance retenue) : index de la minute d'entrée, sens, stop, objectif, sortie."""
    t, o, c, sc, slip = P["t"], P["o"], P["c"], P["sc"], P["slip"]
    s = P["sess"]
    out = dict(sess=[], j=[], d=[], stop=[], tp=[], exit_t=[])
    for i, r in s.iterrows():
        pc, atr = r["pc"], r["atr_d1"]
        if not (pc > 0) or not (atr > 0):
            continue
        op, cl = int(r["open"]), int(r["close"])
        if h in ("H1", "H2"):
            t1 = op + 30 * 60
            k = np.searchsorted(t, t1, side="left") - 1           # dernière minute terminée à t1
            if k < 0 or t[k] < t1 - 300:
                continue
            p1 = c[k]
            if abs(p1 - pc) < cfg["thr"] * atr or p1 == pc:
                continue
            d = 1 if p1 > pc else -1
            te = cl - cfg["E"] * 60
            j = np.searchsorted(t, te, side="left")
            if j >= len(t) or t[j] >= te + 300:
                continue
            entry = o[j] + sc[j] + slip[j] if d == 1 else o[j] - slip[j]
            stop = entry - d * cfg["m"] * atr
            tp = 0.0
            exit_t = cl - 120
        else:
            j0 = np.searchsorted(t, op, side="left")
            if j0 >= len(t) - 1 or t[j0] >= op + 120:
                continue
            gap = o[j0] - pc
            ratio = abs(gap) / atr
            if ratio < cfg["lo"] or ratio > cfg["hi"] or gap == 0:
                continue
            j = j0 + 1
            if t[j] - t[j0] > 300:
                continue
            d = -1 if gap > 0 else 1
            entry = o[j] + sc[j] + slip[j] if d == 1 else o[j] - slip[j]
            tp = pc
            if d * (tp - entry) <= 0:
                continue
            stop = entry - d * abs(gap)
            exit_t = op + cfg["X"] * 60
        out["sess"].append(i)
        out["j"].append(j)
        out["d"].append(d)
        out["stop"].append(stop)
        out["tp"].append(tp)
        out["exit_t"].append(exit_t)
    return {k: np.asarray(v, dtype=np.int64 if k in ("sess", "j", "d", "exit_t") else float) for k, v in out.items()}


@njit(cache=True)
def _manage(t, o, h, l, c, sc, comm, slip, sslip, S_j, S_d, S_stop, S_tp, S_exit, min_cost_mult):
    n = t.shape[0]
    m = S_j.shape[0]
    T_dir = np.empty(m, np.int64); T_ent_t = np.empty(m, np.int64); T_ex_t = np.empty(m, np.int64)
    T_ent = np.empty(m); T_R = np.empty(m); T_rnet = np.empty(m); T_rgross = np.empty(m)
    T_why = np.empty(m, np.int64); T_cost = np.empty(m); T_p0 = np.empty(m, np.int64); T_pn = np.empty(m, np.int64)
    T_keep = np.zeros(m, np.bool_)
    cap = m * 600 + 1024
    P_t = np.empty(cap, np.int64)
    P_u = np.empty(cap)
    npth = 0
    for i in range(m):
        j = S_j[i]
        d = S_d[i]
        entry = o[j] + sc[j] + slip[j] if d == 1 else o[j] - slip[j]
        stop = S_stop[i]
        tp = S_tp[i]
        R = d * (entry - stop)
        crt = 2.0 * comm[j]
        est = sc[j] + 2.0 * slip[j] + crt
        if R <= 0.0 or R < min_cost_mult * est:
            continue
        p0 = npth
        done = False
        ex = 0.0
        why = EOD
        kx = j
        k = j
        while k < n and npth < cap:
            if t[k] >= S_exit[i]:
                ex = o[k] - slip[k] if d == 1 else o[k] + sc[k] + slip[k]
                why = TIME
                done = True
            elif d == 1:
                if l[k] <= stop:
                    if k > j and o[k] <= stop:
                        ex = o[k] - sslip[k]; why = GAP
                    else:
                        ex = stop - sslip[k]; why = STOP
                    done = True
                elif tp > 0 and h[k] >= tp:
                    ex = tp; why = TP; done = True
            else:
                if h[k] + sc[k] >= stop:
                    if k > j and o[k] + sc[k] >= stop:
                        ex = o[k] + sc[k] + sslip[k]; why = GAP
                    else:
                        ex = stop + sslip[k]; why = STOP
                    done = True
                elif tp > 0 and l[k] + sc[k] <= tp:
                    ex = tp; why = TP; done = True
            P_t[npth] = t[k]
            if done:
                P_u[npth] = (d * (ex - entry) - crt) / R
                npth += 1
                kx = k
                break
            P_u[npth] = ((l[k] - entry) - crt) / R if d == 1 else ((entry - (h[k] + sc[k])) - crt) / R
            npth += 1
            k += 1
        if not done:
            kx = min(k, n - 1)
            ex = c[kx] if d == 1 else c[kx] + sc[kx]
            P_t[npth] = t[kx]; P_u[npth] = (d * (ex - entry) - crt) / R; npth += 1
        T_dir[i] = d; T_ent_t[i] = t[j]; T_ex_t[i] = t[kx]; T_ent[i] = entry; T_R[i] = R
        T_rnet[i] = (d * (ex - entry) - crt) / R; T_rgross[i] = d * (ex - entry) / R
        T_why[i] = why; T_cost[i] = est / R; T_p0[i] = p0; T_pn[i] = npth - p0; T_keep[i] = True
    return (T_keep, T_dir, T_ent_t, T_ex_t, T_ent, T_R, T_rnet, T_rgross, T_why, T_cost, T_p0, T_pn,
            P_t[:npth].copy(), P_u[:npth].copy())


def run(P: dict, h: str, cfg: dict, spread_mult: float = 1.0, slip_mult: float = 1.0) -> dict:
    S = setups(P, h, cfg)
    sc = P["sc"] * spread_mult
    if spread_mult != 1.0 or slip_mult != 1.0:
        # le stop et l'objectif sont définis à partir du prix d'entrée de base ; on les recalcule avec les coûts stressés
        S = _restress(P, h, cfg, S, sc, P["slip"] * slip_mult)
    out = _manage(P["t"], P["o"], P["h"], P["l"], P["c"], sc, P["comm"], P["slip"] * slip_mult,
                  P["sslip"] * slip_mult, S["j"], S["d"], S["stop"], S["tp"], S["exit_t"], C.MIN_STOP_COST_MULT)
    keep = out[0]
    keys = ("dir", "entry_t", "exit_t", "entry", "R", "r_net", "r_gross", "why", "cost_r", "p0", "pn")
    df = pd.DataFrame({k: v[keep] for k, v in zip(keys, out[1:12])})
    df["sess"] = S["sess"][keep]
    df["symbol"] = P["symbol"]
    return dict(trades=df, path_t=out[12], path_u=out[13])


def _restress(P, h, cfg, S, sc, slip):
    o = P["o"]
    S = {k: v.copy() for k, v in S.items()}
    for i in range(len(S["j"])):
        j, d = S["j"][i], S["d"][i]
        entry = o[j] + sc[j] + slip[j] if d == 1 else o[j] - slip[j]
        r = P["sess"].iloc[S["sess"][i]]
        if h in ("H1", "H2"):
            S["stop"][i] = entry - d * cfg["m"] * r["atr_d1"]
        else:
            S["stop"][i] = entry - d * abs(o[j - 1] - r["pc"])
    return S
