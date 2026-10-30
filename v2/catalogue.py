"""Catalogue de stratégies intraday S1-S15 (docs/protocole_catalogue.md).

Chaque stratégie produit des setups (minute d'entrée, sens, stop, objectif, sortie, trailing) à partir des seules
données connues à la décision ; la gestion minute par minute est faite par `_manage` (numba), avec les coûts de
v2 (BID, ASK = BID + coût de spread, glissements, commission)."""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from numba import njit

from . import config as C
from .data import load
from .prep import spread_cost

FX4 = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")
IDX = ("US100.cash", "US500.cash", "GER40.cash")
BTC = ("BTCUSD",)
PAIRS = {"PAIR_US100_US500": ("US100.cash", "US500.cash"), "PAIR_EURUSD_GBPUSD": ("EURUSD", "GBPUSD")}

# groupe -> (fuseau fenêtre, début h, m, fin h, m, fuseau clôture forcée, h, m)
GROUP = {
    "fx": ("Europe/London", 7, 0, 16, 0, "America/New_York", 16, 45),
    "us": ("America/New_York", 9, 30, 16, 0, "America/New_York", 15, 58),
    "de": ("Europe/Berlin", 9, 0, 17, 30, "Europe/Berlin", 17, 28),
    "btc": ("America/New_York", 9, 30, 16, 0, "America/New_York", 15, 58),
}
SYM_GROUP = {**{s: "fx" for s in FX4}, "US100.cash": "us", "US500.cash": "us", "GER40.cash": "de", "BTCUSD": "btc",
             "PAIR_US100_US500": "us", "PAIR_EURUSD_GBPUSD": "fx"}

STRATS = {
    "S1": dict(symbols=FX4, grid=dict(rend=(7, 8), stop=("opp", "mid"), exit=("time", "tp1.5"))),
    "S2": dict(symbols=IDX + BTC, grid=dict(k=(0.3, 0.6), s=(0.5, 1.0), filt=(0, 1))),
    "S3": dict(symbols=FX4 + IDX + BTC, grid=dict(a=(0.25, 0.5), exit=("time", "trail"), filt=(0, 1))),
    "S4": dict(symbols=FX4 + IDX + BTC, grid=dict(N=(20, 50), a=(1.5, 2.5), filt=(0, 1))),
    "S5": dict(symbols=IDX, grid=dict(z=(2.0, 2.5), s=(1.0, 2.0), wend=("noon", "late"))),
    "S6": dict(symbols=FX4 + IDX, grid=dict(b=(2.0, 2.5), a=(20, 25), c=(1.0, 2.0))),
    "S7": dict(symbols=FX4 + IDX + BTC, grid=dict(x=(0.0, 0.1), W=(2, 4), exit=("tp2", "time"))),
    "S8": dict(symbols=IDX, grid=dict(q=(0.1, 0.2), hold=(1, 2), filt=(0, 1))),
    "S9": dict(symbols=IDX, grid=dict(fixed=(1,))),
    "S10": dict(symbols=("US100.cash", "US500.cash"), grid=dict(fixed=(1,))),
    "S11": dict(symbols=IDX, grid=dict(fixed=(1,))),
    "S12": dict(symbols=FX4 + IDX + BTC, grid=dict(adaptive=(1,))),
    "S13": dict(symbols=FX4, grid=dict(p=(30, 60), thr=(0.0, 0.1), hold=(30, 60))),
    "S14": dict(symbols=FX4 + IDX, grid=dict(ev=("nfp", "nfp+cpi"), m=(0.0, 0.1), X=(60, 150))),
    "S15": dict(symbols=tuple(PAIRS), grid=dict(N=(48, 96), z0=(2.0, 2.5), exit=("tp", "time"))),
}
NO_SELECTION = ("S9", "S10", "S11")
FIRST_TEST = {**{s: "2018-01-01" for s in FX4}, **{s: "2023-07-01" for s in IDX}, "BTCUSD": "2020-07-01",
              "PAIR_US100_US500": "2023-07-01", "PAIR_EURUSD_GBPUSD": "2018-01-01"}
MIN_TRAIN = {"S14": 30}


def grid(sid: str) -> list[dict]:
    g = STRATS[sid]["grid"]
    keys = list(g)
    return [dict(zip(keys, v)) for v in product(*(g[k] for k in keys))]


def tag(c: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in c.items())


def neighbours(sid: str, c: dict) -> list[dict]:
    g = STRATS[sid]["grid"]
    out = []
    for k, vals in g.items():
        for v in vals:
            if v != c[k]:
                n = dict(c)
                n[k] = v
                out.append(n)
    return out


# ----------------------------------------------------------------------------- préparation
def _ts(d, tz, hh, mm) -> int:
    return int(pd.Timestamp(d.year, d.month, d.day, hh, mm, tz=tz).tz_convert("UTC").value // 10**9)


def _costs(sym, ic, price):
    if ic["quote"] == "JPY":
        comm = ic["comm_usd_lot_side"] / 100_000 * price
    else:
        comm = np.full(len(price), ic["comm_usd_lot_side"] / 100_000) + ic["comm_bps_side"] * 1e-4 * price
    if "slip_bps" in ic:
        return comm, ic["slip_bps"] * 1e-4 * price, ic["stop_slip_bps"] * 1e-4 * price
    return comm, np.full(len(price), ic["slip_px"]), np.full(len(price), ic["stop_slip_px"])


def _tr(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    return np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)


def _bars(t, h, l, c, o, sec):
    """Barres sur grille UTC de `sec` secondes. Renvoie dict avec fin (instant connu), indices M1 de la dernière minute."""
    b = t // sec
    starts = np.r_[0, np.nonzero(np.diff(b))[0] + 1]
    ends = np.r_[starts[1:], len(t)] - 1
    H = np.maximum.reduceat(h, starts)
    L = np.minimum.reduceat(l, starts)
    return dict(start=b[starts] * sec, end=(b[starts] + 1) * sec, i_last=ends, o=o[starts], h=H, l=L, c=c[ends])


def _align(bars_end, values, t):
    """Valeur de la dernière barre terminée à la clôture de chaque minute M1."""
    j = np.searchsorted(bars_end, t + 60, side="right") - 1
    out = np.full(len(t), np.nan)
    ok = j >= 0
    out[ok] = values[j[ok]]
    return out


def _base(sym):
    """Données M1 + coûts d'un instrument réel."""
    ic = C.INSTRUMENTS[sym]
    m = load(sym)
    sc, _ = spread_cost(m, ic)
    comm, slip, sslip = _costs(sym, ic, m["c"])
    return dict(symbol=sym, t=m["t"], o=m["o"], h=m["h"], l=m["l"], c=m["c"], tv=m["tv"].astype(float), sc=sc,
                comm=comm, slip=slip, sslip=sslip, spec=m["spec"])


def _synthetic(name):
    a, b = PAIRS[name]
    A, B = _base(a), _base(b)
    t = np.intersect1d(A["t"], B["t"])
    ia, ib = np.searchsorted(A["t"], t), np.searchsorted(B["t"], t)
    g = lambda X, k, i: X[k][i]
    S = {}
    S["t"] = t
    S["o"] = g(A, "o", ia) / g(B, "o", ib)
    S["c"] = g(A, "c", ia) / g(B, "c", ib)
    # extrêmes de minute : les deux jambes, très corrélées, font leurs extrêmes ensemble (haut/haut, bas/bas)
    hh = g(A, "h", ia) / g(B, "h", ib)
    lll = g(A, "l", ia) / g(B, "l", ib)
    S["h"] = np.maximum.reduce([S["o"], S["c"], hh, lll])
    S["l"] = np.minimum.reduce([S["o"], S["c"], hh, lll])
    ra = 1.0 / g(A, "c", ia)
    rb = 1.0 / g(B, "c", ib)
    for k in ("sc", "comm", "slip", "sslip"):
        S[k] = S["c"] * (g(A, k, ia) * ra + g(B, k, ib) * rb)
    S["tv"] = g(A, "tv", ia) + g(B, "tv", ib)
    S["symbol"] = name
    S["spec"] = dict(contract_size=1.0, volume_step=1e-6, volume_min=1e-6, point=1e-6)
    return S


def prepare(sym: str) -> dict:
    P = _synthetic(sym) if sym in PAIRS else _base(sym)
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    # D1 (jours UTC)
    D = _bars(t, h, l, c, o, 86400)
    trD = _tr(D["h"], D["l"], D["c"])
    atrD = pd.Series(trD).rolling(14, min_periods=14).mean().to_numpy()
    emaD = pd.Series(D["c"]).ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    trendD = np.sign(D["c"] - emaD)
    # M15 / H1 (grille UTC)
    M15 = _bars(t, h, l, c, o, 900)
    H1 = _bars(t, h, l, c, o, 3600)
    trH = _tr(H1["h"], H1["l"], H1["c"])
    H1["atr"] = pd.Series(trH).rolling(14, min_periods=14).mean().to_numpy()
    H1["ema200"] = pd.Series(H1["c"]).ewm(span=200, adjust=False, min_periods=200).mean().to_numpy()
    tr15 = _tr(M15["h"], M15["l"], M15["c"])
    M15["atr"] = pd.Series(tr15).rolling(14, min_periods=14).mean().to_numpy()
    P["atr15"] = _align(M15["end"], M15["atr"], t)
    P["atrH1"] = _align(H1["end"], H1["atr"], t)
    P["M15"], P["H1"] = M15, H1
    # journées
    g = GROUP[SYM_GROUP[sym]]
    tz, sh, sm, eh, em, ftz, fh, fm = g
    t0 = pd.Timestamp(int(t[0]), unit="s").normalize()
    t1 = pd.Timestamp(int(t[-1]), unit="s").normalize()
    rows = []
    for d in pd.date_range(t0, t1, freq="D"):
        if d.dayofweek >= 5:
            continue
        rows.append((d, _ts(d, tz, sh, sm), _ts(d, tz, eh, em), _ts(d, ftz, fh, fm)))
    days = pd.DataFrame(rows, columns=["date", "ws", "we", "fc"])
    days = days[(days["ws"] > t[0]) & (days["fc"] < t[-1])].reset_index(drop=True)
    days["i0"] = np.searchsorted(t, days["ws"].to_numpy(np.int64))
    days["i1"] = np.searchsorted(t, days["we"].to_numpy(np.int64))
    days["n"] = days["i1"] - days["i0"]
    span = (days["we"] - days["ws"]) // 60
    days = days[days["n"] >= 0.5 * span].reset_index(drop=True)          # journée avec données suffisantes
    od = days["ws"].to_numpy(np.int64) // 86400
    j = np.searchsorted(D["start"] // 86400, od, side="left") - 1     # dernier jour UTC terminé avant le début de fenêtre
    ok = j >= 0
    jj = np.maximum(j, 0)
    for k, v in (("atr", atrD), ("trend", trendD), ("pH", D["h"]), ("pL", D["l"]), ("pC", D["c"])):
        days[k] = np.where(ok, v[jj], np.nan)
    days["pR"] = days["pH"] - days["pL"]
    # clôture de la séance précédente (fenêtre) : dernière minute avant `we` de la journée précédente avec données
    pcs = np.full(len(days), np.nan)
    last = np.nan
    for i in range(len(days)):
        pcs[i] = last
        k = days.at[i, "i1"] - 1
        if k >= 0 and t[k] >= days.at[i, "we"] - 900:
            last = c[k]
    days["pcs"] = pcs
    P["days"] = days
    return P


# ----------------------------------------------------------------------------- moteur
@njit(cache=True)
def _manage(t, o, h, l, c, sc, comm, slip, sslip, atr_tr, S_j, S_d, S_stop, S_tp, S_exit, S_trail, min_cost_mult):
    n = t.shape[0]
    m = S_j.shape[0]
    T_keep = np.zeros(m, np.bool_)
    T_ent_t = np.zeros(m, np.int64); T_ex_t = np.zeros(m, np.int64)
    T_ent = np.zeros(m); T_R = np.zeros(m); T_rnet = np.zeros(m); T_rgross = np.zeros(m)
    T_why = np.zeros(m, np.int64); T_cost = np.zeros(m); T_p0 = np.zeros(m, np.int64); T_pn = np.zeros(m, np.int64)
    cap = m * 900 + 1024
    P_t = np.empty(cap, np.int64)
    P_u = np.empty(cap)
    npth = 0
    for i in range(m):
        j = S_j[i]
        d = S_d[i]
        entry = o[j] + sc[j] + slip[j] if d == 1 else o[j] - slip[j]
        stop = S_stop[i]
        tp = S_tp[i]
        kk = S_trail[i]
        R = d * (entry - stop)
        crt = 2.0 * comm[j]
        est = sc[j] + 2.0 * slip[j] + crt
        if R <= 0.0 or R < min_cost_mult * est or not (R == R):
            continue
        if tp > 0 and d * (tp - entry) <= 0:
            continue
        p0 = npth
        done = False
        ex = 0.0
        why = 5
        kx = j
        hh = -1e300
        ll = 1e300
        k = j
        while k < n and npth < cap:
            if t[k] >= S_exit[i]:
                ex = o[k] - slip[k] if d == 1 else o[k] + sc[k] + slip[k]
                why = 4
                done = True
            elif d == 1:
                if l[k] <= stop:
                    if k > j and o[k] <= stop:
                        ex = o[k] - sslip[k]; why = 1
                    else:
                        ex = stop - sslip[k]; why = 0
                    done = True
                elif tp > 0 and h[k] >= tp:
                    ex = tp; why = 3; done = True
            else:
                if h[k] + sc[k] >= stop:
                    if k > j and o[k] + sc[k] >= stop:
                        ex = o[k] + sc[k] + sslip[k]; why = 1
                    else:
                        ex = stop + sslip[k]; why = 0
                    done = True
                elif tp > 0 and l[k] + sc[k] <= tp:
                    ex = tp; why = 3; done = True
            P_t[npth] = t[k]
            if done:
                P_u[npth] = (d * (ex - entry) - crt) / R
                npth += 1
                kx = k
                break
            if d == 1:
                P_u[npth] = (l[k] - entry - crt) / R
            else:
                P_u[npth] = (entry - (h[k] + sc[k]) - crt) / R
            npth += 1
            if h[k] > hh:
                hh = h[k]
            if l[k] < ll:
                ll = l[k]
            if kk > 0 and (t[k] + 60 - t[j]) % 300 == 0 and atr_tr[k] > 0:
                if d == 1:
                    ns = hh - kk * atr_tr[k]
                    if ns > stop:
                        stop = ns
                else:
                    ns = ll + kk * atr_tr[k]
                    if ns < stop:
                        stop = ns
            k += 1
        if not done:
            kx = min(k, n - 1)
            ex = c[kx] if d == 1 else c[kx] + sc[kx]
            P_t[npth] = t[kx]; P_u[npth] = (d * (ex - entry) - crt) / R; npth += 1
        T_keep[i] = True; T_ent_t[i] = t[j]; T_ex_t[i] = t[kx]; T_ent[i] = entry; T_R[i] = R
        T_rnet[i] = (d * (ex - entry) - crt) / R; T_rgross[i] = d * (ex - entry) / R
        T_why[i] = why; T_cost[i] = est / R; T_p0[i] = p0; T_pn[i] = npth - p0
    return (T_keep, T_ent_t, T_ex_t, T_ent, T_R, T_rnet, T_rgross, T_why, T_cost, T_p0, T_pn,
            P_t[:npth].copy(), P_u[:npth].copy())


class _Setups:
    def __init__(self, P):
        self.P = P
        self.rows = []

    def entry_px(self, j, d):
        P = self.P
        return P["o"][j] + P["sc"][j] + P["slip"][j] if d == 1 else P["o"][j] - P["slip"][j]

    def add(self, day, j, d, stop, tp=0.0, exit_t=None, trail=0.0):
        P = self.P
        if j is None or j >= len(P["t"]):
            return
        fc = int(P["_recs"][day]["fc"])
        if P["t"][j] >= fc:
            return
        self.rows.append((day, j, d, stop, tp, min(exit_t, fc) if exit_t is not None else fc, trail))


def _next_bar(t, k, limit=300):
    """Minute suivant la minute k (barre de signal) ; None si trou > 5 min."""
    j = k + 1
    if j >= len(t) or t[j] - t[k] > limit:
        return None
    return j


def _first_at(t, ts, tol=300):
    j = int(np.searchsorted(t, ts, side="left"))
    if j >= len(t) or t[j] >= ts + tol:
        return None
    return j


def _price_before(t, c, ts, tol=600):
    k = int(np.searchsorted(t, ts, side="left")) - 1
    if k < 0 or t[k] < ts - tol:
        return None
    return float(c[k])


def _m5_idx(t, a, b):
    """Indices des minutes de [a, b) qui clôturent une barre M5 (grille d'horloge)."""
    i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
    k = np.arange(i0, i1)
    return k[(t[k] + 60) % 300 == 0]


def build(P: dict, sid: str, cfg: dict, cal: dict | None = None) -> dict:
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    D = P["days"]
    S = _Setups(P)
    grp = SYM_GROUP[P["symbol"]]
    tz = GROUP[grp][0]
    if "_recs" not in P:
        P["_recs"] = D.to_dict("records")
    recs = P["_recs"]
    for day in range(len(D)):
        r = recs[day]
        ws, we, fc = int(r["ws"]), int(r["we"]), int(r["fc"])
        atr = r["atr"]
        if not (atr > 0):
            continue
        if sid == "S1":
            d0 = r["date"]
            a = _ts(d0, "Europe/London", 0, 0)
            rend = _ts(d0, "Europe/London", cfg["rend"], 0)
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, rend)
            if i1 - i0 < 0.8 * (rend - a) / 60:
                continue
            H, L = h[i0:i1].max(), l[i0:i1].min()
            if not (0.1 <= (H - L) / atr <= 1.0):
                continue
            for k in _m5_idx(t, rend, rend + 3 * 3600):
                d = 1 if c[k] > H else (-1 if c[k] < L else 0)
                if d:
                    j = _next_bar(t, k)
                    if j is None:
                        break
                    e = S.entry_px(j, d)
                    stop = (L if d == 1 else H) if cfg["stop"] == "opp" else (H + L) / 2
                    tp = e + d * 1.5 * d * (e - stop) if cfg["exit"] == "tp1.5" else 0.0
                    S.add(day, j, d, stop, tp)
                    break
        elif sid == "S2":
            j0 = _first_at(t, ws, 120)
            if j0 is None or not (r["pR"] > 0):
                continue
            O = o[j0]
            up, dn = O + cfg["k"] * r["pR"], O - cfg["k"] * r["pR"]
            i1 = np.searchsorted(t, we - 3600)
            for k in range(j0, i1):
                d = 1 if c[k] > up else (-1 if c[k] < dn else 0)
                if d:
                    if cfg["filt"] and d != r["trend"]:
                        break
                    j = _next_bar(t, k)
                    if j is None:
                        break
                    e = S.entry_px(j, d)
                    S.add(day, j, d, e - d * cfg["s"] * r["pR"])
                    break
        elif sid == "S3":
            if not (r["pH"] > 0):
                continue
            for k in _m5_idx(t, ws, ws + 4 * 3600):
                d = 1 if c[k] > r["pH"] else (-1 if c[k] < r["pL"] else 0)
                if d:
                    if cfg["filt"] and d != r["trend"]:
                        break
                    j = _next_bar(t, k)
                    if j is None:
                        break
                    e = S.entry_px(j, d)
                    S.add(day, j, d, e - d * cfg["a"] * atr, trail=3.0 if cfg["exit"] == "trail" else 0.0)
                    break
        elif sid == "S4":
            H1 = P["H1"]
            hi_prev = pd.Series(H1["h"]).shift(1).rolling(cfg["N"], min_periods=cfg["N"]).max().to_numpy() \
                if "_dc" not in P or P["_dc"][0] != cfg["N"] else P["_dc"][1]
            lo_prev = pd.Series(H1["l"]).shift(1).rolling(cfg["N"], min_periods=cfg["N"]).min().to_numpy() \
                if "_dc" not in P or P["_dc"][0] != cfg["N"] else P["_dc"][2]
            P["_dc"] = (cfg["N"], hi_prev, lo_prev)
            b0, b1 = np.searchsorted(H1["end"], ws + 3600), np.searchsorted(H1["end"], we - 3600, side="right")
            for bi in range(b0, b1):
                cl = H1["c"][bi]
                d = 1 if cl > hi_prev[bi] else (-1 if cl < lo_prev[bi] else 0)
                if d:
                    if cfg["filt"] and not (d * (cl - H1["ema200"][bi]) > 0):
                        continue
                    j = _first_at(t, int(H1["end"][bi]))
                    if j is None or not (H1["atr"][bi] > 0):
                        break
                    e = S.entry_px(j, d)
                    S.add(day, j, d, e - d * cfg["a"] * H1["atr"][bi], trail=3.0)
                    break
        elif sid == "S5":
            i0 = _first_at(t, ws, 120)
            if i0 is None:
                continue
            wend = _ts(r["date"], tz, 12, 0) if cfg["wend"] == "noon" else we - 3600
            i1 = np.searchsorted(t, wend)
            if i1 <= i0 + 30:
                continue
            seg = slice(i0, i1)
            tp_ = (h[seg] + l[seg] + c[seg]) / 3
            w = np.maximum(P["tv"][seg], 1.0)
            vwap = np.cumsum(tp_ * w) / np.cumsum(w)
            dev = c[seg] - vwap
            n_ = np.arange(1, len(dev) + 1)
            m1 = np.cumsum(dev) / n_
            m2 = np.cumsum(dev * dev) / n_
            sig = np.sqrt(np.maximum(m2 - m1 * m1, 0))
            for k in _m5_idx(t, ws + 1800, wend):
                q = k - i0
                if q < 0 or q >= len(dev) or not (sig[q] > 0):
                    continue
                if abs(dev[q]) > cfg["z"] * sig[q]:
                    d = -1 if dev[q] > 0 else 1
                    j = _next_bar(t, k)
                    if j is None:
                        break
                    e = S.entry_px(j, d)
                    S.add(day, j, d, e - d * cfg["s"] * sig[q], tp=vwap[q])
                    break
        elif sid == "S6":
            M = P["M15"]
            key = ("_bb", cfg["b"])
            if key not in P:
                ma = pd.Series(M["c"]).rolling(20, min_periods=20).mean().to_numpy()
                sd = pd.Series(M["c"]).rolling(20, min_periods=20).std(ddof=0).to_numpy()
                P[key] = (ma, ma + cfg["b"] * sd, ma - cfg["b"] * sd)
            if "_adx" not in P:
                P["_adx"] = _adx(M["h"], M["l"], M["c"])
            ma, up, dn = P[key]
            adx = P["_adx"]
            a0 = ws + (1800 if grp in ("us", "de") else 0)
            a1 = we - (3600 if grp in ("us", "de") else 0)
            b0, b1 = np.searchsorted(M["end"], a0 + 1), np.searchsorted(M["end"], a1, side="right")
            for bi in range(b0, b1):
                if not (adx[bi] < cfg["a"]):
                    continue
                cl = M["c"][bi]
                d = 1 if cl < dn[bi] else (-1 if cl > up[bi] else 0)
                if d:
                    j = _first_at(t, int(M["end"][bi]))
                    if j is None or not (M["atr"][bi] > 0):
                        break
                    e = S.entry_px(j, d)
                    S.add(day, j, d, e - d * cfg["c"] * M["atr"][bi], tp=ma[bi])
                    break
        elif sid == "S7":
            if not (r["pH"] > 0):
                continue
            i0 = np.searchsorted(t, ws)
            iw = np.searchsorted(t, ws + cfg["W"] * 3600)
            if iw <= i0:
                continue
            rmax = np.maximum.accumulate(h[i0:iw])
            rmin = np.minimum.accumulate(l[i0:iw])
            armed_s = rmax > r["pH"] + cfg["x"] * atr
            armed_l = rmin < r["pL"] - cfg["x"] * atr
            for k in _m5_idx(t, ws, ws + cfg["W"] * 3600):
                q = k - i0
                d = 0
                if armed_s[q] and c[k] < r["pH"]:
                    d, stop = -1, rmax[q] + 0.05 * atr
                elif armed_l[q] and c[k] > r["pL"]:
                    d, stop = 1, rmin[q] - 0.05 * atr
                if d:
                    j = _next_bar(t, k)
                    if j is None:
                        break
                    e = S.entry_px(j, d)
                    tp = e + d * 2.0 * d * (e - stop) if cfg["exit"] == "tp2" else 0.0
                    S.add(day, j, d, stop, tp)
                    break
        elif sid == "S8":
            hold = cfg["hold"] * 3600
            k_end = ws + 3600
            while k_end <= we - 3600 - hold:
                a = k_end - 3600
                i0, i1 = np.searchsorted(t, a), np.searchsorted(t, k_end)
                if i1 - i0 >= 30:
                    H, L, Cl = h[i0:i1].max(), l[i0:i1].min(), c[i1 - 1]
                    if H > L:
                        ibs = (Cl - L) / (H - L)
                        d = 1 if ibs < cfg["q"] else (-1 if ibs > 1 - cfg["q"] else 0)
                        if d and not (cfg["filt"] and d != r["trend"]):
                            j = _first_at(t, k_end)
                            if j is not None and P["atrH1"][j - 1] > 0:
                                e = S.entry_px(j, d)
                                S.add(day, j, d, e - d * P["atrH1"][j - 1], exit_t=int(t[j]) + hold)
                            break
                k_end += 3600
        elif sid == "S9":
            pc = r["pcs"]
            if not (pc > 0):
                continue
            p1 = _price_before(t, c, ws + 1800, 300)
            if p1 is None or p1 == pc:
                continue
            d = -1 if p1 > pc else 1
            j = _first_at(t, we - 1800)
            if j is None:
                continue
            e = S.entry_px(j, d)
            S.add(day, j, d, e - d * 0.5 * atr)
        elif sid == "S10":
            if r["date"].date() not in cal["fomc"]:
                continue
            j = _first_at(t, _ts(r["date"], "America/New_York", 9, 31))
            if j is None:
                continue
            e = S.entry_px(j, 1)
            S.add(day, j, 1, e - 0.5 * atr, exit_t=_ts(r["date"], "America/New_York", 13, 55))
        elif sid == "S11":
            if day not in P["_tom"]:
                continue
            j = _first_at(t, ws + 60)
            if j is None:
                continue
            e = S.entry_px(j, 1)
            S.add(day, j, 1, e - 0.5 * atr)
        elif sid == "S13":
            d0 = r["date"]
            fix = _ts(d0, "Europe/London", 16, 0)
            p_end = _price_before(t, c, fix, 120)
            p_start = _price_before(t, c, fix - cfg["p"] * 60, 120)
            if p_end is None or p_start is None or p_end == p_start:
                continue
            if abs(p_end - p_start) < cfg["thr"] * atr:
                continue
            d = -1 if p_end > p_start else 1
            j = _first_at(t, fix + 60)
            if j is None:
                continue
            e = S.entry_px(j, d)
            S.add(day, j, d, e - d * 0.25 * atr, exit_t=int(t[j]) + cfg["hold"] * 60)
        elif sid == "S14":
            dd = r["date"].date()
            evs = cal["nfp"] if cfg["ev"] == "nfp" else (cal["nfp"] | cal["cpi"])
            if dd not in evs:
                continue
            t830 = _ts(r["date"], "America/New_York", 8, 30)
            p0 = _price_before(t, c, t830, 300)
            p5 = _price_before(t, c, t830 + 300, 120)
            if p0 is None or p5 is None or p5 == p0 or abs(p5 - p0) < cfg["m"] * atr:
                continue
            d = 1 if p5 > p0 else -1
            j = _first_at(t, t830 + 300)
            if j is None:
                continue
            e = S.entry_px(j, d)
            if d * (e - p0) <= 0:
                continue
            S.add(day, j, d, p0, exit_t=int(t[j]) + cfg["X"] * 60)
        elif sid == "S15":
            key = ("_z", cfg["N"])
            if key not in P:
                M5c = pd.Series(c[(t + 60) % 300 == 0], index=np.nonzero((t + 60) % 300 == 0)[0])
                mu = M5c.rolling(cfg["N"], min_periods=cfg["N"]).mean()
                sd = M5c.rolling(cfg["N"], min_periods=cfg["N"]).std(ddof=0)
                zz = np.full(len(t), np.nan); mm = np.full(len(t), np.nan); ss = np.full(len(t), np.nan)
                zz[M5c.index] = ((M5c - mu) / sd).to_numpy(); mm[M5c.index] = mu.to_numpy(); ss[M5c.index] = sd.to_numpy()
                P[key] = (zz, mm, ss)
            zz, mm, ss = P[key]
            a0 = ws + (1800 if grp == "us" else 0)
            a1 = we - (3600 if grp == "us" else 0)
            for k in _m5_idx(t, a0, a1):
                if not (abs(zz[k]) > cfg["z0"]):
                    continue
                d = -1 if zz[k] > 0 else 1
                j = _next_bar(t, k)
                if j is None:
                    break
                e = S.entry_px(j, d)
                stop = mm[k] - d * (cfg["z0"] + 1.5) * ss[k]
                if d * (e - stop) <= 0:
                    break
                if cfg["exit"] == "tp":
                    S.add(day, j, d, stop, tp=mm[k])
                else:
                    S.add(day, j, d, stop, exit_t=int(t[j]) + 3600)
                break
    rows = S.rows
    cols = ("day", "j", "d", "stop", "tp", "exit_t", "trail")
    if not rows:
        return {k: np.zeros(0, np.int64 if k in ("day", "j", "d", "exit_t") else float) for k in cols}
    arr = list(zip(*rows))
    return {k: np.asarray(v, dtype=np.int64 if k in ("day", "j", "d", "exit_t") else float) for k, v in zip(cols, arr)}


def _adx(h, l, c, n=14):
    up = np.r_[0, np.diff(h)]
    dn = np.r_[0, -np.diff(l)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _tr(h, l, c)
    a = 1.0 / n
    s = lambda x: pd.Series(x).ewm(alpha=a, adjust=False, min_periods=n).mean().to_numpy()
    atr = s(tr)
    pdi = 100 * s(pdm) / atr
    mdi = 100 * s(mdm) / atr
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    return s(dx)


def tom_days(P: dict) -> set:
    D = P["days"]
    dates = pd.to_datetime(D["date"])
    ym = dates.dt.year * 12 + dates.dt.month
    out = set()
    for key, idx in D.groupby(ym.to_numpy()).groups.items():
        idx = sorted(idx)
        out.add(idx[-1])                       # dernier jour ouvré du mois
        out.update(idx[:3])                    # 3 premiers jours ouvrés
    return out


def run(P: dict, sid: str, cfg: dict, cal=None, setups: dict | None = None) -> dict:
    if sid == "S11" and "_tom" not in P:
        P["_tom"] = tom_days(P)
    S = setups if setups is not None else build(P, sid, cfg, cal)
    atr_tr = P["atrH1"] if sid == "S4" else P["atr15"]
    out = _manage(P["t"], P["o"], P["h"], P["l"], P["c"], P["sc"], P["comm"], P["slip"], P["sslip"], atr_tr,
                  S["j"], S["d"], S["stop"], S["tp"], S["exit_t"], S["trail"], C.MIN_STOP_COST_MULT)
    keep = out[0]
    keys = ("entry_t", "exit_t", "entry", "R", "r_net", "r_gross", "why", "cost_r", "p0", "pn")
    df = pd.DataFrame({k: v[keep] for k, v in zip(keys, out[1:11])})
    df["dir"] = S["d"][keep]
    df["day"] = S["day"][keep]
    df["symbol"] = P["symbol"]
    return dict(trades=df, path_t=out[11], path_u=out[12])


def stressed(P: dict, spread_mult=1.5, slip_mult=2.0) -> dict:
    Q = dict(P)
    Q["sc"] = P["sc"] * spread_mult
    Q["slip"] = P["slip"] * slip_mult
    Q["sslip"] = P["sslip"] * slip_mult
    return Q
