"""Préparation par instrument : coût de spread par minute, sessions, tendance et ATR D1, ATR M15.

Conventions (protocole §3-4) :
- prix BID ; ask reconstruit = bid + coût de spread ;
- D1 = jour UTC ; la tendance et l'ATR D1 d'une session n'utilisent que les jours UTC terminés avant l'ouverture ;
- ATR M15 connu à la clôture de chaque barre M15 (grille UTC)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .data import load


def spread_cost(m: dict, ic: dict) -> np.ndarray:
    """max(profil p75 par quart d'heure serveur, observé si dans la fenêtre de confiance, plancher)."""
    ts = m["t_server"]
    qh = ((ts % 86400) // 900).astype(np.int64)
    t = m["t"]
    a = int(pd.Timestamp(C.PROFILE_FIT[0]).value // 10**9)
    b = int((pd.Timestamp(C.PROFILE_FIT[1]) + pd.Timedelta(days=1)).value // 10**9)
    obs = m["spread_obs"]
    sel = (t >= a) & (t < b) & (obs > 0)
    prof = pd.Series(obs[sel]).groupby(qh[sel]).quantile(C.PROFILE_Q).reindex(range(96))
    prof = prof.fillna(prof.max()).to_numpy(float)
    ta = int(pd.Timestamp(ic["trust"][0]).value // 10**9)
    tb = int((pd.Timestamp(ic["trust"][1]) + pd.Timedelta(days=1)).value // 10**9)
    trusted = (t >= ta) & (t < tb)
    o = np.where(trusted, obs, 0.0)
    return np.maximum(np.maximum(prof[qh], o), ic["floor_px"]), prof


def d1_features(m: dict, lengths=(20, 50)) -> dict:
    """Barres D1 sur jours UTC. Renvoie, par jour UTC d (en jours depuis l'epoch), les valeurs connues à 00:00 UTC de d
    (donc calculées jusqu'au jour d-1 inclus) : signe de tendance par L, ATR14."""
    day = m["t"] // 86400
    df = pd.DataFrame({"d": day, "h": m["h"], "l": m["l"], "c": m["c"]})
    g = df.groupby("d").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    pc = g["c"].shift(1)
    tr = np.maximum(g["h"] - g["l"], np.maximum((g["h"] - pc).abs(), (g["l"] - pc).abs()))
    atr = tr.rolling(14, min_periods=14).mean()
    out = {"days": g.index.to_numpy(np.int64)}
    # valeurs au jour de la barre ; on les appliquera aux sessions des jours suivants via searchsorted strict
    out["atr"] = atr.to_numpy(float)
    out["close"] = g["c"].to_numpy(float)
    for L in lengths:
        ema = g["c"].ewm(span=L, adjust=False, min_periods=L).mean()
        out[f"trend{L}"] = np.sign(g["c"] - ema).fillna(0).to_numpy(float)
    return out


def m15_atr_at_m1(m: dict) -> np.ndarray:
    """ATR14 des barres M15 (grille UTC), valeur connue à la clôture de chaque minute M1."""
    blk = m["t"] // 900
    df = pd.DataFrame({"b": blk, "h": m["h"], "l": m["l"], "c": m["c"]})
    g = df.groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    pc = g["c"].shift(1)
    tr = np.maximum(g["h"] - g["l"], np.maximum((g["h"] - pc).abs(), (g["l"] - pc).abs()))
    atr = tr.rolling(14, min_periods=14).mean().to_numpy(float)
    end = (g.index.to_numpy(np.int64) + 1) * 900          # instant où la barre M15 est connue
    known = m["t"] + 60                                   # clôture de la minute
    j = np.searchsorted(end, known, side="right") - 1     # dernière M15 terminée à la clôture de la minute
    out = np.full(len(known), np.nan)
    ok = j >= 0
    out[ok] = atr[j[ok]]
    return out


def sessions(m: dict, ic: dict) -> pd.DataFrame:
    tz, hh, mm = ic["session"]
    t0 = pd.Timestamp(int(m["t"][0]), unit="s")
    t1 = pd.Timestamp(int(m["t"][-1]), unit="s")
    rows = []
    ch, cm = ic["close_ny"]
    for d in pd.date_range(t0.normalize() - pd.Timedelta(days=1), t1.normalize() + pd.Timedelta(days=1), freq="D"):
        if d.dayofweek >= 5:
            continue
        op = pd.Timestamp(d.year, d.month, d.day, hh, mm, tz=tz)
        op_ny = op.tz_convert("America/New_York")
        cl = op_ny.normalize().replace(hour=ch, minute=cm)
        cl = pd.Timestamp(cl.year, cl.month, cl.day, ch, cm, tz="America/New_York")
        if cl <= op_ny:
            cl = (op_ny.normalize() + pd.Timedelta(days=1))
            cl = pd.Timestamp(cl.year, cl.month, cl.day, ch, cm, tz="America/New_York")
        rows.append((int(op.tz_convert("UTC").value // 10**9), int(cl.tz_convert("UTC").value // 10**9), d.date()))
    s = pd.DataFrame(rows, columns=["open", "close", "local_date"])
    s = s[(s["open"] >= m["t"][0]) & (s["close"] <= m["t"][-1])].reset_index(drop=True)
    return s


def prepare(symbol: str) -> dict:
    ic = C.INSTRUMENTS[symbol]
    m = load(symbol)
    sc, prof = spread_cost(m, ic)
    d1 = d1_features(m, C.GRID_VALUES["L"])
    s = sessions(m, ic)
    od = s["open"].to_numpy(np.int64) // 86400
    # jour D1 le plus récent strictement antérieur au jour UTC de l'ouverture (terminé à 00:00 UTC <= ouverture)
    j = np.searchsorted(d1["days"], od, side="left") - 1
    ok = j >= 0
    jj = np.where(ok, j, 0)
    s["atr_d1"] = np.where(ok, d1["atr"][jj], np.nan)
    for L in C.GRID_VALUES["L"]:
        s[f"trend{L}"] = np.where(ok, d1[f"trend{L}"][jj], 0.0)
    s["i_open"] = np.searchsorted(m["t"], s["open"].to_numpy(np.int64), side="left")
    price = m["c"]
    if ic["quote"] == "JPY":
        comm = ic["comm_usd_lot_side"] / 100_000 * price
    else:
        comm = np.full(len(price), ic["comm_usd_lot_side"] / 100_000) + ic["comm_bps_side"] * 1e-4 * price
    if "slip_bps" in ic:
        slip = ic["slip_bps"] * 1e-4 * price
        sslip = ic["stop_slip_bps"] * 1e-4 * price
    else:
        slip = np.full(len(price), ic["slip_px"])
        sslip = np.full(len(price), ic["stop_slip_px"])
    return dict(symbol=symbol, ic=ic, t=m["t"], o=m["o"], h=m["h"], l=m["l"], c=m["c"], sc=sc, profile=prof,
                atr15=m15_atr_at_m1(m), comm=comm, slip=slip, sslip=sslip, sess=s, spec=m["spec"])
