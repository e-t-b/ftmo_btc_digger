"""Moteur swing en barres journalières (positions sur plusieurs jours, swaps, règles FTMO).

Conventions :
- barres D1 MT5 (heure serveur ; un jour serveur = 17:00 → 17:00 New York, ce qui coïncide avec le reset FTMO
  à quelques heures près : approximation documentée) ;
- signal sur la clôture du jour t, exécution à l'ouverture du jour t+1 (prix BID ; ask = bid + spread) ;
- stop vérifié sur le haut/bas du jour ; si l'ouverture franchit le stop (gap), sortie à l'ouverture ;
- swap débité chaque nuit où la position est ouverte (triple le jour `swap_rollover3days`) ;
- spread : max(spread de la barre, plancher = médiane récente du spread en fraction du prix) ; glissement 1/2 spread."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

D1_DIR = Path(__file__).resolve().parent.parent / "data" / "d1"
COMM_FRAC = {"fx": 3e-5, "crypto": 3.25e-4}          # commission FTMO par côté, en fraction du prix


def _klass(sym):
    if sym in ("BTCUSD", "ETHUSD"):
        return "crypto"
    if len(sym) == 6 and sym.isalpha() and not sym.startswith("XA"):
        return "fx"
    return "other"


def load(sym: str) -> dict:
    df = pd.read_csv(D1_DIR / f"{sym}_D1.csv")
    spec = json.loads((D1_DIR / f"{sym}_spec.json").read_text())
    df["date"] = pd.to_datetime(df["time"], unit="s").dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df = df[df["date"].dt.dayofweek < 5].reset_index(drop=True)           # barres du dimanche (ouverture) écartées
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    sp = df["spread_px"].to_numpy(float)
    recent = sp[-750:]
    floor_frac = float(np.median(recent[recent > 0] / c[-750:][recent > 0])) if (recent > 0).any() else 1e-4
    cfg = C.INSTRUMENTS.get(sym) or {}
    if cfg.get("floor_px"):
        floor_frac = max(floor_frac, cfg["floor_px"] / c[-1])
    if sym in ("BTCUSD", "ETHUSD"):
        floor_frac = max(floor_frac, 1e-4)
    comm = COMM_FRAC.get(_klass(sym), 0.0)
    spread = np.maximum(sp, floor_frac * c) + 2 * comm * c     # commission aller (+ retour) intégrée à l'écart
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    return dict(sym=sym, date=df["date"].to_numpy(), o=o, h=h, l=l, c=c, spread=spread, tr=tr, spec=spec,
                floor_frac=floor_frac)


def swap_per_unit_day(D: dict, i: int, d: int) -> float:
    """Swap d'une nuit, en unités de prix par unité de contrat (négatif = coût), pour une position de sens d."""
    s = D["spec"]
    v = s["swap_long"] if d == 1 else s["swap_short"]
    mode = s["swap_mode"]
    px = D["c"][i]
    if mode == 1:                       # points
        return v * s["point"]
    if mode in (5, 6):                  # intérêt annuel en % du prix
        return v / 100.0 / 360.0 * px
    if mode in (2, 3, 4):               # monnaie par lot : approximé en unités de prix (quote USD)
        return v / s["contract_size"]
    return 0.0


def triple_day(D: dict, i: int) -> int:
    """Nombre de nuits facturées pour la nuit qui suit la barre i (3 le jour du triple swap)."""
    wd = (pd.Timestamp(D["date"][i]).dayofweek + 1) % 7      # 0 = dimanche … 6 = samedi (convention MT5)
    return 3 if wd == D["spec"].get("swap_rollover3days", 3) else 1


@dataclass
class Trade:
    sym: str
    d: int
    i_entry: int
    entry: float
    stop: float
    R: float
    i_exit: int = -1
    exit: float = np.nan
    swap: float = 0.0
    why: str = ""


def simulate(D: dict, signal_fn, exit_fn, stop_fn, start=None, end=None, cost_mult=1.0, swap_mult=1.0) -> list[Trade]:
    """signal_fn(D, i) -> +1/-1/0 sur la clôture du jour i ; exit_fn(D, i, trade) -> True pour sortir à l'ouverture
    du jour suivant ; stop_fn(D, i, d, entry) -> prix du stop initial ; le stop peut être relevé par exit_fn (trade.stop)."""
    n = len(D["c"])
    i0 = 0 if start is None else int(np.searchsorted(D["date"], np.datetime64(start)))
    i1 = n if end is None else int(np.searchsorted(D["date"], np.datetime64(end)))
    trades, pos = [], None
    for i in range(max(i0, 1), min(i1, n - 1)):
        # 1) position ouverte : stop intrajournalier sur la barre i
        if pos is not None and i > pos.i_entry - 1:
            sp = D["spread"][i] * cost_mult
            if pos.d == 1:
                if D["o"][i] <= pos.stop:
                    pos.exit, pos.why = D["o"][i] - sp / 2, "gap"
                elif D["l"][i] <= pos.stop:
                    pos.exit, pos.why = pos.stop - sp / 2, "stop"
            else:
                if D["o"][i] + sp >= pos.stop:
                    pos.exit, pos.why = D["o"][i] + sp * 1.5, "gap"
                elif D["h"][i] + sp >= pos.stop:
                    pos.exit, pos.why = pos.stop + sp / 2, "stop"
            if pos.why:
                pos.i_exit = i
                trades.append(pos)
                pos = None
            else:
                pos.swap += swap_per_unit_day(D, i, pos.d) * triple_day(D, i) * swap_mult
        # 2) décision sur la clôture de i, exécution à l'ouverture de i+1
        if pos is not None and exit_fn(D, i, pos):
            j = i + 1
            sp = D["spread"][j] * cost_mult
            pos.exit = D["o"][j] - sp / 2 if pos.d == 1 else D["o"][j] + sp * 1.5
            pos.i_exit, pos.why = j, "signal"
            trades.append(pos)
            pos = None
            continue
        if pos is None:
            s = signal_fn(D, i)
            if s:
                j = i + 1
                sp = D["spread"][j] * cost_mult
                e = D["o"][j] + sp * 1.5 if s == 1 else D["o"][j] - sp / 2
                st = stop_fn(D, i, s, e)
                R = s * (e - st)
                if R > 0:
                    pos = Trade(D["sym"], s, j, e, st, R)
    if pos is not None:
        k = min(i1, n) - 1
        pos.exit, pos.i_exit, pos.why = D["c"][k], k, "fin"
        trades.append(pos)
    return trades


def to_frame(trades: list[Trade], D: dict) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["sym", "d", "entry_date", "exit_date", "r_gross", "r_net", "swap_r", "why", "days", "R_frac"])
    rows = []
    for t in trades:
        gross = t.d * (t.exit - t.entry)
        rows.append(dict(sym=t.sym, d=t.d, entry_date=D["date"][t.i_entry], exit_date=D["date"][t.i_exit],
                         r_gross=gross / t.R, r_net=(gross + t.swap) / t.R, swap_r=t.swap / t.R, why=t.why,
                         days=t.i_exit - t.i_entry, R_frac=t.R / t.entry, i_entry=t.i_entry, i_exit=t.i_exit,
                         entry=t.entry, stop=t.stop, R=t.R, dsign=t.d))
    return pd.DataFrame(rows)
