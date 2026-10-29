"""Paramètres figés par le protocole docs/protocole_jalon_A.md (sections 1, 3, 4, 6, 7).
Aucune valeur ne doit être modifiée après avoir vu un résultat sans commit séparé et justifié."""
from __future__ import annotations

from itertools import product

INSTRUMENTS = {
    # session : (fuseau, heure, minute) ; close_ny : clôture forcée (heure, minute) à New York
    # usd_side : +1 si un long sur l'instrument est un long dollar, -1 si c'est un short dollar, 0 sinon
    "EURUSD": dict(session=("Europe/London", 8, 0), close_ny=(16, 45), usd_side=-1, quote="USD",
                   trust=("2022-01-01", "2024-12-31"), floor_px=2e-5,
                   comm_usd_lot_side=3.5, comm_bps_side=0.0, slip_px=2e-5, stop_slip_px=5e-5,
                   leverage=30.0, first_test="2018-01-01", asset="forex"),
    "USDJPY": dict(session=("Asia/Tokyo", 9, 0), close_ny=(16, 45), usd_side=+1, quote="JPY",
                   trust=("2022-01-01", "2026-12-31"), floor_px=2e-3,
                   comm_usd_lot_side=3.5, comm_bps_side=0.0, slip_px=2e-3, stop_slip_px=5e-3,
                   leverage=30.0, first_test="2018-01-01", asset="forex"),
    "AUDUSD": dict(session=("Asia/Tokyo", 9, 0), close_ny=(16, 45), usd_side=-1, quote="USD",
                   trust=("2022-01-01", "2026-12-31"), floor_px=2e-5,
                   comm_usd_lot_side=3.5, comm_bps_side=0.0, slip_px=2e-5, stop_slip_px=5e-5,
                   leverage=30.0, first_test="2018-01-01", asset="forex"),
    "US100.cash": dict(session=("America/New_York", 9, 30), close_ny=(15, 55), usd_side=0, quote="USD",
                       trust=("2022-01-01", "2026-12-31"), floor_px=0.5,
                       comm_usd_lot_side=0.0, comm_bps_side=0.0, slip_px=0.5, stop_slip_px=1.5,
                       leverage=15.0, first_test="2023-07-01", asset="index"),
    "BTCUSD": dict(session=("America/New_York", 9, 30), close_ny=(16, 45), usd_side=0, quote="USD",
                   trust=("2022-01-01", "2025-12-31"), floor_px=10.0,
                   comm_usd_lot_side=0.0, comm_bps_side=3.25, slip_bps=1.0, stop_slip_bps=3.0,
                   leverage=1.0, first_test="2020-07-01", asset="crypto"),
}

PROFILE_FIT = ("2022-01-01", "2024-06-30")
PROFILE_Q = 0.75
DEV_END = "2024-06-30"
HOLDOUT = ("2024-07-01", "2026-10-28")

ENTRY_WINDOW_MIN = 180
MIN_STOP_COST_MULT = 3.0
OR_MIN_COVERAGE = 0.8

GRID_KEYS = ("N", "L", "lo", "hi", "k")
GRID_VALUES = dict(N=(30, 60), L=(20, 50), lo=(0.0, 0.15), hi=(0.5, 1.0), k=(2.0, 3.5))
GRID = [dict(zip(GRID_KEYS, v)) for v in product(*(GRID_VALUES[k] for k in GRID_KEYS))]

WF_MIN_TRADES = 60
WF_MIN_T = 1.0

ACCOUNT = 10_000.0
RISK = dict(base=0.005, reduced=0.0025, reduce_below=0.96, halt_below=0.93,
            day_stop=0.02, max_open_risk=0.015, max_same_usd_side=2)

FTMO = dict(target1=0.10, target2=0.05, daily_loss=0.05, max_loss=0.10, min_days=4)

CRITERIA = dict(min_trades=300, min_expectancy_r=0.20, min_pf=1.3, max_dd=0.06, min_pass=0.50,
                holdout_min_pf=1.1)


def cfg_tag(c: dict) -> str:
    return f"N{c['N']}_L{c['L']}_lo{c['lo']}_hi{c['hi']}_k{c['k']}"


def neighbours(c: dict) -> list[dict]:
    out = []
    for key in GRID_KEYS:
        for v in GRID_VALUES[key]:
            if v != c[key]:
                n = dict(c)
                n[key] = v
                out.append(n)
    return out
