"""Données, fuseaux, agrégation causale, coûts, absence de look-ahead."""
import copy
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from eurusd_rl import costs
from eurusd_rl.data import server_to_utc
from eurusd_rl.market import build_market
from eurusd_rl.timeframes import aggregate, last_closed


def test_server_to_utc_dst():
    # hiver : serveur UTC+2 ; été US : UTC+3 ; semaine de décalage mars (US en été, Europe non) : UTC+3
    s = pd.DatetimeIndex(["2025-01-15 15:30", "2025-07-15 15:30", "2025-03-12 15:30", "2025-11-05 15:30"])
    u = server_to_utc(s)
    assert list(u) == list(pd.DatetimeIndex(["2025-01-15 13:30", "2025-07-15 12:30", "2025-03-12 12:30", "2025-11-05 13:30"]))
    # 15:30 serveur = 08:30 New York (NFP) toute l'année
    ny = u.tz_localize("UTC").tz_convert("America/New_York")
    assert all((t.hour, t.minute) == (8, 30) for t in ny)


def test_trading_window_never_contains_prague_midnight(cfg):
    # fenêtre position possible : 02:00 -> 16:45 NY ; minuit Prague ne doit jamais y tomber (jour FTMO = notre jour)
    days = pd.date_range("2015-01-01", "2027-01-01", freq="D")
    pr = pd.DatetimeIndex([pd.Timestamp(d, tz=ZoneInfo("Europe/Prague")) for d in days]).tz_convert("America/New_York")
    mins = pr.hour * 60 + pr.minute
    assert not ((mins >= 2 * 60) & (mins <= 16 * 60 + 45)).any()


def _toy_m1():
    t = pd.date_range("2025-03-03 00:00", periods=60 * 30, freq="1min")
    rng = np.random.default_rng(0)
    c = 1.1 + np.cumsum(rng.normal(0, 1e-4, len(t)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"t_server": t, "t_utc": server_to_utc(t), "open": o, "high": np.maximum(o, c) + 1e-5,
                         "low": np.minimum(o, c) - 1e-5, "close": c, "tick_volume": rng.integers(1, 50, len(t))})


def test_aggregation_matches_manual():
    m1 = _toy_m1()
    m5 = aggregate(m1, "5min")
    g = m1.set_index("t_server").resample("5min", label="left", closed="left")
    ref = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"})
    np.testing.assert_allclose(m5[["open", "high", "low", "close"]].to_numpy(), ref[["open", "high", "low", "close"]].to_numpy())
    np.testing.assert_array_equal(m5["tick_volume"].to_numpy(), ref["tick_volume"].to_numpy())
    assert (m5["end"] - m5["start"] == pd.Timedelta("5min")).all()
    h4 = aggregate(m1, "4h")
    assert set(h4["start"].dt.hour) <= {0, 4, 8, 12, 16, 20}


def test_last_closed_excludes_forming_bar():
    m1 = _toy_m1()
    h1 = aggregate(m1, "1h")
    T = np.array([pd.Timestamp("2025-03-03 10:59"), pd.Timestamp("2025-03-03 11:00")], dtype="datetime64[ns]")
    idx = last_closed(h1["end"].to_numpy(), T)
    assert h1["end"].iloc[idx[0]] <= pd.Timestamp("2025-03-03 10:59")
    assert h1["start"].iloc[idx[0]] == pd.Timestamp("2025-03-03 09:00")   # 10:00-11:00 encore en formation
    assert h1["start"].iloc[idx[1]] == pd.Timestamp("2025-03-03 10:00")


def test_spread_cost_rules():
    t = pd.DatetimeIndex(["2020-06-01 10:00", "2023-06-01 10:00", "2023-06-01 10:01", "2023-06-01 00:02"])
    m1 = pd.DataFrame({"t_server": t, "t_utc": t, "spread_obs_pts": [50, 0, 7, 30]})
    prof = np.full(costs.N_BINS, 2.0)
    prof[0] = 25.0
    s = costs.spread_cost_pts(m1, prof, "2022-01-01", 1.0)
    # 2020 : 50 pts non fiables -> profil ; 2023 : 0 -> profil ; 7 -> observé ; rollover -> profil 25 vs obs 30
    np.testing.assert_allclose(s, [2.0, 2.0, 7.0, 30.0])
    np.testing.assert_allclose(costs.spread_cost_pts(m1, prof, "2022-01-01", 1.0, stress=1.5), [3.0, 3.0, 10.5, 45.0])


@pytest.mark.slow
def test_no_lookahead_on_real_data(cfg, real_m1):
    """Perturber tout ce qui suit l'instant T ne doit changer aucune observation ni aucun stop à des décisions <= T."""
    sub = real_m1[(real_m1["t_utc"] >= "2024-01-01") & (real_m1["t_utc"] < "2024-04-01")].reset_index(drop=True)
    cut = pd.Timestamp("2024-03-01 12:00")
    pert = sub.copy()
    after = pert["t_server"] >= cut
    rng = np.random.default_rng(1)
    shock = 1 + rng.normal(0, 0.01, after.sum())
    for k in ("open", "high", "low", "close"):
        pert.loc[after, k] = pert.loc[after, k] * shock
    pert.loc[after, "high"] = pert.loc[after, ["open", "high", "low", "close"]].max(axis=1)
    pert.loc[after, "low"] = pert.loc[after, ["open", "high", "low", "close"]].min(axis=1)
    pert.loc[after, "tick_volume"] = pert.loc[after, "tick_volume"] * 7
    c2 = copy.deepcopy(cfg)
    c2["execution"]["spread"]["profile_fit"] = ["2024-01-01", "2024-01-31"]   # profil dans la sous-période
    a, b = build_market(c2, sub), build_market(c2, pert)
    assert len(a.dec_m1) == len(b.dec_m1)
    sel = pd.DatetimeIndex(a.dec_t_server) <= cut
    assert sel.sum() > 1000
    np.testing.assert_array_equal(a.ctx[sel], b.ctx[sel])
    np.testing.assert_array_equal(a.dec_stop_dist[sel], b.dec_stop_dist[sel])
    for tf in ("M5", "H1"):
        ia = a.seq_idx[tf][sel]
        np.testing.assert_array_equal(ia, b.seq_idx[tf][sel])
        np.testing.assert_array_equal(a.seq_feats[tf][np.unique(ia)], b.seq_feats[tf][np.unique(ia)])
    # contrôle : la perturbation change bien les décisions postérieures
    assert not np.array_equal(a.ctx[~sel], b.ctx[~sel])


@pytest.mark.slow
def test_real_market_grid(cfg, real_m1):
    mk = build_market(cfg, real_m1[real_m1["t_utc"] < "2016-01-01"].reset_index(drop=True))
    ts = pd.DatetimeIndex(mk.dec_t_server)
    mod = ts.hour * 60 + ts.minute
    assert mod.min() >= 9 * 60 and mod.max() <= 23 * 60 + 45
    ce = mk.dec_can_enter
    assert (mod[ce] < 23 * 60).all()                               # pas d'entrée après 16:00 NY
    assert (mk.dec_m1 <= mk.dec_force_m1).all() and (mk.dec_next_m1 <= mk.dec_force_m1).all()
    # le remplissage commence à T ou après (jamais avant)
    tm = mk.t_server_min[mk.dec_m1]
    assert (tm >= ts.values.astype("datetime64[m]").astype(np.int64)).all()
    # une seule journée par borne de simulation
    assert (mk.m1_day[mk.dec_force_m1] == mk.dec_day).all()
