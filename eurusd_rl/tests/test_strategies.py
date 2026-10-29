"""Bibliothèque de stratégies : causalité, VWAP, opening ranges (DST), swings confirmés, FVG."""
import numpy as np
import pandas as pd
import pytest

from eurusd_rl import strategies as S
from eurusd_rl.data import server_to_utc
from eurusd_rl.timeframes import aggregate


@pytest.mark.slow
@pytest.mark.parametrize("scale", ["M5", "H1", "H4"])
def test_library_no_lookahead(real_m1, scale):
    start = "2023-09-01" if scale != "H4" else "2021-01-01"      # H4 : tendance hebdomadaire, il faut de l'historique
    sub = real_m1[(real_m1["t_utc"] >= start) & (real_m1["t_utc"] < "2024-02-01")].reset_index(drop=True)
    cut = pd.Timestamp("2024-01-10 13:00")                     # heure serveur
    pert = sub.copy()
    after = pert["t_server"] >= cut
    rng = np.random.default_rng(5)
    f = 1 + rng.normal(0, 0.004, after.sum())
    for k in ("open", "high", "low", "close"):
        pert.loc[after, k] = pert.loc[after, k] * f
    pert.loc[after, "high"] = pert.loc[after, ["open", "high", "low", "close"]].max(axis=1)
    pert.loc[after, "low"] = pert.loc[after, ["open", "high", "low", "close"]].min(axis=1)
    pert.loc[after, "tick_volume"] *= 3
    a, b = S.compute(sub, scale), S.compute(pert, scale)
    sel = a["bar_end"] <= np.datetime64(cut)
    assert sel.sum() > 1000
    changed_after = False
    for name in a["signals"]:
        for key, va in a["signals"][name].items():
            vb = b["signals"][name][key]
            np.testing.assert_array_equal(va[sel], vb[sel], err_msg=f"{name}.{key} dépend du futur")
            if not np.array_equal(va[~sel], vb[~sel], equal_nan=True):
                changed_after = True
    assert changed_after


def _m1_from(times_server, o, h, l, c, tv=None):
    t = pd.DatetimeIndex(times_server)
    return pd.DataFrame({"t_server": t, "t_utc": server_to_utc(t), "open": o, "high": h, "low": l, "close": c,
                         "tick_volume": np.ones(len(t)) if tv is None else tv})


def test_opening_range_dst():
    # Londres 08:00 locale = 07:00 UTC en été (2025-07-07), 08:00 UTC en hiver (2025-01-13)
    for day, utc_open in (("2025-07-07", 7), ("2025-01-13", 8)):
        t_utc = pd.date_range(f"{day} 06:00", f"{day} 10:00", freq="1min", inclusive="left")
        # serveur = UTC + 2 (hiver) / + 3 (été US)
        off = 3 if day.startswith("2025-07") else 2
        ts = t_utc + pd.Timedelta(hours=off)
        n = len(ts)
        base = np.full(n, 1.1)
        h = base + 0.0001; l = base - 0.0001
        # pic à l'intérieur de la plage d'ouverture (08:10 locale), et un pic AVANT l'ouverture
        i_in = np.flatnonzero(t_utc == pd.Timestamp(f"{day} {utc_open:02d}:10"))[0]
        i_pre = np.flatnonzero(t_utc == pd.Timestamp(f"{day} {utc_open - 1:02d}:30"))[0]
        h[i_in] = 1.1050; h[i_pre] = 1.1100
        m1 = _m1_from(ts, base, h, l, base)
        m5 = aggregate(m1, "5min")
        oh, ol, lm = S._opening_range(m5, "Europe/London", 8 * 60, 30)
        ready = ~np.isnan(oh)
        assert np.allclose(oh[ready], 1.1050)                   # la plage ignore le pic d'avant l'ouverture
        first_ready = pd.DatetimeIndex(m5["t_utc_end"])[ready][0]
        assert first_ready == pd.Timestamp(f"{day} {utc_open:02d}:30")


def test_swings_confirmed_only_after_k_bars():
    h = np.array([1, 2, 3, 9, 3, 2, 1, 1, 1], float)
    l = h - 0.5
    sh1, sh2, sl1, sl2 = S._swings(h, l, 3)
    assert np.isnan(sh1[5])                # le sommet en 3 n'est pas confirmé avant la clôture de 6
    assert sh1[6] == 9 and sh1[8] == 9


def test_fvg_detection_and_reaction():
    # M15 : gap haussier entre la barre 0 (haut 1.1010) et la barre 2 (bas 1.1020), confirmé à la clôture de 2
    h15 = np.array([1.1010, 1.1030, 1.1040, 1.1040]); l15 = np.array([1.1000, 1.1012, 1.1020, 1.1025])
    c15 = np.array([1.1008, 1.1028, 1.1035, 1.1030])
    # 3 barres M5 : avant confirmation (k=1), après (k=2) entrée dans le gap puis clôture au-dessus
    k = np.array([1, 2, 2], np.int64)
    m5l = np.array([1.1015, 1.1030, 1.1018]); m5h = np.array([1.1030, 1.1040, 1.1030]); m5c = np.array([1.1025, 1.1035, 1.1025])
    trend = np.ones(3, np.int64)
    d, st = S._fvg_track(h15, l15, c15, k, m5l, m5h, m5c, trend, 16)
    assert d[0] == 0                       # gap pas encore confirmé
    assert d[1] == 0                       # pas d'entrée dans le gap
    assert d[2] == 1 and st[2] == pytest.approx(1.1010)


def test_vwap_and_signal_fields_shapes(real_m1):
    sub = real_m1[(real_m1["t_utc"] >= "2024-03-01") & (real_m1["t_utc"] < "2024-05-01")].reset_index(drop=True)
    lib = S.compute(sub)
    n = len(lib["m5_end"])
    for name in S.DIRECTIONAL:
        s = lib["signals"][name]
        d, st = s["dir"], s["stop_px"]
        assert len(d) == n and set(np.unique(d)) <= {-1, 0, 1}
        act = d != 0
        assert act.sum() > 0, name
        # l'invalidation est toujours du bon côté de la clôture du signal
        assert np.all((d[act] == 1) & (st[act] < lib["close"][act]) | (d[act] == -1) & (st[act] > lib["close"][act])), name
        assert np.all(np.isnan(st[~act]))


@pytest.mark.slow
def test_signal_fires_once_and_respects_staleness(cfg, real_m1):
    from eurusd_rl.market import build_market
    sub = real_m1[(real_m1["t_utc"] >= "2023-01-01") & (real_m1["t_utc"] < "2024-01-01")].reset_index(drop=True)
    mk = build_market(cfg, sub)
    lib = S.compute(sub, "H4")
    D0 = S.to_decisions(lib, mk, max_stale_min=0)
    D60 = S.to_decisions(lib, mk, max_stale_min=60)
    for name in S.names_for("H4"):
        d0, d60 = D0[name]["dir"], D60[name]["dir"]
        assert (np.abs(d60) >= np.abs(d0)).all()          # plus de tolérance => au moins les mêmes signaux
        fired = np.flatnonzero(d60 != 0)
        T = mk.dec_t_server[fired]
        k = np.searchsorted(lib["bar_end"], T, side="right") - 1
        assert len(np.unique(k)) == len(k)                 # chaque barre d'entrée émet au plus une fois
        lag = (T - lib["bar_end"][k]).astype("timedelta64[m]").astype(int)
        assert (lag >= 0).all() and (lag <= 60).all()
