"""Tests de la mécanique v2 sur données synthétiques à résultat connu."""
import numpy as np
import pandas as pd

from v2 import sim


def _P(prices_h, prices_l, prices_o, prices_c, open_ts, close_ts, trend=1.0, atr=1.0, sc=0.0):
    n = len(prices_o)
    t = open_ts + np.arange(n, dtype=np.int64) * 60
    sess = pd.DataFrame({"open": [open_ts], "close": [close_ts], "i_open": [0], "trend20": [trend],
                         "trend50": [trend], "atr_d1": [atr]})
    z = np.zeros(n)
    return dict(symbol="X", t=t, o=np.array(prices_o, float), h=np.array(prices_h, float), l=np.array(prices_l, float),
                c=np.array(prices_c, float), sc=np.full(n, sc), atr15=np.full(n, 0.05), comm=z, slip=z, sslip=z, sess=sess)


CFG = dict(N=30, L=20, lo=0.0, hi=10.0, k=100.0)   # trailing neutralisé (k énorme)


def _series(n, or_h=101.0, or_l=99.0):
    o = np.full(n, 100.0); h = np.full(n, 100.5); l = np.full(n, 99.5); c = np.full(n, 100.0)
    h[:30] = or_h; l[:30] = or_l
    return o, h, l, c


def test_breakout_long_then_time_exit_no_costs():
    n = 600
    o, h, l, c = _series(n)
    # minute 34 : la barre M5 [30,35) clôture à 101.5 > 101 -> entrée à l'ouverture de la minute 35
    c[34] = 101.5; h[34] = 101.6
    o[35:] = 102.0; h[35:] = 102.2; l[35:] = 101.8; c[35:] = 102.0
    P = _P(h, l, o, c, 0, 400 * 60)
    r = sim.run(P, CFG)["trades"]
    assert len(r["dir"]) == 1 and r["dir"][0] == 1
    assert r["entry_t"][0] == 35 * 60
    # entrée 102, stop 99 -> R = 3 ; sortie temporelle à l'ouverture de la minute 400 = 102 -> 0 R
    assert abs(r["R"][0] - 3.0) < 1e-9
    assert abs(r["r_net"][0] - 0.0) < 1e-9
    assert sim.REASONS[r["why"][0]] == "time"


def test_counter_trend_breakout_ignored():
    n = 600
    o, h, l, c = _series(n)
    c[34] = 101.5
    P = _P(h, l, o, c, 0, 400 * 60, trend=-1.0)
    r = sim.run(P, CFG)["trades"]
    assert len(r["dir"]) == 0


def test_stop_hit_gives_minus_one_r_and_costs_reduce_r():
    n = 600
    o, h, l, c = _series(n)
    c[34] = 101.5
    o[35] = 101.5; h[35] = 101.6; l[35] = 101.4; c[35] = 101.5
    o[36:] = 100.0; h[36:] = 100.1; l[36:] = 98.0; c[36:] = 98.5   # touche le stop 99 sans gap (ouverture 100 > 99)
    P = _P(h, l, o, c, 0, 400 * 60)
    r = sim.run(P, CFG)["trades"]
    assert abs(r["r_net"][0] + 1.0) < 1e-9 and sim.REASONS[r["why"][0]] == "stop"
    # le spread d'entrée est inclus dans R (mesuré depuis l'ask) : toujours -1 R ; la commission passe sous -1 R
    P2 = _P(h, l, o, c, 0, 400 * 60, sc=0.01)
    assert abs(sim.run(P2, CFG)["trades"]["r_net"][0] + 1.0) < 1e-9
    P3 = _P(h, l, o, c, 0, 400 * 60)
    P3["comm"] = np.full(len(o), 0.01)
    r3 = sim.run(P3, CFG)["trades"]
    assert abs(r3["r_net"][0] - (-1.0 - 0.02 / 2.5)) < 1e-9


def test_gap_through_stop_fills_at_open():
    n = 600
    o, h, l, c = _series(n)
    c[34] = 101.5
    o[35] = 101.5; h[35] = 101.6; l[35] = 101.4; c[35] = 101.5
    o[36:] = 97.0; h[36:] = 97.1; l[36:] = 96.9; c[36:] = 97.0
    P = _P(h, l, o, c, 0, 400 * 60)
    r = sim.run(P, CFG)["trades"]
    # entrée 101.5, R = 2.5, sortie 97 -> -1.8 R
    assert sim.REASONS[r["why"][0]] == "gap" and abs(r["r_net"][0] - (97 - 101.5) / 2.5) < 1e-9


def test_no_lookahead_signal_uses_closed_m5_only():
    """Un dépassement intraminute sans clôture M5 au-delà ne déclenche rien."""
    n = 600
    o, h, l, c = _series(n)
    h[31] = 103.0                     # mèche au-dessus de l'OR, clôtures restent à 100
    P = _P(h, l, o, c, 0, 400 * 60)
    r = sim.run(P, CFG)["trades"]
    assert len(r["dir"]) == 0


def test_trailing_locks_profit():
    n = 600
    o, h, l, c = _series(n)
    c[34] = 101.5
    up = np.linspace(101.5, 110.0, 100)
    o[35:135] = up; c[35:135] = up; h[35:135] = up + 0.05; l[35:135] = up - 0.05
    o[135:] = 104.0; h[135:] = 104.1; l[135:] = 103.9; c[135:] = 104.0
    P = _P(h, l, o, c, 0, 400 * 60)
    P["atr15"][:] = 1.0
    cfg = dict(CFG, k=2.0)
    r = sim.run(P, cfg)["trades"]
    # le trailing est à ~110.05 - 2 = 108 ; la chute à 104 sort à l'ouverture (gap) : 104 > stop initial 99
    assert sim.REASONS[r["why"][0]] in ("gap", "trail") and r["r_net"][0] > 0.5


def test_ftmo_rules():
    from v2 import ftmo
    n = np.array([1, 1, 1, 1, 1])
    # +2.5 % par jour (5 R à 0,5 %) : objectif 10 % atteint au 4e jour avec 4 jours de trading -> réussite en 4 jours
    st, d = ftmo._phase(np.full(5, 5.0), np.zeros(5), n, 0.10, np.arange(5), 250)
    assert (st, d) == (1, 4)
    # un jour avec un flottant à -10 R (-5 %) -> échec par perte journalière
    st, d = ftmo._phase(np.array([0.0]), np.array([-10.0]), np.array([1]), 0.10, np.array([0]), 250)
    assert st == -1
    # objectif atteint en 1 jour mais il faut 4 jours de trading
    st, d = ftmo._phase(np.array([25.0, 0.0]), np.zeros(2), np.array([1, 1]), 0.10, np.array([0, 1, 1, 1]), 250)
    assert (st, d) == (1, 4)
