"""Tests des setups H1/H3 sur une séance synthétique."""
import numpy as np
import pandas as pd

from v2 import intraday


def _P(n=400, pc=100.0):
    t = np.arange(n, dtype=np.int64) * 60
    o = np.full(n, 101.0); h = np.full(n, 101.2); l = np.full(n, 100.8); c = np.full(n, 101.0)
    sess = pd.DataFrame({"open": [0], "close": [390 * 60], "pc": [pc], "atr_d1": [2.0], "i_open": [0]})
    z = np.zeros(n)
    return dict(symbol="X", t=t, o=o, h=h, l=l, c=c, sc=z, comm=z, slip=z, sslip=z, sess=sess)


def test_h1_long_when_first_half_hour_up():
    P = _P()
    P["o"][360:] = 102.0; P["c"][360:] = 102.0; P["h"][360:] = 102.1; P["l"][360:] = 101.9
    P["o"][388:] = 103.0
    r = intraday.run(P, "H1", dict(thr=0.0, E=30, m=0.25))["trades"]
    # r1 = 101/100 - 1 > 0 -> long à l'ouverture de la minute 360 (102), stop 102 - 0.5 ; sortie minute 388 à 103
    assert len(r) == 1 and r["dir"][0] == 1 and r["entry_t"][0] == 360 * 60
    assert abs(r["r_net"][0] - (103 - 102) / 0.5) < 1e-9


def test_h1_threshold_filters_small_moves():
    P = _P()
    assert len(intraday.run(P, "H1", dict(thr=0.6, E=30, m=0.25))["trades"]) == 0   # |101-100| = 1 < 0.6 x 2


def test_h3_fades_gap_and_takes_profit_at_previous_close():
    P = _P()                                   # ouverture 101 > clôture veille 100 : gap +0,5 ATR -> short
    P["o"][5:] = 100.5; P["h"][5:] = 100.6; P["l"][5:] = 99.9; P["c"][5:] = 100.0
    r = intraday.run(P, "H3", dict(lo=0.2, hi=1.2, X=60))["trades"]
    # entrée short à l'ouverture de la minute 1 (101), stop 102, objectif 100 atteint minute 5 -> +1 R
    assert len(r) == 1 and r["dir"][0] == -1
    assert intraday.REASONS[r["why"][0]] == "tp" and abs(r["r_net"][0] - 1.0) < 1e-9
