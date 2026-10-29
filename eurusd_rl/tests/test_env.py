"""Mécanique de l'environnement : remplissage bid/ask, sizing, SL/TP conservatif, limites, clôture intraday."""
import copy

import numpy as np
import pytest

from eurusd_rl import env as E
from eurusd_rl.tests.conftest import toy_market

P0 = 1.10000
FLAT = (P0, P0 + 0.0001, P0 - 0.0001, P0)


def run(mk, cfg, actions, log=50):
    v = E.VecEnv(mk, cfg, 1, 0, len(mk.dec_m1), sequential=True, log_capacity=log)
    rs, viols = [], []
    for a in actions:
        r, d, vi, _ = v.step(np.array([a]))
        rs.append(r[0]); viols.append(vi[0])
        if d[0]:
            break
    trades = v.log[0, :v.nlog[0]]
    return v, np.array(rs), trades, viols


def k_of(cfg):
    return E.make_consts(cfg)


def test_long_take_profit_exact(cfg):
    bars = [FLAT] * 3 + [(P0, 1.1050, P0 - 0.0001, 1.1045)] + [FLAT] * 6
    mk = toy_market(bars)
    v, rs, tr, _ = run(mk, cfg, [1, 0, 0])
    assert len(tr) == 1
    entry = P0 + 0.00002 + 0.00002                    # ask + glissement
    assert tr[0, 4] == pytest.approx(entry)
    assert tr[0, 6] == pytest.approx(0.5)             # 50 $ / (10 pips x 100 000)
    assert tr[0, 8] == E.R_TP
    assert tr[0, 5] == pytest.approx(entry + 4 * 0.0010)
    assert tr[0, 10] == pytest.approx(0.5 * 1e5 * 0.004 - 2 * 0.5 * 3.5)
    # la somme des récompenses = résultat net en R nominal
    assert rs.sum() == pytest.approx(tr[0, 10] / 50.0)


def test_stop_and_tp_same_bar_counts_as_stop(cfg):
    bars = [FLAT] * 2 + [(P0, 1.1100, 1.0950, P0)] + [FLAT] * 7
    tr = run(toy_market(bars), cfg, [1, 0, 0])[2]
    assert tr[0, 8] == E.R_STOP
    stop = P0 + 0.00004 - 0.0010
    assert tr[0, 5] == pytest.approx(stop - 0.00005)


def test_gap_through_stop_fills_at_open(cfg):
    gap = 1.0950
    bars = [FLAT] * 2 + [(gap, gap + 0.0001, gap - 0.0001, gap)] + [FLAT] * 7
    tr = run(toy_market(bars), cfg, [1, 0, 0])[2]
    assert tr[0, 8] == E.R_STOP and tr[0, 5] == pytest.approx(gap - 0.00005)


def test_short_uses_ask_for_exit(cfg):
    sp = 0.00002
    bars = [FLAT] * 2 + [(P0, P0 + 0.0001, 1.0950, 1.0960)] + [FLAT] * 7
    tr = run(toy_market(bars, spread=sp), cfg, [2, 0, 0])[2]
    entry = P0 - 0.00002
    assert tr[0, 4] == pytest.approx(entry)
    # TP short : ask bas = 1.0950 + spread <= entry - 0.004
    assert tr[0, 8] == E.R_TP and tr[0, 5] == pytest.approx(entry - 0.004)
    # un short dont seul le BID touche le stop mais pas l'ASK n'est pas stoppé
    stop = entry + 0.0010
    bars2 = [FLAT] * 2 + [(P0, stop - sp / 2, P0 - 0.0001, P0)] + [FLAT] * 7
    tr2 = run(toy_market(bars2, spread=sp), cfg, [2, 0, 0])[2]
    assert tr2[0, 8] == E.R_STOP      # ask haut = stop - sp/2 + sp >= stop -> stoppé
    bars3 = [FLAT] * 2 + [(P0, stop - sp - 0.00001, P0 - 0.0001, P0)] + [FLAT] * 7
    tr3 = run(toy_market(bars3, spread=sp), cfg, [2, 0, 0])[2]
    assert tr3[0, 8] == E.R_FORCE


def test_forced_close_and_no_overnight(cfg):
    mk = toy_market([FLAT] * 10, n_days=2)
    v, rs, tr, _ = run(mk, cfg, [1] + [0] * 30)
    assert tr[0, 8] == E.R_FORCE
    assert mk.m1_day[int(tr[0, 2])] == 0          # sortie le jour même
    assert v.I[0, E.I_POS] == 0


def test_close_action(cfg):
    tr = run(toy_market([FLAT] * 15), cfg, [1, 3, 0])[2]
    assert tr[0, 8] == E.R_CLOSE
    assert tr[0, 5] == pytest.approx(P0 - 0.00002)        # bid - glissement à l'ouverture suivante


def test_sizing_and_rejections(cfg):
    k = k_of(cfg)
    assert E.lots_for(k, 0.0010, 1.1, 10000.0) == pytest.approx(0.5)
    assert E.lots_for(k, 0.0007, 1.1, 10000.0) == pytest.approx(0.71)      # arrondi à la baisse
    assert E.lots_for(k, 0.0001, 1.1, 10000.0) == 0.0                     # 5 lots : marge 18 333 $ > 90 % équité
    assert E.lots_for(k, 0.6, 1.1, 10000.0) == 0.0                        # < 0,01 lot
    assert E.lots_for(k, 0.0, 1.1, 10000.0) == 0.0


def test_entry_mask_respects_limits(cfg):
    c2 = copy.deepcopy(cfg)
    c2["internal_risk"]["max_trades_per_day"] = 10
    # chaque trade : entrée puis stop immédiat (-~1,08 R) ; 3 pertes -> -1,6 % < -1,5 % -> arrêt du jour
    lose = [FLAT, (P0, P0 + 0.0001, 1.0980, 1.0985), FLAT, FLAT, FLAT]
    mk = toy_market(lose * 6, step=5)
    v = E.VecEnv(mk, c2, 1, 0, len(mk.dec_m1), sequential=True, log_capacity=50)
    n_entries = 0
    for _ in range(len(mk.dec_m1)):
        m = v.masks()[0]
        a = 1 if m[1] else 0
        n_entries += a
        _, d, _, _ = v.step(np.array([a]))
        if d[0]:
            break
    assert n_entries == 3 and v.I[0, E.I_HDAY] == 1


def test_max_trades_per_day(cfg):
    mk = toy_market([FLAT] * 60, step=5)
    v = E.VecEnv(mk, cfg, 1, 0, len(mk.dec_m1), sequential=True, log_capacity=50)
    n = 0
    for _ in range(len(mk.dec_m1)):
        m = v.masks()[0]
        a = 1 if m[1] else (3 if m[3] else 0)
        n += a == 1
        _, d, _, _ = v.step(np.array([a]))
        if d[0]:
            break
    assert n == cfg["internal_risk"]["max_trades_per_day"]


def test_ftmo_daily_violation_on_gap(cfg):
    gap = 1.0500                                     # gap de 500 pips : perte >> 5 %
    bars = [FLAT] * 2 + [(gap, gap, gap, gap)] + [FLAT] * 7
    v, rs, tr, viols = run(toy_market(bars), cfg, [1, 0, 0])
    i = viols.index(1)
    assert rs[i] < -cfg["reward"]["violation_penalty_r"]
    assert v.I[0, E.I_VIOL] == 1


def test_invalid_action_counted(cfg):
    v = run(toy_market([FLAT] * 10), cfg, [3, 0])[0]       # CLOSE sans position
    assert v.I[0, E.I_NINVALID] == 1


# ------------------------------------------------------------------ mode swing
def _overnight(mk):
    import dataclasses
    n = len(mk.dec_m1)
    force = np.zeros(n, bool); force[-1] = True
    jn = np.r_[mk.dec_m1[1:], mk.dec_force_m1[-1]]
    can = np.ones(n, bool); can[mk.dec_is_last] = False
    return dataclasses.replace(mk, dec_force=force, dec_next_m1=jn, dec_can_enter=can)


def test_swing_holds_overnight_and_charges_swap(cfg):
    mk = _overnight(toy_market([FLAT] * 10, n_days=3))
    mk.m1_wday[:] = np.repeat([0, 1, 2], 10)              # lundi, mardi, mercredi
    v, rs, tr, _ = run(mk, cfg, [1] + [0] * 40)
    assert len(tr) == 1 and tr[0, 8] == E.R_FORCE          # tenue jusqu'à la fin des données, pas de clôture le soir
    assert mk.m1_day[int(tr[0, 2])] == 2
    lots = 0.5
    swap = 2 * lots * cfg["data"]["swap_long_pts"]          # deux nuits ordinaires (lundi, mardi), en USD
    expected = (tr[0, 5] - tr[0, 4]) * lots * 1e5 - 2 * lots * 3.5 + swap
    assert tr[0, 10] == pytest.approx(expected)
    assert v.F[0, E.S_BAL] - 10000 == pytest.approx(expected)


def test_swing_triple_swap_on_wednesday_and_short_credit(cfg):
    mk = _overnight(toy_market([FLAT] * 10, n_days=2))
    mk.m1_wday[:] = np.repeat([2, 3], 10)                 # mercredi -> jeudi
    tr = run(mk, cfg, [2] + [0] * 30)[2]
    lots = 0.5
    swap = 3 * lots * cfg["data"]["swap_short_pts"]
    expected = (tr[0, 4] - tr[0, 5]) * lots * 1e5 - 2 * lots * 3.5 + swap
    assert tr[0, 10] == pytest.approx(expected)


def test_swing_weekend_gap_through_stop(cfg):
    gap = 1.0950
    bars_fri = [FLAT] * 10
    mk = _overnight(toy_market(bars_fri, n_days=2))
    # lundi : ouverture en gap sous le stop
    j_mon = 10
    mk.o[j_mon] = gap; mk.h[j_mon] = gap + 0.0001; mk.l[j_mon] = gap - 0.0001; mk.c[j_mon] = gap
    mk.m1_wday[:] = np.repeat([4, 0], 10)
    tr = run(mk, cfg, [1] + [0] * 30)[2]
    assert tr[0, 8] == E.R_STOP and tr[0, 5] == pytest.approx(gap - 0.00005)


# ------------------------------------------------------------------ étiquettes = environnement
@pytest.mark.slow
def test_labels_match_environment(cfg, real_m1):
    import copy
    from eurusd_rl.labels import make_labels
    from eurusd_rl.market import build_market
    sub = real_m1[(real_m1["t_utc"] >= "2024-01-01") & (real_m1["t_utc"] < "2024-03-01")].reset_index(drop=True)
    c = copy.deepcopy(cfg)
    c["execution"]["spread"]["profile_fit"] = ["2024-01-01", "2024-02-29"]
    c["internal_risk"]["daily_halt"] = c["internal_risk"]["weekly_halt"] = c["internal_risk"]["total_halt"] = 1.0
    mk = build_market(c, sub)
    rng = np.random.default_rng(0)
    cand = rng.choice(np.flatnonzero(mk.dec_can_enter), 60, replace=False)
    lab = make_labels(mk, c, cand, c["trade"]["take_profit_r"])
    for n, d in enumerate(cand):
        for side, a, key in ((1, 1, "long"), (-1, 2, "short")):
            v = E.VecEnv(mk, c, 1, int(d), len(mk.dec_m1), sequential=True, log_capacity=5)
            v.step(np.array([a]))
            while v.nlog[0] == 0:
                _, done, _, _ = v.step(np.array([0]))
                if done[0]:
                    break
            tr = v.log[0, 0]
            assert int(tr[2]) == lab[f"exit_m1_{key}"][n]                    # même barre de sortie
            # même R au facteur d'arrondi de lot près (R env = net / risque nominal ; R label = net / (sd x lots))
            r_env_unit = tr[10] / (tr[6] * 1e5 * tr[7])
            assert r_env_unit == pytest.approx(lab[f"R_{key}"][n], abs=1e-9)
