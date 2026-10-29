import numpy as np
import pandas as pd
import pytest

from eurusd_rl.data import ROOT, load_config, load_m1
from eurusd_rl.market import Market


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def real_m1(cfg):
    path = (ROOT / cfg["data"]["path"]).resolve()
    if not path.exists():
        pytest.skip("données EURUSD absentes")
    return load_m1(cfg)


def toy_market(bars, spread=0.00002, n_days=1, step=5, stop=0.0010, last_force=True):
    """Marché minimal : `bars` = liste (o, h, l, c) M1 par jour ; décisions toutes les `step` barres.
    La dernière décision de chaque jour est la clôture forcée (à l'ouverture de la dernière barre)."""
    O, H, L, C, day = [], [], [], [], []
    for dd in range(n_days):
        for (o, h, l, c) in bars:
            O.append(o); H.append(h); L.append(l); C.append(c); day.append(dd)
    O, H, L, C = (np.asarray(x, float) for x in (O, H, L, C))
    day = np.asarray(day, np.int64)
    n_per = len(bars)
    dec_m1, dec_day = [], []
    for dd in range(n_days):
        base = dd * n_per
        for j in range(0, n_per - 1, step):
            dec_m1.append(base + j); dec_day.append(dd)
        dec_m1.append(base + n_per - 1); dec_day.append(dd)          # décision finale = clôture forcée
    dec_m1 = np.asarray(dec_m1, np.int64); dec_day = np.asarray(dec_day, np.int64)
    is_last = np.r_[dec_day[1:] != dec_day[:-1], True]
    jf = np.array([dd * n_per + n_per - 1 for dd in dec_day], np.int64)
    jn = np.where(is_last, jf, np.r_[dec_m1[1:], 0])
    n = len(dec_m1)
    t = pd.date_range("2025-03-03 09:00", periods=n, freq="5min").values
    return Market(o=O, h=H, l=L, c=C, spread=np.full(len(O), spread), m1_day=day,
                  t_server_min=np.arange(len(O)), dec_m1=dec_m1, dec_next_m1=jn, dec_force_m1=jf,
                  dec_day=dec_day, dec_week=np.zeros(n, np.int64), dec_can_enter=~is_last,
                  dec_is_last=is_last, dec_stop_dist=np.full(n, stop), dec_t_utc=t, dec_t_server=t,
                  seq_feats={"M5": np.zeros((10, 6), np.float32), "H1": np.zeros((10, 6), np.float32)},
                  seq_idx={"M5": np.zeros(n, np.int64), "H1": np.zeros(n, np.int64)},
                  ctx=np.zeros((n, 11), np.float32))
