"""Notation des avis du test en temps réel : cohérence avec les étiquettes de recherche."""
import numpy as np
import pandas as pd
import pytest

from eurusd_rl.labels import make_labels
from eurusd_rl.market import build_market
from forward_llm.score_forward import score, summarize, utc_to_server


@pytest.mark.slow
def test_forward_scoring_matches_labels(cfg, real_m1):
    sub = real_m1[(real_m1["t_utc"] >= "2023-01-01") & (real_m1["t_utc"] < "2023-03-01")].reset_index(drop=True)
    # avis fictifs : 07:15 UTC (08:15 Paris en hiver), un achat, une vente, un « none », un avis tardif
    dec = pd.DataFrame({"decision_time_utc": pd.to_datetime(["2023-02-06 07:15", "2023-02-07 07:15", "2023-02-08 07:15",
                                                              "2023-02-09 08:30"]),
                        "direction": ["long", "short", "none", "long"], "confidence": [70, 40, 10, 90],
                        "file": ["a", "b", "c", "d"]})
    sc = score(dec, sub, cfg)
    assert list(sc["status"]) == ["noté", "noté", "noté", "exclu (tardif)"]
    # 07:15 UTC + 5 min = 09:20 serveur en hiver ; l'entrée est la première décision >= 09:20 serveur
    assert pd.Timestamp(sc.loc[0, "entry_time_server"]) == pd.Timestamp("2023-02-06 09:20")
    c2 = dict(cfg); c2 = {**cfg, "execution": {**cfg["execution"], "spread": {**cfg["execution"]["spread"],
                                                                              "profile_fit": ["2023-01-02", "2024-01-02"]}}}
    mk = build_market(c2, sub)
    T = pd.DatetimeIndex(mk.dec_t_server)
    d = np.flatnonzero(T == pd.Timestamp("2023-02-06 09:20"))
    lab = make_labels(mk, c2, d, 2.0)
    assert sc.loc[0, "R_long"] == pytest.approx(lab["R_long"][0], abs=1e-4)
    assert sc.loc[0, "R"] == sc.loc[0, "R_long"] and sc.loc[1, "R"] == sc.loc[1, "R_short"] and sc.loc[2, "R"] == 0.0
    s = summarize(sc)
    assert s["n_directionnels"] == 2 and s["n_notes"] == 3


def test_utc_to_server_dst():
    assert utc_to_server(pd.Timestamp("2023-02-06 07:15")) == pd.Timestamp("2023-02-06 09:15")
    assert utc_to_server(pd.Timestamp("2023-07-06 06:15")) == pd.Timestamp("2023-07-06 09:15")
