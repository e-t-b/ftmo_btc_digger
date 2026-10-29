"""Sérialisation sûre, déterminisme d'une seed, reprise exacte d'un entraînement."""
import copy
import pickle

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eurusd_rl import train as T
from eurusd_rl.market import build_market


@pytest.fixture(scope="module")
def small(cfg, real_m1):
    c = copy.deepcopy(cfg)
    c["splits"]["train"] = ["2024-02-01", "2024-03-31"]
    c["splits"]["calibration"] = ["2024-04-01", "2024-04-30"]
    c["execution"]["spread"]["profile_fit"] = ["2023-06-01", "2023-12-31"]
    c["ppo"].update(n_env=4, n_steps=32, minibatch=64, epochs=2, eval_every=2, checkpoint_every=2)
    m1 = real_m1[(real_m1["t_utc"] >= "2023-06-01") & (real_m1["t_utc"] < "2024-05-01")].reset_index(drop=True)
    return c, build_market(c, m1)


def _params(run):
    ck = T.load_ckpt(run / "latest.pt")
    return {k: v.clone() for k, v in ck["model"].items()}, ck


def test_determinism_same_seed(tmp_path, small):
    c, mk = small
    T.train(c, tmp_path / "a", 3, 4, mk=mk, log=lambda *_: None)
    T.train(c, tmp_path / "b", 3, 4, mk=mk, log=lambda *_: None)
    pa, ca = _params(tmp_path / "a")
    pb, cb = _params(tmp_path / "b")
    for k in pa:
        assert torch.equal(pa[k], pb[k]), k
    assert torch.equal(ca["env_F"], cb["env_F"]) and torch.equal(ca["env_I"], cb["env_I"])


def test_different_seed_differs(tmp_path, small):
    c, mk = small
    T.train(c, tmp_path / "a", 3, 2, mk=mk, log=lambda *_: None)
    T.train(c, tmp_path / "b", 4, 2, mk=mk, log=lambda *_: None)
    pa, _ = _params(tmp_path / "a")
    pb, _ = _params(tmp_path / "b")
    assert any(not torch.equal(pa[k], pb[k]) for k in pa)


def test_exact_resume(tmp_path, small):
    c, mk = small
    T.train(c, tmp_path / "full", 5, 4, mk=mk, log=lambda *_: None)
    T.train(c, tmp_path / "part", 5, 2, mk=mk, log=lambda *_: None)
    T.train(c, tmp_path / "part", 5, 4, resume=True, mk=mk, log=lambda *_: None)
    pa, ca = _params(tmp_path / "full")
    pb, cb = _params(tmp_path / "part")
    for k in pa:
        assert torch.equal(pa[k], pb[k]), k
    assert torch.equal(ca["env_I"], cb["env_I"])


def test_checkpoint_loads_weights_only_and_rejects_code(tmp_path, small):
    c, mk = small
    T.train(c, tmp_path / "r", 1, 2, mk=mk, log=lambda *_: None)
    ck = T.load_ckpt(tmp_path / "r" / "latest.pt")
    assert ck["update"] == 2
    # un fichier contenant un objet arbitraire (exécution de code au chargement) doit être refusé

    class Evil:
        def __reduce__(self):
            return (print, ("CODE EXÉCUTÉ",))

    bad = tmp_path / "evil.pt"
    with open(bad, "wb") as f:
        pickle.dump({"model": Evil()}, f)
    with pytest.raises(Exception):
        T.load_ckpt(bad)
