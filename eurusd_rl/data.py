"""Chargement de l'historique M1 EURUSD (FTMO MT5) et conversion d'horaires.

Format d'entrée : npz compressé xz produit sans perte depuis l'export MT5 (prix BID entiers en points).
Convention serveur FTMO vérifiée (pic NFP à 15:30 serveur chaque année 2015-2026) :
heure serveur = heure de New York + 7 h ; donc UTC = serveur - 2 h (NY heure normale) ou - 3 h (NY heure d'été).
Sortie : DataFrame indexé par position, colonnes
  t_server, t_utc (datetime64[ns] naïfs), open/high/low/close (BID, float), tick_volume, spread_obs_pts (int).
"""
from __future__ import annotations

import hashlib
import io
import lzma
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path(__file__).parent / "config" / "default.yaml"
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def server_to_utc(t_server: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Heure serveur (NY + 7 h) -> UTC. Le statut DST est lu sur l'heure de New York correspondante.
    NY = serveur - 7 h ; on localise cette heure dans America/New_York puis on convertit en UTC."""
    t_server = pd.DatetimeIndex(t_server)
    ny_wall = t_server - pd.Timedelta(hours=7)
    # Les bascules DST ont lieu le dimanche à 02:00 NY, marché fermé : aucune heure ambiguë ou inexistante.
    loc = ny_wall.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise")
    return loc.tz_convert("UTC").tz_localize(None)


def load_m1(cfg: dict, verify_hash: bool = True) -> pd.DataFrame:
    path = (ROOT / cfg["data"]["path"]).resolve()
    if verify_hash:
        got = sha256_file(path)
        if got != cfg["data"]["sha256"]:
            raise ValueError(f"Hash des données inattendu : {got}")
    raw = lzma.decompress(path.read_bytes())
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    tm = np.cumsum(z["t_d"].astype(np.int64))              # minutes epoch, heure serveur
    c = np.cumsum(z["c_d"].astype(np.int64))
    h = c + z["h_c"].astype(np.int64)
    l = c - z["c_l"].astype(np.int64)
    o = c + z["o_c"].astype(np.int64)
    pt = cfg["data"]["point"]
    t_server = pd.to_datetime(tm * 60, unit="s")
    df = pd.DataFrame({
        "t_server": t_server.values,
        "open": o * pt, "high": h * pt, "low": l * pt, "close": c * pt,
        "o_pts": o, "h_pts": h, "l_pts": l, "c_pts": c,
        "tick_volume": z["tv"].astype(np.int64),
        "spread_obs_pts": z["sp"].astype(np.int64),
    })
    drop = pd.to_datetime(cfg["data"].get("drop_server_days", []))
    if len(drop):
        df = df[~df["t_server"].dt.normalize().isin(drop)]
    df = df.reset_index(drop=True)
    df.insert(1, "t_utc", server_to_utc(pd.DatetimeIndex(df["t_server"])).values)
    return df


def split_mask(t_utc: pd.Series | pd.DatetimeIndex, rng: list[str]) -> np.ndarray:
    t = pd.DatetimeIndex(t_utc)
    a = pd.Timestamp(rng[0])
    b = pd.Timestamp(rng[1]) + pd.Timedelta(days=1)
    return np.asarray((t >= a) & (t < b))
