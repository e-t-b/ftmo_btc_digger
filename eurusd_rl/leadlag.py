"""Étape 1, partie 2a — Avance-retard : un autre instrument annonce-t-il le prochain mouvement d'EURUSD ?

Pré-enregistré le 2026-10-29 à 21:22 (Paris), avant toute mesure.
Instruments : GBPUSD (depuis 2015), US500.cash et GER40.cash (depuis fin 2017), même serveur FTMO, prix BID M1.
Grille commune : minutes serveur présentes pour EURUSD ET l'instrument ; heures 23:00-01:00 serveur exclues (rollover).
Pour chaque horizon h ∈ {1, 5, 15, 60} minutes, à chaque instant T multiple de h :
  rX  = rendement de X sur [T - h, T]   (connu à T)
  rE  = rendement d'EURUSD sur [T, T + h] en pips (futur)
  L1 (tous les instants)        : occurrence = sign(rX) x rE                    -> 3 instruments x 4 horizons = 12 tests
  L2 (grands mouvements de X)   : idem, seulement si |rX| >= 2 écarts-types de rX sur les 20 derniers jours
                                   (écart-type calculé sur le passé uniquement)  -> 12 tests
Le signe attendu n'est pas imposé (test bilatéral). Pour les indices, une hausse des actions (« risk-on ») est
classiquement associée à un dollar plus faible, mais on ne présuppose rien.
Critère « exploitable » : |moyenne| >= 2,0 pips par occurrence, |t| >= 3,1 (Bonferroni à 5 % sur 24 tests),
même signe >= 70 % des années. Période : entraînement ; signe vérifié sur la calibration. 2026 non lu.
"""
from __future__ import annotations

import io
import json
import lzma
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, load_config, load_m1, server_to_utc, sha256_file
from .predictability import summarize

PIP = 1e-4
THRESH = {"min_pips": 2.0, "min_t": 3.1, "min_year_share": 0.70}
INSTR = {  # fichier, sha256, point
    "GBPUSD": ("GBPUSD_M1.npz.xz", "0f0056cfe7ccd78d5136dd24275413198c6c3a80f1ded886a90e6c2de3059279", 1e-5),
    "US500.cash": ("US500.cash_M1.npz.xz", "1a8857e3d1db3d71fae4c0715ede307c85415326c8f1ba1d2666426201337320", 0.01),
    "GER40.cash": ("GER40.cash_M1.npz.xz", "929307779e7f7b6906d9ef14b2ce89bf7c94ba9a48de1d414d9f3f0023dc26f6", 0.01),
}
OUT = Path(__file__).parent / "reports"


def load_close(name: str) -> pd.Series:
    """Série des clôtures BID M1 indexée par minute serveur (int64 minutes epoch)."""
    f, sha, pt = INSTR[name]
    path = ROOT.parent / "data_eurusd" / f
    if sha256_file(path) != sha:
        raise ValueError(f"hash inattendu pour {name}")
    z = np.load(io.BytesIO(lzma.decompress(path.read_bytes())), allow_pickle=False)
    tm = np.cumsum(z["t_d"].astype(np.int64))
    c = np.cumsum(z["c_d"].astype(np.int64)) * pt
    return pd.Series(c, index=tm)


def occurrences(e: pd.Series, x: pd.Series, h: int, end_minute: int):
    """Occurrences L1 et L2 pour l'horizon h (minutes). e : EURUSD, x : instrument, mêmes index minute serveur.
    La clôture de la minute t est connue à t + 1 ; on note par commodité P(T) = clôture de la minute T - 1."""
    grid = np.intersect1d(e.index.to_numpy(), x.index.to_numpy())
    grid = grid[grid < end_minute]
    ce = e.reindex(grid).to_numpy(); cx = x.reindex(grid).to_numpy()
    pos = pd.Series(np.arange(len(grid)), index=grid)
    T = grid[(grid % h == 0)] + 0                    # instants de décision (début de la minute T)
    tod = T % 1440
    T = T[(tod >= 60) & (tod < 23 * 60)]              # hors rollover
    # P(T) = clôture de la minute T-1 ; on exige que T-1, T-1-h et T-1+h existent exactement (pas de trou)
    i0 = pos.reindex(T - 1).to_numpy(); im = pos.reindex(T - 1 - h).to_numpy(); ip = pos.reindex(T - 1 + h).to_numpy()
    ok = np.isfinite(i0) & np.isfinite(im) & np.isfinite(ip)
    T, i0, im, ip = T[ok], i0[ok].astype(int), im[ok].astype(int), ip[ok].astype(int)
    rX = np.log(cx[i0] / cx[im])
    rE = (ce[ip] - ce[i0]) / PIP
    # écart-type passé de rX sur 20 jours (en nombre d'occurrences ≈ 20 jours x occurrences/jour), décalé d'une occurrence
    per_day = max(int(22 * 60 / h), 1)
    sd = pd.Series(rX).rolling(20 * per_day, min_periods=5 * per_day).std().shift(1).to_numpy()
    big = np.abs(rX) >= 2 * sd
    dates = pd.to_datetime(T * 60, unit="s").values
    return {"L1": (np.sign(rX) * rE, dates), "L2": (np.where(big, np.sign(rX) * rE, np.nan), dates)}


def main():
    cfg = load_config()
    m1 = load_m1(cfg)
    e = pd.Series(m1["close"].to_numpy(), index=pd.DatetimeIndex(m1["t_server"]).values.astype("datetime64[m]").astype(np.int64))
    end_cal = pd.Timestamp(cfg["splits"]["calibration"][1]) + pd.Timedelta(days=1)
    end_minute = int(end_cal.value // 60_000_000_000) + 2 * 60       # 2026-01-01 00:00 UTC = 02:00 serveur (hiver)
    res, flags = {}, []
    for name in INSTR:
        x = load_close(name)
        for h in (1, 5, 15, 60):
            occ = occurrences(e, x, h, end_minute)
            for fam, (vals, dates) in occ.items():
                key = f"{fam}_{name}_{h}min"
                d = pd.DatetimeIndex(dates)
                rng_t, rng_c = cfg["splits"]["train"], cfg["splits"]["calibration"]
                st = (d >= pd.Timestamp(rng_t[0])) & (d < pd.Timestamp(rng_t[1]) + pd.Timedelta(days=1))
                sc = (d >= pd.Timestamp(rng_c[0])) & (d < pd.Timestamp(rng_c[1]) + pd.Timedelta(days=1))
                r_t = summarize(vals[st], d[st].year.to_numpy())
                r_c = summarize(vals[sc], d[sc].year.to_numpy())
                ok = ("t" in r_t and abs(r_t["mean_pips"]) >= THRESH["min_pips"] and abs(r_t["t"]) >= THRESH["min_t"]
                      and r_t["year_share_same_sign"] >= THRESH["min_year_share"])
                res[key] = {"train": r_t, "calibration": {k: v for k, v in r_c.items() if k != "by_year"}, "exploitable": ok}
                if ok:
                    flags.append(key)
                print(f"{key:28s} n={r_t.get('n'):8d} moy={r_t.get('mean_pips', float('nan')):+.3f} p t={r_t.get('t', float('nan')):+6.2f} "
                      f"années={r_t.get('year_share_same_sign', float('nan')):.0%} | calib moy={r_c.get('mean_pips', float('nan')):+.3f} "
                      f"t={r_c.get('t', float('nan')):+.2f}" + ("  <== EXPLOITABLE" if ok else ""), flush=True)
    out = {"thresholds": THRESH, "n_tests": len(res), "results": res, "exploitable": flags}
    (OUT / "leadlag_eurusd.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print("Exploitables :", flags)


if __name__ == "__main__":
    main()
