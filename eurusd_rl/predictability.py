"""Étape 1 — Carte de prévisibilité EURUSD (études d'événements, AUCUNE stratégie).

Pré-enregistré le 2026-10-29 à 21:10 (Paris), avant toute mesure. Données : prix BID M1 (le spread s'annule dans un
rendement mesuré sur la même série). Période de mesure : ENTRAÎNEMENT 2015-01-05 -> 2025-06-30 ; la calibration
(2025-07 -> 2025-12) ne sert qu'à vérifier le SIGNE des effets retenus. Validation 2026 et test protégé : non utilisés.

Familles de tests (44 au total) :
  E1  Rendement moyen de chaque heure (heure de New York, 24 tests) et des 4 sessions (Asie 17-03 NY, Europe 03-08,
      chevauchement 08-12, après-midi US 12-17 NY).
  E2  Fixing de Londres 16:00 (heure de Londres, DST géré) : 60 min avant et 60 min après, tous les jours (2 tests) ;
      dernier jour ouvré du mois, rendement signé par le SENS INVERSE du mouvement du mois en cours (2 tests).
  E3  Rendement du jour serveur (17:00 NY -> 17:00 NY) par jour de la semaine (5 tests).
  E4  Après un grand mouvement H1 (|r| >= 2,5 x ATR H1 des 24 barres précédentes) : rendement signé dans le sens du
      mouvement sur 1 h, 4 h et 24 h (3 tests, événements sans chevauchement) ; après un grand jour (|r| >= 2 x ATR
      D1 20 jours) : jour suivant signé (1 test).
  E5  Gap du lundi (|gap| >= 5 pips) : rendement de l'ouverture à 03:00 NY, signé dans le sens du comblement (1 test).
  E6  Momentum temporel : rendement des 5 jours suivants signé par les 20 jours passés (échantillon hebdomadaire),
      rendement du jour suivant signé par les 5 jours passés (2 tests).

Critère « effet exploitable » (les trois à la fois) :
  |moyenne| >= 2,0 pips par occurrence (coût aller-retour ~1,3 pip + marge) ;
  |t| >= 3,25 (Bonferroni à 5 % sur 44 tests) ;
  même signe que la moyenne dans >= 70 % des années civiles.
Puis signe confirmé sur la calibration (puissance faible : indicatif).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_config, load_m1
from .market import atr
from .timeframes import aggregate

PIP = 1e-4
THRESH = {"min_pips": 2.0, "min_t": 3.25, "min_year_share": 0.70}
OUT = Path(__file__).parent / "reports"


class Px:
    """Prix BID connu à l'instant T (heure serveur) : clôture de la dernière M1 terminée à T, au plus 15 min avant."""

    def __init__(self, m1: pd.DataFrame):
        self.t = pd.DatetimeIndex(m1["t_server"]).values.astype("datetime64[m]").astype(np.int64)
        self.c = m1["close"].to_numpy()
        self.o = m1["open"].to_numpy()

    def at(self, T) -> np.ndarray:
        Tm = np.asarray(pd.DatetimeIndex(T).values.astype("datetime64[m]").astype(np.int64))
        i = np.searchsorted(self.t, Tm, side="left") - 1
        ok = (i >= 0) & (Tm - self.t[np.maximum(i, 0)] <= 15)
        return np.where(ok, self.c[np.maximum(i, 0)], np.nan)


def server_from_local(dates, hh, mm, tz):
    """Heure locale (tz) d'une date -> heure serveur (NY + 7 h)."""
    loc = pd.DatetimeIndex(pd.to_datetime(dates) + pd.Timedelta(hours=hh, minutes=mm)).tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    ny = loc.tz_convert("America/New_York").tz_localize(None)
    return ny + pd.Timedelta(hours=7)


def stat(x: np.ndarray, years: np.ndarray) -> dict:
    """Ici `years` est en fait le vecteur des DATES des occurrences (datetime64) ; voir occ()."""
    return {"x": np.asarray(x, float), "date": np.asarray(years)}


def summarize(x: np.ndarray, years: np.ndarray) -> dict:
    ok = np.isfinite(x)
    x, years = x[ok], years[ok]
    n = len(x)
    if n < 20:
        return {"n": int(n)}
    m = float(x.mean())
    t = m / (x.std(ddof=1) / np.sqrt(n))
    yy = pd.Series(x).groupby(years).mean()
    share = float((np.sign(yy) == np.sign(m)).mean())
    return {"n": int(n), "mean_pips": round(m, 3), "t": round(float(t), 2), "year_share_same_sign": round(share, 3),
            "by_year": {int(k): round(float(v), 2) for k, v in yy.items()}}


def tests(m1: pd.DataFrame) -> dict:
    px = Px(m1)
    ts = pd.DatetimeIndex(m1["t_server"])
    days = pd.DatetimeIndex(np.unique(ts.normalize()))
    days = days[days.dayofweek < 5]
    yrs = days.values
    R = {}

    # E1 heures et sessions (heure NY ; serveur = NY + 7 h)
    for H in range(24):
        a = px.at(days + pd.Timedelta(hours=H)); b = px.at(days + pd.Timedelta(hours=H + 1))
        R[f"E1_heure_NY_{(H - 7) % 24:02d}h"] = stat((b - a) / PIP, yrs)
    for name, (s, e) in {"asie_17-03NY": (0, 10), "europe_03-08NY": (10, 15), "chevauchement_08-12NY": (15, 19),
                         "apres-midi_US_12-17NY": (19, 24)}.items():
        a = px.at(days + pd.Timedelta(hours=s)); b = px.at(days + pd.Timedelta(hours=e) - pd.Timedelta(minutes=5))
        R[f"E1_session_{name}"] = stat((b - a) / PIP, yrs)

    # E2 fixing de Londres 16:00
    # le jour serveur D (00:00 serveur = 17:00 NY la veille) contient le fixing de la date calendaire D (≈ 18:00 serveur)
    fix = server_from_local(days, 16, 0, "Europe/London")
    p0, p1, p2 = px.at(fix - pd.Timedelta(minutes=60)), px.at(fix), px.at(fix + pd.Timedelta(minutes=60))
    R["E2_fixing_avant_60min"] = stat((p1 - p0) / PIP, yrs)
    R["E2_fixing_apres_60min"] = stat((p2 - p1) / PIP, yrs)
    per = days.to_period("M")
    last_bd = pd.Series(days, index=days).groupby(per).transform("max").to_numpy() == days.values
    first_day = pd.Series(days, index=days).groupby(per).transform("min").values
    mtd = (p0 - px.at(pd.DatetimeIndex(first_day) + pd.Timedelta(minutes=10))) / PIP
    sgn = -np.sign(mtd)
    R["E2_fin_de_mois_avant_fixing_signe_inverse_mois"] = stat(np.where(last_bd, sgn * (p1 - p0) / PIP, np.nan), yrs)
    R["E2_fin_de_mois_apres_fixing_signe_inverse_mois"] = stat(np.where(last_bd, sgn * (p2 - p1) / PIP, np.nan), yrs)

    # E3 jour de la semaine (jour serveur complet)
    d1 = aggregate(m1, "1D")
    d1 = d1[pd.DatetimeIndex(d1["start"]).dayofweek < 5].reset_index(drop=True)
    r_d = (d1["close"] - d1["open"]).to_numpy() / PIP
    dow = pd.DatetimeIndex(d1["start"]).dayofweek.to_numpy()
    y_d = pd.DatetimeIndex(d1["start"]).values
    for k, nm in enumerate(["lundi", "mardi", "mercredi", "jeudi", "vendredi"]):
        R[f"E3_{nm}"] = stat(np.where(dow == k, r_d, np.nan), y_d)

    # E4 grands mouvements H1 et D1
    h1 = aggregate(m1, "1h")
    c = h1["close"].to_numpy(); o = h1["open"].to_numpy()
    r1 = c - o
    a24 = pd.Series(atr(h1, 24)).shift(1).to_numpy()             # ATR des 24 barres précédentes
    big = np.abs(r1) >= 2.5 * a24
    ends = pd.DatetimeIndex(h1["end"])
    y_h = ends.values
    for hz in (1, 4, 24):
        idx = np.flatnonzero(big)
        keep, last = [], -10 ** 9
        for i in idx:                                             # événements sans chevauchement d'horizon
            if (ends[i] - ends[last]).total_seconds() / 3600 >= hz if last >= 0 else True:
                keep.append(i); last = i
        keep = np.array(keep, int)
        fut = px.at(ends[keep] + pd.Timedelta(hours=hz)) - px.at(ends[keep])
        x = np.full(len(h1), np.nan)
        x[keep] = np.sign(r1[keep]) * fut / PIP
        R[f"E4_grand_mvt_H1_continuation_{hz}h"] = stat(x, y_h)
    aD = pd.Series(atr(d1, 20)).shift(1).to_numpy()
    bigD = np.abs(d1["close"] - d1["open"]).to_numpy() >= 2 * aD
    nxt = np.r_[r_d[1:], np.nan]
    R["E4_grand_jour_continuation_jour_suivant"] = stat(np.where(bigD, np.sign(r_d) * nxt, np.nan), y_d)

    # E5 gap du lundi : du vendredi clôture à l'ouverture du lundi, puis ouverture -> 03:00 NY (10:00 serveur)
    mon = d1[pd.DatetimeIndex(d1["start"]).dayofweek == 0].index.to_numpy()
    mon = mon[mon > 0]
    gap = (d1["open"].to_numpy()[mon] - d1["close"].to_numpy()[mon - 1]) / PIP
    later = px.at(pd.DatetimeIndex(d1["start"].to_numpy()[mon]) + pd.Timedelta(hours=10))
    fill = -np.sign(gap) * (later - d1["open"].to_numpy()[mon]) / PIP
    R["E5_gap_lundi_comblement"] = stat(np.where(np.abs(gap) >= 5, fill, np.nan), y_d[mon])

    # E6 momentum temporel
    cl = d1["close"].to_numpy()
    past20 = cl - np.r_[np.full(20, np.nan), cl[:-20]]
    fut5 = np.r_[cl[5:], np.full(5, np.nan)] - cl
    weekly = dow == 4                                             # échantillon du vendredi : pas de chevauchement
    R["E6_momentum_20j_vers_5j"] = stat(np.where(weekly, np.sign(past20) * fut5 / PIP, np.nan), y_d)
    past5 = cl - np.r_[np.full(5, np.nan), cl[:-5]]
    fut1 = np.r_[cl[1:], np.nan] - cl
    R["E6_momentum_5j_vers_1j"] = stat(np.sign(past5) * fut1 / PIP, y_d)
    return R


def verdict(r: dict) -> bool:
    return ("t" in r and abs(r["mean_pips"]) >= THRESH["min_pips"] and abs(r["t"]) >= THRESH["min_t"]
            and r["year_share_same_sign"] >= THRESH["min_year_share"])


def _period(occ: dict, rng: list[str]) -> dict:
    d = pd.DatetimeIndex(occ["date"])
    sel = (d >= pd.Timestamp(rng[0])) & (d < pd.Timestamp(rng[1]) + pd.Timedelta(days=1))
    return summarize(occ["x"][sel], d[sel].year.to_numpy())


def main():
    cfg = load_config()
    m1 = load_m1(cfg)
    # données chargées jusqu'à la fin de la calibration seulement : 2026 (validation) n'est jamais lue
    end_cal = pd.Timestamp(cfg["splits"]["calibration"][1]) + pd.Timedelta(days=1)
    m1 = m1[pd.DatetimeIndex(m1["t_utc"]) < end_cal].reset_index(drop=True)
    OCC = tests(m1)
    assert len(OCC) == 44, len(OCC)
    RT = {k: _period(v, cfg["splits"]["train"]) for k, v in OCC.items()}
    out = {"thresholds": THRESH, "n_tests": len(RT), "train": RT, "exploitable": [k for k, v in RT.items() if verdict(v)]}
    out["calibration_sign_check"] = {k: _period(OCC[k], cfg["splits"]["calibration"]) for k in out["exploitable"]}
    (OUT / "predictability_eurusd.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    for k, v in RT.items():
        flag = "  <== EXPLOITABLE" if verdict(v) else ""
        print(f"{k:52s} n={v.get('n'):5d} moy={v.get('mean_pips', float('nan')):+7.2f} p  t={v.get('t', float('nan')):+6.2f}  "
              f"années même signe={v.get('year_share_same_sign', float('nan')):.0%}{flag}")
    print("Exploitables :", out["exploitable"])


if __name__ == "__main__":
    main()
