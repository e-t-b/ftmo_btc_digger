"""Rapport qualité de l'historique M1 (à produire avant tout entraînement).

Usage : python -m eurusd_rl.qc  -> eurusd_rl/reports/data_quality.json et .md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_config, load_m1, sha256_file, ROOT


def quality(m1: pd.DataFrame) -> dict:
    ts = pd.DatetimeIndex(m1["t_server"])
    tu = pd.DatetimeIndex(m1["t_utc"])
    o, h, l, c = (m1[k].to_numpy() for k in ("open", "high", "low", "close"))
    dt = np.diff(ts.values.astype("datetime64[s]").astype(np.int64)) / 60.0
    sp = m1["spread_obs_pts"].to_numpy()
    r = {
        "rows": int(len(m1)),
        "first_utc": str(tu[0]), "last_utc": str(tu[-1]),
        "duplicates": int(pd.Index(ts).duplicated().sum()),
        "non_monotonic": int((dt <= 0).sum()),
        "ohlc_invalid": int(((h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)).sum()),
        "price_nonpositive": int((np.c_[o, h, l, c] <= 0).any(1).sum()),
        "weekend_rows_server": int((ts.dayofweek >= 5).sum()),
        "tick_volume_zero": int((m1["tick_volume"].to_numpy() <= 0).sum()),
    }
    wk = (dt > 5) & (dt < 2000)
    g = pd.DataFrame({"from": ts[:-1][wk], "to": ts[1:][wk], "min": dt[wk]})
    hr = g["from"].dt.hour
    r["gaps_gt5min_weekday"] = int(len(g))
    r["gaps_gt5min_outside_rollover"] = int(((hr != 23) & (hr != 0)).sum())
    r["largest_gaps_outside_rollover"] = g[(hr != 23) & (hr != 0)].sort_values("min", ascending=False).head(10).astype(str).values.tolist()
    r["holiday_gaps_gt1day"] = g[g["min"] > 1400].astype(str).values.tolist()
    y = ts.year
    per_year = pd.DataFrame({"y": y, "sp": sp, "zero": sp <= 0, "tv": m1["tick_volume"].to_numpy()}).groupby("y").agg(
        rows=("sp", "size"), spread_median=("sp", "median"), spread_p95=("sp", lambda s: float(s.quantile(.95))),
        spread_zero_share=("zero", "mean"), tick_volume_median=("tv", "median"))
    per_year["days"] = pd.Series(ts.normalize()).groupby(y).nunique().to_numpy()
    r["per_year"] = per_year.round(3).reset_index().to_dict(orient="records")
    q = pd.Series(sp).groupby(ts.hour).quantile(.95)
    r["spread_p95_by_server_hour"] = {int(k): float(v) for k, v in q.items()}
    lr = np.abs(np.diff(np.log(c))) * 1e4
    r["abs_ret_1m_bps"] = {"p50": float(np.median(lr)), "p9999": float(np.quantile(lr, .9999)), "max": float(lr.max())}
    # vérification du fuseau : minute du plus grand mouvement les premiers vendredis du mois (NFP 08:30 NY = 15:30 serveur)
    d = pd.DataFrame({"r": np.r_[0, lr]}, index=ts)
    nfp = d[(ts.dayofweek == 4) & (ts.day <= 7)]
    pk = pd.DatetimeIndex(nfp.groupby(nfp.index.normalize())["r"].idxmax().to_numpy())
    r["nfp_peak_minute_share_1530_server"] = float(((pk.hour == 15) & (pk.minute.isin([30, 31]))).mean())
    return r


def to_markdown(r: dict, sha: str) -> str:
    L = ["# Rapport qualité — EURUSD M1 (FTMO MT5)", "", f"Fichier : SHA-256 `{sha}`", "",
         f"- Lignes : {r['rows']:,} ; UTC {r['first_utc']} → {r['last_utc']}".replace(",", " "),
         f"- Doublons : {r['duplicates']} ; non monotones : {r['non_monotonic']} ; OHLC incohérents : {r['ohlc_invalid']} ; prix ≤ 0 : {r['price_nonpositive']}",
         f"- Lignes le week-end (serveur) : {r['weekend_rows_server']} ; tick volume nul : {r['tick_volume_zero']}",
         f"- Trous > 5 min en semaine : {r['gaps_gt5min_weekday']} dont hors rollover : {r['gaps_gt5min_outside_rollover']}",
         f"- Fuseau : pic NFP à 15:30-15:31 serveur dans {r['nfp_peak_minute_share_1530_server']:.0%} des premiers vendredis (attendu : majorité)",
         f"- |rendement 1 min| : médiane {r['abs_ret_1m_bps']['p50']:.2f} bp, p99,99 {r['abs_ret_1m_bps']['p9999']:.1f} bp, max {r['abs_ret_1m_bps']['max']:.1f} bp",
         "", "## Par année", "", "| Année | Lignes | Jours | Spread médian | Spread p95 | Part spread 0 | Tick vol. médian |", "|---|---|---|---|---|---|---|"]
    for p in r["per_year"]:
        L.append(f"| {p['y']} | {p['rows']} | {p['days']} | {p['spread_median']} | {p['spread_p95']} | {p['spread_zero_share']:.1%} | {p['tick_volume_median']} |")
    L += ["", "## Trous hors rollover (10 plus longs)", ""]
    L += [f"- {a} → {b} ({float(m):.0f} min)" for a, b, m in r["largest_gaps_outside_rollover"]]
    L += ["", "## Conséquences retenues", "",
          "- Prix BID seuls : l'ask est reconstruit avec le modèle de spread conservateur (`costs.py`).",
          "- Spread historique non homogène (2020-2021 à 50 pts constants, zéros fréquents en 2025-2026) : non utilisé comme feature ; coût = max(profil p75 2022-2024, observé depuis 2022, 1 pt).",
          "- Tick volume non homogène entre années : uniquement en ratio à sa propre moyenne glissante passée.",
          "- Aucune entrée pendant l'heure du rollover ; clôture forcée à 16:45 NY.",
          "- Jours partiels exclus : 2015-01-02 et 2026-10-29 (jour d'export)."]
    return "\n".join(L) + "\n"


def main():
    cfg = load_config()
    m1 = load_m1(cfg)
    r = quality(m1)
    out = Path(__file__).parent / "reports"
    out.mkdir(exist_ok=True)
    sha = sha256_file((ROOT / cfg["data"]["path"]).resolve())
    (out / "data_quality.json").write_text(json.dumps(r, indent=1, default=str), encoding="utf-8")
    (out / "data_quality.md").write_text(to_markdown(r, sha), encoding="utf-8")
    print(to_markdown(r, sha))


if __name__ == "__main__":
    main()
