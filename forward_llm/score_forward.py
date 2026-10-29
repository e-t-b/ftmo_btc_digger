"""Notation des avis quotidiens (protocole : forward_llm/PROTOCOL.md).

Usage :
  python -m forward_llm.score_forward --csv chemin/EURUSD_M1.csv          # export MT5 (export_mt5.py)
  python -m forward_llm.score_forward --research-data                      # tests : fichier de recherche (jusqu'à 2026-10-28)
Écrit forward_llm/results/score.json et score.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eurusd_rl.data import load_config, load_m1, server_to_utc
from eurusd_rl.labels import label_one
from eurusd_rl.market import build_market

HERE = Path(__file__).parent
DELAY_MIN = 5          # entrée au plus tôt 5 minutes après l'avis
CUTOFF_PARIS = (9, 0)  # avis poussé après 09:00 Paris = exclu
TP_R = 2.0


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts = pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s"))
    out = pd.DataFrame({"t_server": ts.values, "open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float),
                        "low": df["low"].to_numpy(float), "close": df["close"].to_numpy(float),
                        "tick_volume": df["tick_volume"].to_numpy(np.int64),
                        "spread_obs_pts": np.rint(df["spread_px"].to_numpy(float) / 1e-5).astype(np.int64)})
    out = out[~out["t_server"].duplicated()].sort_values("t_server").reset_index(drop=True)
    out.insert(1, "t_utc", server_to_utc(pd.DatetimeIndex(out["t_server"])).values)
    pt = 1e-5
    for k, c in (("o_pts", "open"), ("h_pts", "high"), ("l_pts", "low"), ("c_pts", "close")):
        out[k] = np.rint(out[c] / pt).astype(np.int64)
    return out


def utc_to_server(t_utc: pd.Timestamp) -> pd.Timestamp:
    ny = pd.Timestamp(t_utc).tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    return ny + pd.Timedelta(hours=7)


def load_decisions() -> pd.DataFrame:
    rows = []
    for f in sorted((HERE / "decisions").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["file"] = f.name
        rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["decision_time_utc", "direction", "confidence", "file"])
    df = pd.DataFrame(rows)
    df["decision_time_utc"] = pd.to_datetime(df["decision_time_utc"], utc=True).dt.tz_localize(None)
    return df.sort_values("decision_time_utc").reset_index(drop=True)


def score(decisions: pd.DataFrame, m1: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    # profil de spread : fenêtre 2022-2024 si l'export la couvre, sinon la première année disponible de l'export
    t0 = pd.Timestamp(m1["t_utc"].iloc[0])
    if t0 > pd.Timestamp(cfg["execution"]["spread"]["profile_fit"][0]):
        cfg = {**cfg, "execution": {**cfg["execution"], "spread": {**cfg["execution"]["spread"],
               "profile_fit": [str(t0.date()), str((t0 + pd.Timedelta(days=365)).date())]}}}
    mk = build_market(cfg, m1)
    ex, pt = cfg["execution"], cfg["data"]["point"]
    comm_px = 2 * ex["commission_per_lot_side"] / cfg["data"]["contract_size"]
    slips = (ex["entry_slippage_pts"] * pt, ex["exit_slippage_pts"] * pt, ex["stop_slippage_pts"] * pt)
    T = pd.DatetimeIndex(mk.dec_t_server)
    out = []
    for _, r in decisions.iterrows():
        t_dec_utc = pd.Timestamp(r["decision_time_utc"])
        paris = t_dec_utc.tz_localize("UTC").tz_convert("Europe/Paris")
        late = (paris.hour, paris.minute) > CUTOFF_PARIS
        t_srv = utc_to_server(t_dec_utc) + pd.Timedelta(minutes=DELAY_MIN)
        cand = np.flatnonzero((T >= t_srv) & mk.dec_can_enter & (T.normalize() == t_srv.normalize()))
        rec = {"file": r.get("file"), "decision_time_utc": str(t_dec_utc), "direction": r["direction"],
               "confidence": r.get("confidence"), "late": bool(late)}
        if late or len(cand) == 0:
            rec["status"] = "exclu (tardif)" if late else "pas de données"
            out.append(rec)
            continue
        d = cand[0]
        j, jf, sd = int(mk.dec_m1[d]), int(mk.dec_force_m1[d]), float(mk.dec_stop_dist[d])
        rl = label_one(mk.o, mk.h, mk.l, mk.spread, j, jf, 1, sd, TP_R, *slips, comm_px)[0]
        rs = label_one(mk.o, mk.h, mk.l, mk.spread, j, jf, -1, sd, TP_R, *slips, comm_px)[0]
        rec.update(status="noté", entry_time_server=str(T[d]), stop_pips=round(sd * 1e4, 1),
                   R_long=round(rl, 4), R_short=round(rs, 4),
                   R=round(rl if r["direction"] == "long" else rs if r["direction"] == "short" else 0.0, 4))
        out.append(rec)
    return pd.DataFrame(out)


def summarize(sc: pd.DataFrame) -> dict:
    ok = sc[sc["status"] == "noté"] if len(sc) else sc
    dirn = ok[ok["direction"].isin(["long", "short"])] if len(ok) else ok

    def st(x):
        x = np.asarray(x, float)
        if len(x) < 2:
            return {"n": int(len(x)), "mean_R": float(x.mean()) if len(x) else None}
        return {"n": int(len(x)), "mean_R": round(float(x.mean()), 4), "t": round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2),
                "win_rate": round(float((x > 0).mean()), 3)}

    res = {"n_avis": int(len(sc)), "n_notes": int(len(ok)), "n_directionnels": int(len(dirn)),
           "ia": st(dirn["R"]) if len(dirn) else {"n": 0},
           "ref_hasard": st(((dirn["R_long"] + dirn["R_short"]) / 2)) if len(dirn) else {"n": 0},
           "ref_toujours_achat": st(dirn["R_long"]) if len(dirn) else {"n": 0},
           "ref_toujours_vente": st(dirn["R_short"]) if len(dirn) else {"n": 0}}
    if len(dirn):
        hi = dirn[dirn["confidence"].astype(float) >= 60]
        lo = dirn[dirn["confidence"].astype(float) < 60]
        res["confiance_>=60"] = st(hi["R"]); res["confiance_<60"] = st(lo["R"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--research-data", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.research_data:
        m1 = load_m1(cfg)
    else:
        m1 = load_csv(a.csv)
    sc = score(load_decisions(), m1, cfg)
    res = summarize(sc)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "score.json").write_text(json.dumps({"summary": res, "decisions": sc.to_dict(orient="records")},
                                                             indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
