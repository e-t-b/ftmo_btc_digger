"""Diagnostic des trades S26 (journée de tendance) : d'où viennent les pertes ?

Sources : Dukascopy 2013-2021-08 et FTMO développement (2021-09 → 2024-06), indices uniquement.
La période finale FTMO (2024-07 →) n'est PAS utilisée ici."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from . import dukascopy as DK
from . import lot4
from .catalogue_run import prepared

OUT = Path("reports_v2")
DEV_END = int(pd.Timestamp("2024-07-01").value // 10**9)


def trades_with_context(P, end):
    S = lot4.setups(P, "S26")
    r = K.run(P, "S26", {}, setups=S)
    tr = r["trades"]
    tr = tr[tr["entry_t"] < end].copy()
    t, o, h, l, c = P["t"], P["o"], P["h"], P["l"], P["c"]
    D = P["days"]
    rows = []
    for i, x in tr.iterrows():
        day = int(x["day"])
        dr = D.iloc[day]
        ws, fc, atr = int(dr["ws"]), int(dr["fc"]), dr["atr"]
        j0 = K._first_at(t, ws, 120)
        je = np.searchsorted(t, x["entry_t"])
        jx = np.searchsorted(t, x["exit_t"])
        d = int(x["dir"])
        O = o[j0]
        e = x["entry"]
        R = x["R"]
        seg_h, seg_l = h[je:jx + 1], l[je:jx + 1]
        mfe = (seg_h.max() - e) / R if d == 1 else (e - seg_l.min()) / R
        mae = (e - seg_l.min()) / R if d == 1 else (seg_h.max() - e) / R
        jc = np.searchsorted(t, fc) - 1
        # trajectoire : R latent toutes les 30 min après l'entrée (clôture), si toujours en position
        traj = {}
        for m in (30, 60, 120, 180, 240, 300):
            k = np.searchsorted(t, x["entry_t"] + m * 60) - 1
            traj[f"r_{m}m"] = d * (c[k] - e) / R if k < jc else np.nan
        pc = dr["pcs"]
        rows.append(dict(
            idx=i, symbol=P["symbol"], year=pd.Timestamp(int(x["entry_t"]), unit="s").year,
            dow=pd.Timestamp(int(x["entry_t"]), unit="s").dayofweek, dir=d, r_net=x["r_net"], r_gross=x["r_gross"],
            why=K.np.array(["stop", "gap", "trail", "tp", "time", "eod"])[int(x["why"])],
            move_atr=d * (e - O) / atr, R_atr=R / atr, mfe=mfe, mae=mae,
            gap_atr=d * (O - pc) / atr if pc > 0 else np.nan,
            trend=d * dr["trend"], atr_rel=atr / O, prev_range_atr=dr["pR"] / atr,
            hold_min=(x["exit_t"] - x["entry_t"]) / 60, **traj))
    return pd.DataFrame(rows)


def bucket(df, col, bins, labels=None):
    g = df.groupby(pd.cut(df[col], bins, labels=labels), observed=True)["r_net"]
    return g.agg(n="size", mean="mean").round(3)


def main():
    frames = []
    for dk in DK.MAP:
        frames.append(trades_with_context(K.prepare(dk), DK.END).assign(src="dukascopy"))
    for ft in ("US500.cash", "US100.cash", "GER40.cash"):
        frames.append(trades_with_context(prepared(ft), DEV_END).assign(src="ftmo"))
    T = pd.concat(frames, ignore_index=True)
    T.to_csv(OUT / "diag_s26_trades.csv", index=False)
    pd.set_option("display.width", 200)
    for src, g in [("TOUT", T)] + list(T.groupby("src")):
        print(f"\n######## {src} : n={len(g)} R net={g.r_net.mean():.3f} gagnants={(g.r_net > 0).mean():.2f}")
        print("raisons de sortie :\n", g.groupby("why")["r_net"].agg(n="size", mean="mean", sum="sum").round(3).to_string())
        print("MFE (gain max atteint, R) quantiles :", g.mfe.quantile([.25, .5, .75, .9]).round(2).to_dict())
        print("trades perdants dont MFE >= 1R :", round(((g.r_net < 0) & (g.mfe >= 1)).mean(), 3),
              "| >= 0.5R :", round(((g.r_net < 0) & (g.mfe >= 0.5)).mean(), 3))
        print("trajectoire moyenne (R latent) :", g[[c for c in g if c.startswith("r_")and c.endswith("m")]].mean().round(3).to_dict())
        print("taille du stop (R en ATR D1) :\n", bucket(g, "R_atr", [0, 0.6, 0.8, 1.0, 5]).to_string())
        print("mouvement à l'entrée (ATR) :\n", bucket(g, "move_atr", [0, 0.6, 0.8, 1.0, 5]).to_string())
        print("gap dans le sens du trade (ATR) :\n", bucket(g, "gap_atr", [-5, -0.3, 0, 0.3, 5]).to_string())
        print("tendance D1 alignée :\n", g.groupby("trend")["r_net"].agg(n="size", mean="mean").round(3).to_string())
        print("sens :\n", g.groupby("dir")["r_net"].agg(n="size", mean="mean").round(3).to_string())
        print("volatilité (ATR/prix) :\n", bucket(g, "atr_rel", [0, 0.008, 0.012, 0.018, 1]).to_string())
        print("jour :\n", g.groupby("dow")["r_net"].agg(n="size", mean="mean").round(3).to_string())


if __name__ == "__main__":
    main()
