"""Étage 1 bis — Prédicteur deep learning sur séquences longues de bougies (« prévoir la suite du graphique »).

Pré-enregistré le 2026-10-29 à 22:30 (Paris), avant toute mesure.
Question : un réseau qui lit directement de longues séquences multi-unités prédit-il mieux que LightGBM si un achat
(ou une vente) avec TP 2:1 touchera sa cible avant son stop ?
Entrées : 6 features par bougie close (rendement, range, corps, mèches en ATR, volume relatif) sur
  M5 x 96 (8 h), M15 x 96 (24 h), H1 x 120 (5 j), H4 x 60 (10 j), D1 x 60 (3 mois) + contexte (11).
Architecture : un encodeur GRU (32) + attention temporelle par unité, fusion MLP (64), 4 sorties :
  P(achat TP2 gagnant), P(vente TP2 gagnant) [BCE], rendements futurs 60 et 240 min en ATR H1 [Huber, poids 0,5].
Lignes et étiquettes : identiques à l'étage 1 (predictor.dataset) : mêmes trades, mêmes coûts.
Walk-forward purgé (3 plis) : entraînement < 2021 -> test 2021-2022 ; < 2023 -> test 2023-2024 ;
  < 2025 -> test 2025-01 -> 2025-06. Arrêt précoce sur les 10 % les plus récents de la fenêtre (patience 2, max 8
  époques), Adam lr 1e-3, lot 512, seed 1. Hyperparamètres fixes, aucune recherche.
Comparaison loyale : LightGBM de l'étage 1 réentraîné sur les MÊMES plis.
Porte (par sens, hors échantillon 2021 -> 2025-06) : identique à l'étage 1 (Spearman des déciles >= 0,8 ; meilleur
  dixième R net >= +0,05, t journalier >= 3, >= 300 trades, >= 65 % d'années positives).
Données tronquées au 2025-12-31 ; validation 2026 et test protégé non lus.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import load_config, load_m1
from .features_pred import price_at
from .market import CTX_FEATS, atr, bar_features, build_market
from .model import TFEncoder
from .predictor import dataset, evaluate_oos, fit as fit_lgbm
from .timeframes import aggregate, last_closed

SEQ = {"M5": 96, "M15": 96, "H1": 120, "H4": 60, "D1": 60}
RULE = {"M5": "5min", "M15": "15min", "H1": "1h", "H4": "4h", "D1": "1D"}
FOLDS = [("2021-01-01", "2023-01-01"), ("2023-01-01", "2025-01-01"), ("2025-01-01", "2025-07-01")]
OUT = Path(__file__).parent / "reports"


class Net(nn.Module):
    def __init__(self, h=32, fuse=64):
        super().__init__()
        self.enc = nn.ModuleDict({tf: TFEncoder(6, h) for tf in SEQ})
        self.ctx = nn.Sequential(nn.Linear(len(CTX_FEATS), h), nn.Tanh())
        self.body = nn.Sequential(nn.Linear(2 * h * len(SEQ) + h, fuse), nn.ReLU(), nn.Dropout(0.1), nn.Linear(fuse, fuse), nn.ReLU())
        self.cls = nn.Linear(fuse, 2)
        self.reg = nn.Linear(fuse, 2)

    def forward(self, xs, ctx):
        z = [self.enc[tf](xs[tf]) for tf in SEQ] + [self.ctx(ctx)]
        b = self.body(torch.cat(z, -1))
        return self.cls(b), self.reg(b)


def prepare(cfg):
    D = dataset(cfg)
    m1 = load_m1(cfg)
    m1 = m1[pd.DatetimeIndex(m1["t_utc"]) < pd.Timestamp("2026-01-01")].reset_index(drop=True)
    mk = build_market(cfg, m1)
    idx = D["dec"].to_numpy()
    T = mk.dec_t_server[idx]
    feats, last = {}, {}
    for tf, rule in RULE.items():
        b = aggregate(m1, rule)
        f, _ = bar_features(b)
        feats[tf] = torch.from_numpy(f)
        last[tf] = last_closed(b["end"].to_numpy(), T)
    ctx = torch.from_numpy(mk.ctx[idx].copy())
    # cibles auxiliaires : rendements futurs 60 et 240 min en ATR H1 (futur = cible, jamais une entrée)
    t_min = pd.DatetimeIndex(m1["t_server"]).values.astype("datetime64[m]").astype(np.int64)
    c = m1["close"].to_numpy()
    Tm = T.astype("datetime64[m]").astype(np.int64)
    p0 = price_at(t_min, c, Tm)
    h1 = aggregate(m1, "1h")
    a_h1 = atr(h1, 14)[np.maximum(last_closed(h1["end"].to_numpy(), T), 0)]
    aux = np.stack([np.clip((price_at(t_min, c, Tm + m) - p0) / a_h1, -10, 10) for m in (60, 240)], 1)
    aux = np.nan_to_num(aux, nan=0.0).astype(np.float32)
    return D, feats, last, ctx, torch.from_numpy(aux)


def batch(rows, feats, last, ctx):
    xs = {}
    for tf, L in SEQ.items():
        li = torch.from_numpy(last[tf][rows])
        ii = (li[:, None] - torch.arange(L - 1, -1, -1)[None, :]).clamp(min=0)
        x = feats[tf][ii]
        x[li < L - 1] = 0.0                     # historique insuffisant : séquence neutre
        xs[tf] = x
    return xs, ctx[rows]


def train_fold(tr_rows, te_rows, Y, aux, feats, last, ctx, seed=1, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    cut = int(len(tr_rows) * 0.9)
    fit_rows, val_rows = tr_rows[:cut], tr_rows[cut:]
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce, hub = nn.BCEWithLogitsLoss(), nn.HuberLoss()
    Yt, At = torch.from_numpy(Y.astype(np.float32)), aux

    def loss_on(rows):
        xs, cx = batch(rows, feats, last, ctx)
        lo, rg = net(xs, cx)
        return bce(lo, Yt[rows]) + 0.5 * hub(rg, At[rows])

    best, best_state, bad = np.inf, None, 0
    g = np.random.default_rng(seed)
    for ep in range(8):
        net.train()
        perm = g.permutation(fit_rows)
        t0 = time.time()
        for i in range(0, len(perm), 512):
            loss = loss_on(perm[i:i + 512])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([loss_on(val_rows[i:i + 2048]).item() for i in range(0, len(val_rows), 2048)]))
        log(f"   époque {ep + 1} val={vl:.5f} ({time.time() - t0:.0f} s)")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    net.load_state_dict(best_state)
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(te_rows), 2048):
            xs, cx = batch(te_rows[i:i + 2048], feats, last, ctx)
            out.append(torch.sigmoid(net(xs, cx)[0]).numpy())
    return np.concatenate(out)


def main():
    cfg = load_config()
    D, feats, last, ctx, aux = prepare(cfg)
    dates = pd.DatetimeIndex(D["date"])
    R = {s: D[f"y_R_{s}_tp2"].to_numpy() for s in ("long", "short")}
    Y = np.stack([(R["long"] > 0), (R["short"] > 0)], 1).astype(np.float32)
    lgb_feats = [c for c in D.columns if not c.startswith("y_") and c not in ("dec", "date")]
    oos_dl = np.full((len(D), 2), np.nan)
    oos_gb = np.full((len(D), 2), np.nan)
    t0 = time.time()
    for a, b in FOLDS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        tr = np.flatnonzero(dates < a - pd.Timedelta(days=2))
        te = np.flatnonzero((dates >= a) & (dates < b))
        print(f"pli test {a.date()} -> {b.date()} : entraînement {len(tr)} lignes, test {len(te)}", flush=True)
        oos_dl[te] = train_fold(tr, te, Y, aux, feats, last, ctx, log=lambda s: print(s, flush=True))
        for k, s in enumerate(("long", "short")):
            m = fit_lgbm(D.iloc[tr][lgb_feats], Y[tr, k].astype(int))
            oos_gb[te, k] = m.predict_proba(D.iloc[te][lgb_feats])[:, 1]
    sel = np.isfinite(oos_dl[:, 0])
    out = {"seq": SEQ, "folds": FOLDS, "results": {}}
    for k, s in enumerate(("long", "short")):
        for name, oos in (("deep", oos_dl), ("lightgbm", oos_gb)):
            r = evaluate_oos(oos[sel, k], R[s][sel], dates[sel].values, dates[sel].year.to_numpy())
            out["results"][f"{name}_{s}_tp2"] = r
            t = r["top"]
            print(f"{name:9s} {s:5s} AUC={r['auc']} Spearman={r['spearman_decile']} base={r['base_mean_R']} | top10% n={t['n']} "
                  f"R={t['mean_R']} t_jour={t['t_daily']} années+={t['year_share_pos']:.0%} gagnants={t['win_rate']} | "
                  f"porte={'OUI' if r['pass'] else 'non'}", flush=True)
            print("   déciles R:", [d["mean"] for d in r["deciles"]], flush=True)
    out["elapsed_s"] = round(time.time() - t0)
    (OUT / "dl_predictor.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
