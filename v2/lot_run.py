"""Évaluation d'un lot de stratégies sans paramétrage + correction de Holm sur la famille cumulée.

La famille cumulée est tenue dans reports_v2/famille.json (une entrée par stratégie testée, avec sa p-valeur
de développement). Usage : python -m v2.lot_run lot2"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogue as K
from .catalogue_run import evaluate, holm, prepared
from .jalon_a import ts

OUT = Path("reports_v2")
E = ts("2024-06-30") + 86400
REG = OUT / "famille.json"


def registry() -> dict:
    if REG.exists():
        return json.loads(REG.read_text())
    d = json.loads((OUT / "catalogue_dev.json").read_text())
    reg = {k: dict(p=v["p"], lot="catalogue") for k, v in d["holm"].items()}
    REG.write_text(json.dumps(reg, indent=1))
    return reg


def lot_modules(name):
    if name == "lot2":
        from . import lot2
        return lot2.LOT2, lot2.run
    if name == "lot3":
        from . import lot3
        return lot3.LOT, lot3.run
    if name == "lot4":
        from . import lot4
        return lot4.LOT, lot4.run
    if name == "lot5":
        from . import lot5
        return lot5.LOT, lot5.run
    raise SystemExit(f"lot inconnu : {name}")


def collect(Ps, strat_syms, runner, stress=False):
    runs, cands = {}, {}
    for sid, syms in strat_syms.items():
        oo = []
        for s in syms:
            P = K.stressed(Ps[s]) if stress else Ps[s]
            r = runner(P, sid)
            df = r["trades"]
            df = df[df["entry_t"] < E].copy()
            df["sid"], df["ci"] = sid, 0
            df["pkey"] = [(sid, s, 0)] * len(df)
            runs[(sid, s, 0)] = dict(trades=df, path_t=r["path_t"], path_u=r["path_u"])
            oo.append(df)
        cands[sid] = pd.concat(oo, ignore_index=True)
    return runs, cands


def main():
    name = sys.argv[1]
    strat_syms, runner = lot_modules(name)
    reg = registry()
    for sid in strat_syms:
        if sid in reg and reg[sid]["lot"] != name:
            raise SystemExit(f"{sid} déjà dans la famille")
    syms = sorted({s for v in strat_syms.values() for s in v})
    Ps = {s: prepared(s) for s in syms}
    runs, cands = collect(Ps, strat_syms, runner)
    table = {}
    for sid, cd in cands.items():
        start = str(min(pd.Timestamp(Ps[s]["days"]["date"].iloc[0]) for s in strat_syms[sid]).date())
        acc, days, M, mc = evaluate(cd, runs, Ps, start)
        per = acc.groupby("symbol").agg(n=("r_eff", "size"), exp_r=("r_eff", "mean")).round(4).reset_index().to_dict("records") \
            if acc is not None else []
        table[sid] = dict(metrics=M, monte_carlo=mc, per_symbol=per, start=start)
        reg[sid] = dict(p=M["p"], lot=name)
        print(sid, {k: M.get(k) for k in ("n", "expectancy_r", "pf", "t_stat", "p", "max_dd_pct")}, "MC", mc.get("p_pass"), per, flush=True)
    H = holm({k: v["p"] for k, v in reg.items()}, m=len(reg))
    runs_st, cands_st = collect(Ps, strat_syms, runner, stress=True)
    cands_out, pistes = [], []
    for sid in strat_syms:
        _, _, Mst, _ = evaluate(cands_st[sid], runs_st, Ps, table[sid]["start"], n_paths=500)
        table[sid]["stress"] = {k: Mst.get(k) for k in ("n", "expectancy_r", "pf", "t_stat")}
        table[sid]["holm"] = H[sid]
        pos_stress = (Mst.get("expectancy_r") or -1) > 0
        if H[sid]["significant"] and pos_stress:
            cands_out.append(sid)
        if table[sid]["metrics"]["p"] < 0.05 and pos_stress:
            pistes.append(sid)
    REG.write_text(json.dumps(reg, indent=1))
    res = dict(lot=name, family_size=len(reg), strategies=table, candidates=cands_out, pistes=pistes)
    (OUT / f"{name}_dev.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(dict(family=len(reg), candidates=cands_out, pistes=pistes,
                          holm={k: H[k] for k in strat_syms}), indent=1, default=str))


if __name__ == "__main__":
    main()
