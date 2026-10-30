"""M1, étape 2 : évaluation des trades retenus par v2.meta (processus séparé, limite mémoire).

Usage : python -m v2.meta_eval base | stress | final"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .catalogue_run import evaluate, holm
from .jalon_a import tstat
from .meta import pool

OUT = Path("reports_v2")
KEY = ["sid", "symbol", "entry_t"]


def main():
    step = sys.argv[1]
    S = pd.read_csv(OUT / "meta_selected_trades.csv")
    if step in ("base", "stress"):
        Ps, runs, T, _ = pool(stress=(step == "stress"))
        T = T.merge(S[KEY], on=KEY, how="inner")
        keep = set(T["pkey"])
        runs = {k: v for k, v in runs.items() if k in keep}
        gc.collect()
        x = T["r_net"].to_numpy()
        if step == "stress":
            out = dict(n=int(len(x)), expectancy_r=round(float(x.mean()), 4), t_stat=round(tstat(x), 2))
        else:
            _, days, M, mc = evaluate(T, runs, Ps, "2018-01-01")
            out = dict(metrics=M, monte_carlo=mc)
        (OUT / f"meta_eval_{step}.json").write_text(json.dumps(out, indent=2, default=str))
        print(step, json.dumps(out, default=str)[:1500])
    else:
        base = json.loads((OUT / "meta_eval_base.json").read_text())
        st = json.loads((OUT / "meta_eval_stress.json").read_text())
        log = json.loads((OUT / "meta_semesters.json").read_text())
        reg = json.loads((OUT / "famille.json").read_text())
        M, mc = base["metrics"], base["monte_carlo"]
        reg["M1"] = dict(p=M.get("p", 1.0), lot="meta")
        (OUT / "famille.json").write_text(json.dumps(reg, indent=1))
        H = holm({k: v["p"] for k, v in reg.items()}, m=len(reg))
        cr = dict(holm=H["M1"]["significant"], stress=st["expectancy_r"] > 0, n=M.get("n", 0) >= 300,
                  pf=(M.get("pf") or 0) >= 1.3, dd=-M.get("max_dd_pct", -99) <= 6, mc=mc.get("p_pass", 0) >= 0.5)
        sel_by = S.groupby("sid")["r_net"].agg(["size", "mean"]).round(3).reset_index().to_dict("records")
        res = dict(semesters=log, selected=M, monte_carlo=mc, stress=st, holm=H["M1"], family=len(reg), criteria=cr,
                   candidate=all(cr.values()), selected_by_strategy=sel_by)
        (OUT / "meta_dev.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: res[k] for k in ("selected", "monte_carlo", "stress", "holm", "criteria", "candidate")}, indent=1, default=str))


if __name__ == "__main__":
    main()
