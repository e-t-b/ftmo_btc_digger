"""Portefeuille multi-instruments avec le module de risque (protocole §6).

Les métriques pluriannuelles sont mesurées à risque constant (0,5 % du solde initial par trade) : la réduction
à 0,25 % et l'arrêt sous 93 % n'ont de sens que pour une tentative de challenge et sont appliqués dans le
Monte Carlo (ftmo.py). Tout le reste (stop journalier interne sur equity flottante, risque ouvert, côté dollar,
marge, arrondi des lots) est appliqué ici."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

PRAGUE = "Europe/Prague"


def prague_day(ts: np.ndarray) -> np.ndarray:
    d = pd.to_datetime(ts, unit="s").tz_localize("UTC").tz_convert(PRAGUE)
    return d.normalize().tz_localize(None).values.astype("datetime64[D]")


def _units(sym: str, spec: dict, R_px: float, entry: float, risk_usd: float) -> tuple[float, float, float]:
    """Unités (arrondies au pas), risque réel en $, notionnel en $."""
    ic = C.INSTRUMENTS[sym]
    usd_per_px = 1.0 / entry if ic["quote"] == "JPY" else 1.0
    step = spec["volume_step"] * spec["contract_size"]
    vmin = spec["volume_min"] * spec["contract_size"]
    u = np.floor(risk_usd / (R_px * usd_per_px) / step) * step
    if u < vmin - 1e-12:
        return 0.0, 0.0, 0.0
    notional = u if sym.startswith("USD") else u * entry
    return u, u * R_px * usd_per_px, notional


class _Open:
    __slots__ = ("i", "sym", "d", "risk", "notional", "exit_t", "pt", "pu", "rnet")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def value(self, tau: int) -> float:
        """R latent (pire minute) à l'instant tau (trade ouvert) ; R final après la sortie."""
        if tau >= self.exit_t:
            return self.rnet
        j = np.searchsorted(self.pt, tau, side="right") - 1
        return float(self.pu[j]) if j >= 0 else 0.0

    def values(self, taus: np.ndarray) -> np.ndarray:
        j = np.searchsorted(self.pt, taus, side="right") - 1
        v = np.where(j >= 0, self.pu[np.maximum(j, 0)], 0.0)
        return np.where(taus >= self.exit_t, self.rnet, v)


def run(cands: pd.DataFrame, paths: dict, specs: dict, risk_frac: float = C.RISK["base"],
        account: float = C.ACCOUNT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cands : trades candidats (colonnes symbol, dir, entry_t, exit_t, entry, R, r_net, p0, pn), triés par entrée.
    paths[symbol] = (path_t, path_u) du run d'où viennent les trades.
    Renvoie (trades acceptés avec pnl $ et R effectif, journal quotidien FTMO)."""
    cands = cands.sort_values("entry_t", kind="stable").reset_index(drop=True)
    cands["fday"] = prague_day(cands["entry_t"].to_numpy())
    base_usd = C.RISK["base"] * account
    risk_usd = risk_frac * account
    acc_rows, day_rows = [], []
    for fday, g in cands.groupby("fday", sort=True):
        opened: list[_Open] = []
        halted_at = None
        for i, r in g.iterrows():
            te = int(r["entry_t"])
            if halted_at is not None:
                break
            # stop journalier interne déclenché avant cette entrée ?
            trig = _trigger_time(opened, until=te, limit=-C.RISK["day_stop"] * account)
            if trig is not None:
                halted_at = trig
                break
            live = [o for o in opened if o.exit_t > te]
            if sum(o.risk for o in live) + risk_usd > C.RISK["max_open_risk"] * account + 1e-9:
                continue
            side = C.INSTRUMENTS[r["symbol"]]["usd_side"] * int(r["dir"])
            if side != 0 and sum(1 for o in live if C.INSTRUMENTS[o.sym]["usd_side"] * o.d == side) >= C.RISK["max_same_usd_side"]:
                continue
            if any(o.sym == r["symbol"] for o in live):
                continue
            bloc = C.INSTRUMENTS[r["symbol"]].get("bloc")       # actifs quasi identiques (ex. indices US)
            if bloc and sum(1 for o in live if C.INSTRUMENTS[o.sym].get("bloc") == bloc and o.d == int(r["dir"])) \
                    >= C.RISK.get("max_same_bloc", 99):
                continue
            u, risk_real, notional = _units(r["symbol"], specs[r["symbol"]], float(r["R"]), float(r["entry"]), risk_usd)
            if u <= 0:
                continue
            lev = C.INSTRUMENTS[r["symbol"]]["leverage"]
            used = sum(o.notional / C.INSTRUMENTS[o.sym]["leverage"] for o in live)
            room = account - used                      # equity ~ solde (approximation prudente : pas de levier sur les gains)
            if notional / lev > room:
                scale = max(room * lev / notional, 0.0)
                u2, risk_real, notional = _units(r["symbol"], specs[r["symbol"]], float(r["R"]), float(r["entry"]),
                                                 risk_real * scale)
                if u2 <= 0:
                    continue
            pt, pu = paths[r["pkey"] if "pkey" in r.index else r["symbol"]]
            p0, pn = int(r["p0"]), int(r["pn"])
            opened.append(_Open(i=i, sym=r["symbol"], d=int(r["dir"]), risk=risk_real, notional=notional,
                                exit_t=int(r["exit_t"]), pt=pt[p0:p0 + pn], pu=pu[p0:p0 + pn], rnet=float(r["r_net"])))
        # fin de journée : stop interne éventuellement déclenché par les trades déjà ouverts
        if halted_at is None:
            halted_at = _trigger_time(opened, until=None, limit=-C.RISK["day_stop"] * account)
        times = _grid(opened)
        worst = 0.0
        for o in opened:
            r_eff = o.rnet
            ex_t = o.exit_t
            cut = False
            if halted_at is not None and o.exit_t > halted_at:
                r_eff = o.value(halted_at)
                ex_t = halted_at
                cut = True
            acc_rows.append(dict(**cands.loc[o.i].to_dict(), risk_usd=o.risk, r_eff=r_eff, pnl=r_eff * o.risk,
                                 exit_eff=ex_t, cut_by_day_stop=cut))
        if len(times):
            if halted_at is not None:
                times = times[times <= halted_at]
            eq = sum(o.risk * o.values(times) for o in opened) if len(times) else np.zeros(0)
            worst = float(min(eq.min(), 0.0)) if len(eq) else 0.0
        pnl = sum(a["pnl"] for a in acc_rows[len(acc_rows) - len(opened):]) if opened else 0.0
        day_rows.append(dict(fday=fday, n=len(opened), pnl=pnl, worst=worst, R=pnl / base_usd,
                             worst_R=worst / base_usd, halted=halted_at is not None))
    return pd.DataFrame(acc_rows), pd.DataFrame(day_rows)


def _grid(opened):
    if not opened:
        return np.zeros(0, np.int64)
    return np.unique(np.concatenate([o.pt for o in opened]))


def _trigger_time(opened, until, limit):
    if not opened:
        return None
    times = _grid(opened)
    if until is not None:
        times = times[times < until]
    if not len(times):
        return None
    eq = sum(o.risk * o.values(times) for o in opened)
    hit = np.nonzero(eq <= limit)[0]
    return int(times[hit[0]]) if len(hit) else None
