"""Environnement intraday EURUSD vectorisé (numba).

Actions : 0 WAIT (ou conserver la position), 1 BUY, 2 SELL, 3 CLOSE.
Une seule position. Entrée/sortie à l'ouverture de la barre M1 de remplissage (latence configurable).
Simulation M1 conservative entre deux décisions :
  - ask = bid + spread de coût ; achat au ask + glissement, vente au bid - glissement ;
  - stop et TP dans la même barre -> stop ; ouverture au-delà du stop (gap) -> sortie à l'ouverture - glissement ;
    ouverture au-delà du TP -> sortie AU TP (jamais mieux) ;
  - clôture forcée à l'ouverture de la barre de clôture du jour ; aucune position au rollover.
Limites : perte journalière FTMO (base = solde de début de journée ; le jour FTMO Prague coïncide avec notre
journée car aucune position n'est ouverte entre 16:45 NY et 02:00 NY), perte totale statique, arrêts internes
jour/semaine/total, nombre de trades par jour, contrôles de taille et de marge.
Récompense de l'étape A : variation d'équité en R (risque nominal) - pénalité de violation.
"""
from __future__ import annotations

import numba as nb
import numpy as np

from .market import Market, gather_seq, CTX_FEATS

# indices des constantes
(K_INIT, K_RISK, K_COMM, K_CS, K_LEV, K_MARG, K_DHALT, K_WHALT, K_THALT, K_DLOSS, K_TLOSS, K_TPR,
 K_SIN, K_SOUT, K_SSTOP, K_LAT, K_MAXTR, K_VPEN, K_LOTSTEP, K_LOTMIN, K_PT, K_SWL, K_SWS, K_SW3) = range(24)
# indices de l'état flottant
(S_BAL, S_EQ, S_DAYB, S_WEEKB, S_ENTRY, S_STOP, S_TP, S_LOTS, S_STOPD, S_SWAP) = range(10)
N_FSTATE = 10
# état entier
(I_POS, I_TRADES, I_HDAY, I_HWEEK, I_HTOT, I_VIOL, I_ENTRYDEC, I_PTR, I_END, I_NBLOCK, I_NINVALID) = range(11)
# raisons de sortie
R_STOP, R_TP, R_CLOSE, R_FORCE, R_VIOL = 1, 2, 3, 4, 5
N_ACC = 10   # features de compte


def make_consts(cfg: dict) -> np.ndarray:
    ch, ir, ex, tr = cfg["challenge"], cfg["internal_risk"], cfg["execution"], cfg["trade"]
    pt = cfg["data"]["point"]
    k = np.zeros(24)
    k[K_INIT] = ch["initial_capital"]; k[K_RISK] = ir["risk_per_trade"] * ch["initial_capital"]
    k[K_COMM] = ex["commission_per_lot_side"]; k[K_CS] = cfg["data"]["contract_size"]; k[K_LEV] = ch["leverage"]
    k[K_MARG] = ir["max_margin_usage"]; k[K_DHALT] = ir["daily_halt"]; k[K_WHALT] = ir["weekly_halt"]
    k[K_THALT] = ir["total_halt"]; k[K_DLOSS] = ch["max_daily_loss"]; k[K_TLOSS] = ch["max_total_loss"]
    k[K_TPR] = tr["take_profit_r"]; k[K_SIN] = ex["entry_slippage_pts"] * pt; k[K_SOUT] = ex["exit_slippage_pts"] * pt
    k[K_SSTOP] = ex["stop_slippage_pts"] * pt; k[K_LAT] = ex["latency_bars_m1"]; k[K_MAXTR] = ir["max_trades_per_day"]
    k[K_VPEN] = cfg["reward"]["violation_penalty_r"]; k[K_LOTSTEP] = 0.01; k[K_LOTMIN] = 0.01; k[K_PT] = pt
    # swaps en USD par lot et par nuit (EURUSD : points x point x taille de contrat)
    k[K_SWL] = cfg["data"].get("swap_long_pts", 0.0) * pt * cfg["data"]["contract_size"]
    k[K_SWS] = cfg["data"].get("swap_short_pts", 0.0) * pt * cfg["data"]["contract_size"]
    k[K_SW3] = cfg["data"].get("swap_triple_weekday", 2)
    return k


@nb.njit(cache=True)
def lots_for(k, stop_dist, price, equity):
    """Taille en lots pour risquer K_RISK sur stop_dist ; 0 si invalide (taille, marge, stop)."""
    if not (stop_dist > 0.0):
        return 0.0
    raw = k[K_RISK] / (stop_dist * k[K_CS])
    lots = np.floor(raw / k[K_LOTSTEP] + 1e-9) * k[K_LOTSTEP]
    if lots < k[K_LOTMIN]:
        return 0.0
    margin = lots * k[K_CS] * price / k[K_LEV]          # EURUSD : marge en EUR convertie en USD au prix
    if margin > k[K_MARG] * equity:
        return 0.0
    return lots


@nb.njit(cache=True)
def _close(fs, ist, k, px, j, reason, log, nlog, e, dec):
    pos = ist[I_POS]
    pnl = pos * (px - fs[S_ENTRY]) * fs[S_LOTS] * k[K_CS] - fs[S_LOTS] * k[K_COMM]
    fs[S_BAL] += pnl
    if log.shape[0] > 0:
        n = nlog[e]
        if n < log.shape[1]:
            log[e, n, 0] = ist[I_ENTRYDEC]; log[e, n, 1] = dec; log[e, n, 2] = j; log[e, n, 3] = pos
            log[e, n, 4] = fs[S_ENTRY]; log[e, n, 5] = px; log[e, n, 6] = fs[S_LOTS]; log[e, n, 7] = fs[S_STOPD]
            log[e, n, 8] = reason; log[e, n, 9] = fs[S_BAL]
            # pnl net (commission d'entrée incluse) en USD et en R nominal
            gross = pos * (px - fs[S_ENTRY]) * fs[S_LOTS] * k[K_CS]
            log[e, n, 10] = gross - 2 * fs[S_LOTS] * k[K_COMM] + fs[S_SWAP]
            log[e, n, 11] = log[e, n, 10] / k[K_RISK]
            nlog[e] = n + 1
    ist[I_POS] = 0
    fs[S_LOTS] = 0.0
    fs[S_SWAP] = 0.0


@nb.njit(cache=True)
def _sim_bars(fs, ist, k, o, h, l, c, sp, j0, j1, entry_bar, log, nlog, e, dec, day, wday):
    """Simule les barres [j0, j1). Retourne 1 si violation FTMO. Swap appliqué au passage de minuit serveur."""
    for j in range(j0, j1):
        pos = ist[I_POS]
        if pos == 0:
            return 0
        if j != entry_bar and j > 0 and day[j] != day[j - 1]:
            mult = 3.0 if wday[j - 1] == k[K_SW3] else 1.0
            sw = mult * fs[S_LOTS] * (k[K_SWL] if pos == 1 else k[K_SWS])
            fs[S_SWAP] += sw
            fs[S_BAL] += sw
        stop, tp = fs[S_STOP], fs[S_TP]
        if pos == 1:
            if j != entry_bar and o[j] <= stop:
                _close(fs, ist, k, o[j] - k[K_SSTOP], j, R_STOP, log, nlog, e, dec)
            elif j != entry_bar and o[j] >= tp:
                _close(fs, ist, k, tp, j, R_TP, log, nlog, e, dec)
            elif l[j] <= stop:
                _close(fs, ist, k, stop - k[K_SSTOP], j, R_STOP, log, nlog, e, dec)
            elif h[j] >= tp:
                _close(fs, ist, k, tp, j, R_TP, log, nlog, e, dec)
            else:
                worst = fs[S_BAL] + (l[j] - fs[S_ENTRY]) * fs[S_LOTS] * k[K_CS]
                if worst <= fs[S_DAYB] - k[K_DLOSS] * k[K_INIT] or worst <= k[K_INIT] * (1 - k[K_TLOSS]):
                    _close(fs, ist, k, l[j], j, R_VIOL, log, nlog, e, dec)
                    return 1
        else:
            ao, ah, al = o[j] + sp[j], h[j] + sp[j], l[j] + sp[j]
            if j != entry_bar and ao >= stop:
                _close(fs, ist, k, ao + k[K_SSTOP], j, R_STOP, log, nlog, e, dec)
            elif j != entry_bar and ao <= tp:
                _close(fs, ist, k, tp, j, R_TP, log, nlog, e, dec)
            elif ah >= stop:
                _close(fs, ist, k, stop + k[K_SSTOP], j, R_STOP, log, nlog, e, dec)
            elif al <= tp:
                _close(fs, ist, k, tp, j, R_TP, log, nlog, e, dec)
            else:
                worst = fs[S_BAL] - (ah - fs[S_ENTRY]) * fs[S_LOTS] * k[K_CS]
                if worst <= fs[S_DAYB] - k[K_DLOSS] * k[K_INIT] or worst <= k[K_INIT] * (1 - k[K_TLOSS]):
                    _close(fs, ist, k, ah, j, R_VIOL, log, nlog, e, dec)
                    return 1
        # violation juste après une sortie (gap extrême)
        if fs[S_BAL] <= fs[S_DAYB] - k[K_DLOSS] * k[K_INIT] or fs[S_BAL] <= k[K_INIT] * (1 - k[K_TLOSS]):
            return 1
    return 0


@nb.njit(cache=True)
def _equity(fs, ist, k, c, sp, j):
    pos = ist[I_POS]
    if pos == 0:
        return fs[S_BAL]
    px = c[j] if pos == 1 else c[j] + sp[j]
    return fs[S_BAL] + pos * (px - fs[S_ENTRY]) * fs[S_LOTS] * k[K_CS]


@nb.njit(cache=True)
def entry_allowed(fs, ist, k, can_enter, stop_d, o, d, jfill):
    if not can_enter[d] or ist[I_HDAY] or ist[I_HWEEK] or ist[I_HTOT] or ist[I_TRADES] >= k[K_MAXTR]:
        return False
    return lots_for(k, stop_d[d], o[jfill], fs[S_EQ]) > 0.0


@nb.njit(cache=True)
def step_all(F, I, k, actions, o, h, l, c, sp, dec_m1, dec_next, dec_force, dec_day, dec_week, can_enter,
             is_last, stop_d, log, nlog, rewards, dones, viols, day, wday):
    """is_last : clôture forcée à cette décision (dec_force du marché)."""
    n_env = F.shape[0]
    for e in range(n_env):
        fs, ist = F[e], I[e]
        d = ist[I_PTR]
        eq0 = fs[S_EQ]
        a = actions[e]
        j = dec_m1[d]
        jn = dec_next[d]
        jfill = min(j + int(k[K_LAT]), max(jn - 1, j))
        viol = 0
        entry_bar = -1
        if is_last[d]:
            # clôture forcée à l'ouverture de jf ; simuler [j, jf) d'abord
            viol = _sim_bars(fs, ist, k, o, h, l, c, sp, j, jn, -1, log, nlog, e, d, day, wday)
            if viol == 0 and ist[I_POS] != 0:
                jf = dec_force[d]
                px = o[jf] - k[K_SOUT] if ist[I_POS] == 1 else o[jf] + sp[jf] + k[K_SOUT]
                _close(fs, ist, k, px, jf, R_FORCE, log, nlog, e, d)
        else:
            if ist[I_POS] == 0 and (a == 1 or a == 2):
                if entry_allowed(fs, ist, k, can_enter, stop_d, o, d, jfill):
                    side = 1 if a == 1 else -1
                    sd = stop_d[d]
                    lots = lots_for(k, sd, o[jfill], fs[S_EQ])
                    px = o[jfill] + sp[jfill] + k[K_SIN] if side == 1 else o[jfill] - k[K_SIN]
                    ist[I_POS] = side
                    fs[S_ENTRY] = px; fs[S_LOTS] = lots; fs[S_STOPD] = sd
                    fs[S_STOP] = px - side * sd
                    fs[S_TP] = px + side * k[K_TPR] * sd
                    fs[S_BAL] -= lots * k[K_COMM]
                    ist[I_TRADES] += 1
                    ist[I_ENTRYDEC] = d
                    entry_bar = jfill
                else:
                    ist[I_NBLOCK] += 1
            elif ist[I_POS] != 0 and a == 3:
                viol = _sim_bars(fs, ist, k, o, h, l, c, sp, j, jfill, -1, log, nlog, e, d, day, wday)  # latence
                if viol == 0 and ist[I_POS] != 0:
                    px = o[jfill] - k[K_SOUT] if ist[I_POS] == 1 else o[jfill] + sp[jfill] + k[K_SOUT]
                    _close(fs, ist, k, px, jfill, R_CLOSE, log, nlog, e, d)
            elif a != 0:
                ist[I_NINVALID] += 1
            if viol == 0:
                start = entry_bar if entry_bar >= 0 else (jfill + 1 if a == 3 else j)
                viol = _sim_bars(fs, ist, k, o, h, l, c, sp, start, jn, entry_bar, log, nlog, e, d, day, wday)
        jlast = max(jn - 1, j)
        if is_last[d]:
            jlast = dec_force[d]
        fs[S_EQ] = _equity(fs, ist, k, c, sp, jlast)
        r = (fs[S_EQ] - eq0) / k[K_RISK]
        if viol:
            ist[I_VIOL] += 1
            r -= k[K_VPEN]
        # limites internes (équité flottante incluse)
        if fs[S_EQ] <= fs[S_DAYB] - k[K_DHALT] * k[K_INIT]:
            ist[I_HDAY] = 1
        if fs[S_EQ] <= fs[S_WEEKB] - k[K_WHALT] * k[K_INIT]:
            ist[I_HWEEK] = 1
        if fs[S_EQ] <= k[K_INIT] * (1 - k[K_THALT]):
            ist[I_HTOT] = 1
        rewards[e] = r
        viols[e] = viol
        nd = d + 1
        done = viol == 1 or nd >= ist[I_END]
        if not done:
            if dec_day[nd] != dec_day[d]:
                fs[S_DAYB] = fs[S_BAL]; ist[I_TRADES] = 0; ist[I_HDAY] = 0
                if dec_week[nd] != dec_week[d]:
                    fs[S_WEEKB] = fs[S_BAL]; ist[I_HWEEK] = 0
        ist[I_PTR] = nd
        dones[e] = done


class VecEnv:
    """N environnements indépendants sur une même plage de décisions [lo, hi)."""

    def __init__(self, mk: Market, cfg: dict, n_env: int, lo: int, hi: int, episode_days: int = 5,
                 seed: int = 0, log_capacity: int = 0, sequential: bool = False):
        self.mk, self.k = mk, make_consts(cfg)
        self.n, self.lo, self.hi, self.ep_days = n_env, lo, hi, episode_days
        self.rng = np.random.default_rng(seed)
        self.F = np.zeros((n_env, N_FSTATE))
        self.I = np.zeros((n_env, 11), dtype=np.int64)
        self.log = np.zeros((n_env, log_capacity, 12))
        self.nlog = np.zeros(n_env, dtype=np.int64)
        self.sequential = sequential
        days = mk.dec_day[lo:hi]
        self.day_starts = lo + np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
        self.day_starts_next = np.r_[self.day_starts[1:], hi]
        self._seq_pos = 0
        self.rew = np.zeros(n_env); self.done = np.zeros(n_env, dtype=np.bool_); self.viol = np.zeros(n_env, dtype=np.int64)
        for e in range(n_env):
            self._reset(e)

    def _reset(self, e):
        k = self.k
        if self.sequential:
            a, b = self.lo, self.hi
        else:
            # début d'épisode : un jour aléatoire ; fin : ep_days jours plus tard (ou fin de plage)
            i = self.rng.integers(0, len(self.day_starts))
            a = self.day_starts[i]
            b = self.day_starts_next[min(i + self.ep_days - 1, len(self.day_starts) - 1)]
        self.F[e] = 0.0
        self.F[e, S_BAL] = self.F[e, S_EQ] = self.F[e, S_DAYB] = self.F[e, S_WEEKB] = k[K_INIT]
        self.I[e] = 0
        self.I[e, I_PTR], self.I[e, I_END] = a, b

    # ------------------------------------------------------------------ observation
    def masks(self) -> np.ndarray:
        mk, k = self.mk, self.k
        m = np.zeros((self.n, 4), dtype=np.bool_)
        m[:, 0] = True
        for e in range(self.n):
            d = self.I[e, I_PTR]
            if mk.dec_force[d]:
                continue
            if self.I[e, I_POS] == 0:
                jfill = min(mk.dec_m1[d] + int(k[K_LAT]), max(mk.dec_next_m1[d] - 1, mk.dec_m1[d]))
                ok = entry_allowed(self.F[e], self.I[e], k, mk.dec_can_enter, mk.dec_stop_dist, mk.o, d, jfill)
                m[e, 1] = m[e, 2] = ok
            else:
                m[e, 3] = True
        return m

    def account_feats(self) -> np.ndarray:
        mk, k, F, I = self.mk, self.k, self.F, self.I
        d = I[:, I_PTR]
        jm = np.maximum(mk.dec_m1[d] - 1, 0)
        pos = I[:, I_POS].astype(float)
        bid = mk.c[jm]
        px = np.where(pos >= 0, bid, bid + mk.spread[jm])
        sd = np.where(F[:, S_STOPD] > 0, F[:, S_STOPD], 1.0)
        unreal = pos * (px - F[:, S_ENTRY]) / sd                     # en R de la position
        in_pos = pos != 0
        R = k[K_RISK]
        bars_in = np.where(in_pos, (d - I[:, I_ENTRYDEC]) / 50.0, 0.0)
        f = np.c_[pos, np.where(in_pos, unreal, 0.0), np.where(in_pos, 1.0 + unreal, 0.0) / 2,
                  np.where(in_pos, k[K_TPR] - unreal, 0.0) / 4, bars_in,
                  (F[:, S_EQ] - F[:, S_DAYB]) / R / 3, (F[:, S_EQ] - F[:, S_WEEKB]) / R / 8,
                  (F[:, S_EQ] - k[K_INIT]) / R / 20, I[:, I_TRADES] / k[K_MAXTR],
                  (I[:, I_HDAY] | I[:, I_HWEEK] | I[:, I_HTOT]).astype(float)]
        return np.clip(f, -10, 10).astype(np.float32)

    def obs(self) -> dict:
        d = self.I[:, I_PTR]
        return {"M5": gather_seq(self.mk, "M5", d), "H1": gather_seq(self.mk, "H1", d),
                "ctx": self.mk.ctx[d], "acc": self.account_feats(), "mask": self.masks()}

    # ------------------------------------------------------------------ step
    def step(self, actions: np.ndarray):
        mk = self.mk
        step_all(self.F, self.I, self.k, actions.astype(np.int64), mk.o, mk.h, mk.l, mk.c, mk.spread,
                 mk.dec_m1, mk.dec_next_m1, mk.dec_force_m1, mk.dec_day, mk.dec_week, mk.dec_can_enter,
                 mk.dec_force, mk.dec_stop_dist, self.log, self.nlog, self.rew, self.done, self.viol,
                 mk.m1_day, mk.m1_wday)
        rew, done, viol = self.rew.copy(), self.done.copy(), self.viol.copy()
        info = {"equity": self.F[:, S_EQ].copy()}
        for e in np.flatnonzero(done):
            if not self.sequential:
                self._reset(e)
            elif viol[e] and self.I[e, I_PTR] < self.I[e, I_END]:
                # évaluation : la violation est comptée, le compte est réinitialisé, la chronologie continue
                k = self.k
                keep = self.I[e, [I_PTR, I_END, I_VIOL, I_NBLOCK, I_NINVALID]].copy()
                self.F[e] = 0.0
                self.F[e, S_BAL] = self.F[e, S_EQ] = self.F[e, S_DAYB] = self.F[e, S_WEEKB] = k[K_INIT]
                self.I[e] = 0
                self.I[e, [I_PTR, I_END, I_VIOL, I_NBLOCK, I_NINVALID]] = keep
                done[e] = False
        return rew, done, viol, info
