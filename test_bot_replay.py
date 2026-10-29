"""Test d'équivalence bot <-> backtest avec un faux terminal MT5 qui rejoue des bougies 1m.

Le bot (Bot.step) est appelé à chaque frontière de barre ; le faux courtier exécute les ordres au marché
à l'open de la minute courante et déclenche les stops broker minute par minute.
Contrôle : mêmes trades (côté, heure d'entrée, heure de sortie) que le moteur de backtest, à coûts de slippage nuls.

Usage : python test_bot_replay.py --configs donch_1h_n48_k2 ema_1h_12_48 orb_15m_s8_r2 bb_15m_n48_z2 --days 40"""
import argparse
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

import bot_mt5
import strategies
from data import make_synthetic
from engine import Costs, run

SERVER_OFFSET_H = 2


class FakeMT5:
    TIMEFRAME_M1 = 1
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    TRADE_ACTION_DEAL, TRADE_ACTION_SLTP = 1, 6
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED = 10009, 10008

    def __init__(self, m1: pd.DataFrame):
        self.m1, self.i = m1, 0
        self.o, self.h, self.l = (m1[c].to_numpy() for c in ("open", "high", "low"))
        self.sp = m1["spread_px"].to_numpy()
        self.pos, self.deals, self.balance, self.ticket = None, [], 100000.0, 1000
        self.epoch = ((m1.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s") + SERVER_OFFSET_H * 3600).to_numpy()
        self.rec = m1.assign(time=self.epoch, spread=(m1["spread_px"] / 0.01).round().astype(int),
                             real_volume=0).reset_index(drop=True)[
            ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]].to_records(index=False)

    # --- API MetaTrader5 (sous-ensemble utilisé par le bot)
    def initialize(self, *a, **k): return True
    def shutdown(self): pass
    def last_error(self): return (1, "ok")
    def symbol_select(self, s, e): return True
    def account_info(self):
        return SimpleNamespace(login=1, trade_mode=0, currency="USD", equity=self.balance)
    def terminal_info(self): return SimpleNamespace(trade_allowed=True)
    def symbol_info(self, s):
        return SimpleNamespace(point=0.01, digits=2, trade_contract_size=1.0, volume_min=0.001, volume_max=100.0,
                               volume_step=0.001, trade_stops_level=0, filling_mode=2, currency_profit="USD")
    def symbol_info_tick(self, s):
        return SimpleNamespace(bid=self.o[self.i], ask=self.o[self.i] + self.sp[self.i], time=int(self.epoch[self.i]))
    def copy_rates_from_pos(self, s, tf, pos, count):
        hi = self.i - (pos - 1)
        return self.rec[max(hi - count, 0):hi]
    def positions_get(self, symbol=None):
        return [self.pos] if self.pos is not None else []
    def order_send(self, req):
        t = self.m1.index[self.i]
        if req["action"] == self.TRADE_ACTION_SLTP:
            self.pos.sl = req["sl"]
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, price=0.0)
        if "position" in req:
            self._close(req["price"], t, "bot")
        else:
            self.ticket += 1
            self.pos = SimpleNamespace(ticket=self.ticket, type=req["type"], volume=req["volume"], price_open=req["price"],
                                       sl=req["sl"], time=int(self.epoch[self.i]), magic=req["magic"], t_open=t)
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, price=req["price"])

    # --- simulation
    def _close(self, px, t, how):
        p = self.pos
        side = 1 if p.type == self.POSITION_TYPE_BUY else -1
        self.balance += side * (px - p.price_open) * p.volume
        self.deals.append(dict(entry_time=p.t_open, exit_time=t, side=side, entry_px=p.price_open, exit_px=px, how=how))
        self.pos = None

    def advance_to(self, new_i):
        for k in range(self.i, new_i):
            p = self.pos
            if p is not None and p.sl:
                if p.type == self.POSITION_TYPE_BUY and self.l[k] <= p.sl:
                    self._close(min(self.o[k], p.sl), self.m1.index[k], "stop")
                elif p.type == self.POSITION_TYPE_SELL and self.h[k] + self.sp[k] >= p.sl:
                    self._close(max(self.o[k] + self.sp[k], p.sl), self.m1.index[k], "stop")
        self.i = new_i


def replay(cfg, m1, seg_start, seg_end, tmp):
    fake = FakeMT5(m1)
    args = SimpleNamespace(symbol="BTCUSD", magic=1, risk_pct=0.25, max_leverage=1.0, max_daily_loss_pct=50.0,
                           max_spread_mult=100.0, ref_spread_bps=2.0, deviation=50, tz_mode="fixed",
                           tz_hours=SERVER_OFFSET_H, settle_seconds=0, state_dir=tmp, kill_file="STOP_NONE",
                           send_orders=True, allow_real=False, act_on_first_bar=True)
    bot = bot_mt5.Bot(fake, cfg, args, now_fn=lambda: m1.index[fake.i])
    bot.connect()
    tf = pd.Timedelta(cfg.tf)
    bounds = [i for i in range(len(m1)) if seg_start <= m1.index[i] < seg_end and m1.index[i] == m1.index[i].floor(tf)]
    quiet = []
    import builtins
    for i in bounds:
        fake.advance_to(i)
        orig = builtins.print
        builtins.print = lambda *a, **k: None          # silence les journaux JSON pendant le rejeu
        try:
            bot.step()
        finally:
            builtins.print = orig
    fake.advance_to(bounds[-1])
    return pd.DataFrame(fake.deals), bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--mode", default="trend")
    a = ap.parse_args()
    m1 = make_synthetic(a.mode, days=330)
    seg_start = m1.index[0] + pd.Timedelta(days=200)
    seg_end = seg_start + pd.Timedelta(days=a.days)
    cfgs = {c.name: c for c in strategies.default_configs()}
    allok = True
    for name in a.configs:
        cfg = cfgs[name]
        with tempfile.TemporaryDirectory() as tmp:
            live, bounds = replay(cfg, m1, seg_start, seg_end, tmp)
        bars = strategies.complete_bars(m1, cfg.tf)
        tgt, stop = strategies.build_signals(cfg, bars)
        tgt[:cfg.warmup] = 0
        bt = run(bars, tgt, stop, Costs(slip_bps=0.0), cfg.trail, cfg.max_hold, start=cfg.warmup)
        tf = pd.Timedelta(cfg.tf)
        first, last = bounds[0], bounds[-1]
        t_first, t_last = m1.index[first], m1.index[last]
        # on compare les trades ENTRÉS dans le segment, sauf la 1re barre (le bot démarre à plat)
        bt_seg = bt[(bt.entry_time > t_first) & (bt.entry_time <= t_last - tf) & (bt.exit_time <= t_last)].reset_index(drop=True)
        lv = live[(live.entry_time > t_first) & (live.entry_time <= t_last - tf) & (live.exit_time <= t_last)].reset_index(drop=True)
        lv["exit_bar"] = lv["exit_time"].dt.floor(tf)
        n_bt, n_lv = len(bt_seg), len(lv)
        m = min(n_bt, n_lv)
        same = (bt_seg.loc[:m - 1, "entry_time"].to_numpy() == lv.loc[:m - 1, "entry_time"].to_numpy()).all() and \
               (bt_seg.loc[:m - 1, "side"].to_numpy() == lv.loc[:m - 1, "side"].to_numpy()).all() and \
               np.array_equal(bt_seg.loc[:m - 1, "exit_time"].to_numpy(), lv.loc[:m - 1, "exit_bar"].to_numpy())
        ok = n_bt == n_lv and same and n_bt > 0
        allok &= ok
        print(f"{name:<22} trades backtest={n_bt:>3} bot={n_lv:>3}  identiques={same}  -> {'OK' if ok else 'ÉCART'}")
        if not ok:
            print(bt_seg[["entry_time", "exit_time", "side", "reason"]].head(8).to_string())
            print(lv[["entry_time", "exit_time", "exit_bar", "side", "how"]].head(8).to_string())
    raise SystemExit(0 if allok else 1)


if __name__ == "__main__":
    main()
