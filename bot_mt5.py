"""Bot MT5 : exécute UNE stratégie de strategies.py (même code que le backtest) sur un compte DÉMO.

Prérequis : Windows, terminal MT5 ouvert et connecté, « Algo Trading » activé, `pip install MetaTrader5`.
Usage     : python bot_mt5.py --list
            python bot_mt5.py --symbol BTCUSD --strategy donch_1h_n48_k2 --spec mt5_export/BTCUSD_spec.json
            (journalise seulement)  ->  ajouter --send-orders pour envoyer réellement les ordres (compte démo)

Garde-fous : refuse un compte réel sans --allow-real ; risque par trade en % de l'équité avec plafond de levier ;
stop broker posé à chaque entrée ; filtre de spread ; perte journalière maximale ; fichier STOP pour tout fermer.
Au premier lancement la barre courante est marquée « traitée » (pas d'entrée tardive) : le bot n'agit qu'à partir
du prochain changement de signal (--act-on-first-bar pour désactiver cela)."""
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd

import common
import strategies

try:
    import MetaTrader5 as _mt5
except ImportError:
    _mt5 = None


class Bot:
    def __init__(self, mt5, cfg, a, now_fn=None):
        self.mt5, self.cfg, self.a = mt5, cfg, a
        self.now = now_fn or (lambda: pd.Timestamp.now(tz="UTC").tz_localize(None))
        self.state_path = Path(a.state_dir) / f"state_{a.magic}.json"
        self.log_path = Path(a.state_dir) / f"bot_log_{a.magic}.csv"
        Path(a.state_dir).mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.info = None

    # ------------------------------------------------------------------ utilitaires
    def log(self, event, **kw):
        row = dict(t=str(self.now()), event=event, **kw)
        print(json.dumps(row, default=str), flush=True)
        new = not self.log_path.exists()
        with open(self.log_path, "a", encoding="utf-8") as f:
            if new:
                f.write("t;event;details\n")
            f.write(f"{row['t']};{event};{json.dumps(kw, default=str)}\n")

    def save(self):
        self.state_path.write_text(json.dumps(self.state))

    def to_utc(self, epoch_seconds) -> pd.DatetimeIndex:
        return common.server_to_utc(pd.to_datetime(pd.Series(epoch_seconds), unit="s"), self.a.tz_mode, self.a.tz_hours)

    # ------------------------------------------------------------------ connexion et contrôles
    def connect(self):
        m = self.mt5
        if not m.initialize():
            raise SystemExit(f"mt5.initialize() a échoué : {m.last_error()}")
        acc, term = m.account_info(), m.terminal_info()
        if acc is None:
            raise SystemExit("Pas de compte connecté dans le terminal MT5.")
        if acc.trade_mode != 0 and not self.a.allow_real:
            raise SystemExit("Compte NON démo détecté : refus (relancez avec --allow-real si c'est voulu).")
        if self.a.send_orders and not getattr(term, "trade_allowed", True):
            raise SystemExit("« Algo Trading » est désactivé dans le terminal MT5.")
        if not m.symbol_select(self.a.symbol, True):
            raise SystemExit(f"Symbole {self.a.symbol} introuvable.")
        self.info = m.symbol_info(self.a.symbol)
        if self.info.currency_profit != acc.currency:
            raise SystemExit(f"Devise de profit {self.info.currency_profit} != devise du compte {acc.currency} : "
                             "le dimensionnement suppose les deux identiques.")
        self.log("connect", account=acc.login, mode=acc.trade_mode, equity=acc.equity, symbol=self.a.symbol,
                 strategy=self.cfg.name, send_orders=self.a.send_orders, contract=self.info.trade_contract_size,
                 vol_min=self.info.volume_min, vol_step=self.info.volume_step, stops_level=self.info.trade_stops_level)

    def fetch_m1(self):
        m = self.mt5
        count = (self.cfg.warmup + 10) * self.cfg.tf_min + self.cfg.tf_min
        rates = m.copy_rates_from_pos(self.a.symbol, m.TIMEFRAME_M1, 1, count)   # pos 1 : écarte la bougie en cours
        if rates is None or len(rates) < 0.9 * count:
            self.log("data_insufficient", got=None if rates is None else len(rates), need=count)
            return None
        df = pd.DataFrame(rates)
        idx = self.to_utc(df["time"])
        out = pd.DataFrame({"open": df["open"].to_numpy(float), "high": df["high"].to_numpy(float),
                            "low": df["low"].to_numpy(float), "close": df["close"].to_numpy(float),
                            "spread_px": df["spread"].to_numpy(float) * self.info.point,
                            "tick_volume": df["tick_volume"].to_numpy(float)}, index=pd.DatetimeIndex(idx, name="time"))
        out = out[~out.index.duplicated()].sort_index()
        lag = (self.now() - out.index[-1]).total_seconds() / 60
        if lag < -3:
            raise SystemExit(f"Heure incohérente (dernière bougie {-lag:.0f} min dans le futur) : vérifiez --tz-mode/--tz-hours.")
        if lag > 15:
            self.log("data_stale", minutes=round(lag, 1))
            return None
        return out

    # ------------------------------------------------------------------ position et ordres
    def get_position(self):
        ps = [p for p in (self.mt5.positions_get(symbol=self.a.symbol) or []) if p.magic == self.a.magic]
        if not ps:
            return None
        if len(ps) > 1:
            self.log("warning_multiple_positions", n=len(ps))
        p = ps[0]
        return dict(side=1 if p.type == self.mt5.POSITION_TYPE_BUY else -1, ticket=p.ticket, volume=p.volume,
                    price=p.price_open, sl=p.sl, time=self.to_utc([p.time])[0])

    def _fill(self):
        fm = self.info.filling_mode
        return self.mt5.ORDER_FILLING_IOC if fm & 2 else (self.mt5.ORDER_FILLING_FOK if fm & 1 else self.mt5.ORDER_FILLING_RETURN)

    def _send(self, req, what):
        if not self.a.send_orders:
            self.log("DRY_RUN", what=what, request={k: v for k, v in req.items() if k != "comment"})
            return None
        res = self.mt5.order_send(req)
        ok = res is not None and res.retcode in (self.mt5.TRADE_RETCODE_DONE, self.mt5.TRADE_RETCODE_PLACED)
        self.log("order_ok" if ok else "ORDER_FAILED", what=what, retcode=None if res is None else res.retcode,
                 price=None if res is None else res.price, last_error=None if ok else self.mt5.last_error())
        return res if ok else None

    def lots_for(self, price, stop_dist):
        acc, info = self.mt5.account_info(), self.info
        cap = acc.equity * self.a.max_leverage / (price * info.trade_contract_size)
        lots = acc.equity * self.a.risk_pct / 100 / (stop_dist * info.trade_contract_size) if stop_dist > 0 else cap
        lots = min(lots, cap, info.volume_max)
        lots = math.floor(lots / info.volume_step + 1e-9) * info.volume_step
        lots = round(lots, 8)
        return lots if lots >= info.volume_min else 0.0

    def open_position(self, side, stop_dist):
        m, info = self.mt5, self.info
        tick = m.symbol_info_tick(self.a.symbol)
        price = tick.ask if side == 1 else tick.bid
        spread = tick.ask - tick.bid
        ref = self.a.ref_spread_bps / 1e4 * price
        if spread > self.a.max_spread_mult * ref:
            return self.log("skip_entry_spread", spread=round(spread, 2), max=round(self.a.max_spread_mult * ref, 2))
        lots = self.lots_for(price, stop_dist)
        if lots <= 0:
            return self.log("skip_entry_size", reason="lot minimum > risque autorisé", stop_dist=round(stop_dist, 2))
        sl = 0.0
        if stop_dist > 0:
            sl = round(price - stop_dist if side == 1 else price + stop_dist, info.digits)
            if abs(price - sl) < info.trade_stops_level * info.point:
                return self.log("skip_entry_stop_too_close", stops_level=info.trade_stops_level)
        req = dict(action=m.TRADE_ACTION_DEAL, symbol=self.a.symbol, volume=lots,
                   type=m.ORDER_TYPE_BUY if side == 1 else m.ORDER_TYPE_SELL, price=price, sl=sl, tp=0.0,
                   deviation=self.a.deviation, magic=self.a.magic, comment="bot_mt5", type_time=m.ORDER_TIME_GTC,
                   type_filling=self._fill())
        self._send(req, f"open {'BUY' if side == 1 else 'SELL'} {lots} @ {price} sl {sl}")

    def close_position(self, pos, reason):
        m = self.mt5
        tick = m.symbol_info_tick(self.a.symbol)
        buy = pos["side"] == -1
        req = dict(action=m.TRADE_ACTION_DEAL, symbol=self.a.symbol, volume=pos["volume"],
                   type=m.ORDER_TYPE_BUY if buy else m.ORDER_TYPE_SELL, position=pos["ticket"],
                   price=tick.ask if buy else tick.bid, deviation=self.a.deviation, magic=self.a.magic,
                   comment="bot_mt5", type_time=m.ORDER_TIME_GTC, type_filling=self._fill())
        self._send(req, f"close ({reason}) ticket {pos['ticket']}")

    def trail_stop(self, pos, sig):
        m, info = self.mt5, self.info
        if pos["side"] == 1:
            new = sig["high"] - sig["stop"]
            better = pos["sl"] == 0 or new > pos["sl"]
        else:
            new = sig["low"] + sig["spread"] + sig["stop"]
            better = pos["sl"] == 0 or new < pos["sl"]
        if sig["stop"] <= 0 or not better:
            return
        tick = m.symbol_info_tick(self.a.symbol)
        gap = info.trade_stops_level * info.point
        if (pos["side"] == 1 and new > tick.bid - gap) or (pos["side"] == -1 and new < tick.ask + gap):
            return self.log("trail_skipped_too_close", new_sl=new)
        req = dict(action=m.TRADE_ACTION_SLTP, symbol=self.a.symbol, position=pos["ticket"],
                   sl=round(new, info.digits), tp=0.0, magic=self.a.magic)
        self._send(req, f"trail sl -> {round(new, info.digits)}")

    # ------------------------------------------------------------------ gardes
    def guard(self):
        """Perte journalière maximale : ferme et suspend jusqu'au lendemain (UTC). Renvoie True si suspendu."""
        today = str(self.now().date())
        eq = self.mt5.account_info().equity
        if self.state.get("day") != today:
            self.state.update(day=today, day_equity=eq, halted=False)
            self.save()
        if not self.state.get("halted") and eq < self.state["day_equity"] * (1 - self.a.max_daily_loss_pct / 100):
            self.state["halted"] = True
            self.save()
            self.log("HALT_daily_loss", equity=eq, day_start=self.state["day_equity"])
            pos = self.get_position()
            if pos:
                self.close_position(pos, "daily loss limit")
        return bool(self.state.get("halted"))

    def flatten(self, reason):
        pos = self.get_position()
        if pos:
            self.close_position(pos, reason)

    # ------------------------------------------------------------------ une décision par barre
    def step(self):
        m1 = self.fetch_m1()
        if m1 is None:
            return "no-data"
        sig = strategies.latest_signal(self.cfg, m1)
        if sig is None:
            self.log("warmup_insufficient")
            return "warmup"
        bar = str(sig["bar_start"])
        if self.state.get("last_bar") == bar:
            return "already"
        if self.state.get("last_bar") is None and not self.a.act_on_first_bar:
            self.state["last_bar"] = bar
            self.save()
            self.log("init", bar=bar, tgt=sig["tgt"], note="barre courante marquée traitée, attente du prochain changement")
            return "init"
        halted = self.guard()
        pos = self.get_position()
        tf = pd.Timedelta(self.cfg.tf)
        self.log("bar", bar=bar, tgt=sig["tgt"], fresh=sig["fresh"], stop_dist=round(sig["stop"], 2),
                 pos=None if pos is None else pos["side"])
        if pos:
            held = (sig["bar_start"] + tf - pos["time"].floor(tf)) / tf
            timed_out = bool(self.cfg.max_hold) and held >= self.cfg.max_hold
            if sig["tgt"] != pos["side"] or timed_out:
                self.close_position(pos, "signal" if sig["tgt"] != pos["side"] else "time")
                pos = self.get_position() if self.a.send_orders else None
        opened = False
        if pos is None and sig["tgt"] != 0 and sig["fresh"] and not halted:
            self.open_position(sig["tgt"], sig["stop"])
            pos = self.get_position() if self.a.send_orders else None
            opened = True
        if pos and self.cfg.trail and not opened:
            # le stop suiveur ne se met à jour qu'avec une barre pendant laquelle la position était ouverte
            self.trail_stop(pos, sig)
        self.state["last_bar"] = bar
        self.save()
        return "done"

    def run_forever(self):
        tf = pd.Timedelta(self.cfg.tf)
        while True:
            if Path(self.a.kill_file).exists():
                self.log("KILL_FILE", note="fermeture de tout et arrêt")
                self.flatten("kill file")
                return
            self.guard()
            now = self.now()
            expected = now.floor(tf) - tf          # début de la dernière barre qui doit être complète
            last = self.state.get("last_bar")
            if (now - now.floor(tf)).total_seconds() >= self.a.settle_seconds and (last is None or pd.Timestamp(last) < expected):
                self.step()
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--symbol", default="BTCUSD")
    ap.add_argument("--strategy", help="nom exact (voir --list), de préférence celui retenu par le backtest")
    ap.add_argument("--spec", help="BTCUSD_spec.json exporté (utilisé pour le spread de référence)")
    ap.add_argument("--risk-pct", type=float, default=0.25, help="risque par trade en %% de l'équité (à la distance de stop)")
    ap.add_argument("--max-leverage", type=float, default=1.0, help="notionnel maximal / équité")
    ap.add_argument("--max-daily-loss-pct", type=float, default=2.0)
    ap.add_argument("--max-spread-mult", type=float, default=3.0, help="refuse d'entrer si spread > mult × spread de référence")
    ap.add_argument("--ref-spread-bps", type=float, default=None)
    ap.add_argument("--deviation", type=int, default=50, help="slippage max accepté, en points")
    ap.add_argument("--magic", type=int, default=26102901)
    ap.add_argument("--tz-mode", default="eet_us_dst", choices=["eet_us_dst", "fixed"])
    ap.add_argument("--tz-hours", type=int, default=0)
    ap.add_argument("--settle-seconds", type=int, default=5)
    ap.add_argument("--state-dir", default="bot_state")
    ap.add_argument("--kill-file", default="STOP")
    ap.add_argument("--send-orders", action="store_true", help="sans ce drapeau : journal seulement")
    ap.add_argument("--allow-real", action="store_true")
    ap.add_argument("--act-on-first-bar", action="store_true")
    a = ap.parse_args()
    cfgs = {c.name: c for c in strategies.default_configs()}
    if a.list:
        print("\n".join(cfgs))
        return
    if a.strategy not in cfgs:
        raise SystemExit("--strategy requis (voir --list)")
    if a.ref_spread_bps is None:
        a.ref_spread_bps = json.loads(Path(a.spec).read_text())["spread_median_bps"] if a.spec else 3.0
    if _mt5 is None:
        raise SystemExit("Module MetaTrader5 absent : pip install MetaTrader5 (Windows).")
    bot = Bot(_mt5, cfgs[a.strategy], a)
    bot.connect()
    try:
        bot.run_forever()
    finally:
        _mt5.shutdown()


if __name__ == "__main__":
    main()
