"""Exporte l'historique M1 d'un symbole depuis votre terminal MT5 (Windows) + les spécifications du contrat.

Prérequis : terminal MT5 ouvert et connecté à votre compte (démo suffit), `pip install MetaTrader5`.
Usage     : python export_mt5.py --symbol BTCUSD --from 2023-01-01 --out mt5_export

Sortie    : <out>/<SYMBOL>_M1.csv  (time = epoch en HEURE SERVEUR, open/high/low/close en BID, spread_px en prix)
            <out>/<SYMBOL>_spec.json (point, taille de contrat, pas de lot, stops level, swaps, décalage serveur)
L'historique M1 disponible dépend du courtier (souvent 1 à 3 ans) ; MT5 peut devoir le télécharger
d'abord : ouvrez un graphique M1 du symbole et faites défiler vers la gauche si l'export revient court."""
import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSD")
    ap.add_argument("--from", dest="start", default="2023-01-01")
    ap.add_argument("--to", dest="end", default=None)
    ap.add_argument("--out", default="mt5_export")
    ap.add_argument("--chunk-days", type=int, default=15)
    a = ap.parse_args()
    if mt5 is None:
        raise SystemExit("Module MetaTrader5 absent : pip install MetaTrader5 (Windows, terminal MT5 requis).")
    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() a échoué : {mt5.last_error()}")
    try:
        sym = a.symbol
        if not mt5.symbol_select(sym, True):
            raise SystemExit(f"Symbole {sym} introuvable chez ce courtier (vérifiez le suffixe : BTCUSD, BTCUSD.a, BTCUSDm...).")
        info = mt5.symbol_info(sym)
        tick = mt5.symbol_info_tick(sym)
        acc, term = mt5.account_info(), mt5.terminal_info()
        off = round((tick.time - datetime.now(timezone.utc).timestamp()) / 3600)
        t0 = pd.Timestamp(a.start, tz="UTC")
        t1 = pd.Timestamp(a.end, tz="UTC") if a.end else pd.Timestamp.now(tz="UTC")
        parts, cur = [], t0
        while cur < t1:
            nxt = min(cur + timedelta(days=a.chunk_days), t1)
            r = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M1, cur.to_pydatetime(), nxt.to_pydatetime())
            if r is not None and len(r):
                parts.append(pd.DataFrame(r))
            cur = nxt
        if not parts:
            raise SystemExit("Aucune donnée reçue : ouvrez un graphique M1 du symbole pour déclencher le téléchargement.")
        df = pd.concat(parts, ignore_index=True).drop_duplicates("time").sort_values("time")
        df["spread_px"] = df["spread"].astype(float) * info.point
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        df[["time", "open", "high", "low", "close", "tick_volume", "spread_px", "real_volume"]].to_csv(
            out / f"{sym}_M1.csv", index=False)
        px = float(df["close"].iloc[-1])
        spec = dict(symbol=sym, broker=getattr(term, "company", None), server=getattr(acc, "server", None),
                    account_mode={0: "DEMO", 1: "CONCOURS", 2: "REEL"}.get(getattr(acc, "trade_mode", -1), "?"),
                    digits=info.digits, point=info.point, contract_size=info.trade_contract_size,
                    volume_min=info.volume_min, volume_max=info.volume_max, volume_step=info.volume_step,
                    stops_level_points=info.trade_stops_level, freeze_level_points=info.trade_freeze_level,
                    swap_mode=info.swap_mode, swap_long=info.swap_long, swap_short=info.swap_short,
                    swap_rollover3days=info.swap_rollover3days, spread_float=info.spread_float,
                    spread_now_points=info.spread, last_close=px,
                    spread_median_bps=round(float(np.median(df["spread_px"]) / df["close"].median() * 1e4), 2),
                    server_offset_hours_now=off, rows=len(df),
                    first_bar_server=str(pd.to_datetime(df["time"].iloc[0], unit="s")),
                    last_bar_server=str(pd.to_datetime(df["time"].iloc[-1], unit="s")))
        (out / f"{sym}_spec.json").write_text(json.dumps(spec, indent=2, default=str))
        print(json.dumps(spec, indent=2, default=str))
        print(f"\n{len(df):,} bougies M1 écrites dans {out / (sym + '_M1.csv')}")
        print("À vérifier : le décalage serveur 'now' doit valoir +3 (heure d'été US) ou +2 (hiver) si votre courtier "
              "suit la convention GMT+2/+3 ; sinon lancez le backtest avec --tz-mode fixed --tz-hours N.")
        print("Swaps : convertissez swap_long/swap_short (selon swap_mode) en bps/jour pour --swap-long-bps-day / --swap-short-bps-day.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
