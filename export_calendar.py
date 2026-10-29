"""Exporte le calendrier économique du terminal MT5 (actual / forecast / previous) en CSV.
Usage : python export_calendar.py --from 2023-01-01 --out mt5_export
Écrit calendar_events.csv (une ligne par publication). Heures = heure serveur du courtier (comme les bougies)."""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", default="2023-01-01")
ap.add_argument("--currencies", default="USD,EUR,GBP,JPY,AUD,NZD,CAD,CHF")
ap.add_argument("--out", default="mt5_export")
a = ap.parse_args()
if not mt5.initialize():
    raise SystemExit(f"initialize() a échoué : {mt5.last_error()}")
Path(a.out).mkdir(exist_ok=True)
start, end = datetime.fromisoformat(a.start), datetime.now() + timedelta(days=1)
rows = []
for cur in a.currencies.split(","):
    ev = mt5.calendar_event_by_currency(cur)
    if ev is None:
        print(cur, "aucun événement", mt5.last_error()); continue
    meta = {e.id: e for e in ev}
    t0 = start
    while t0 < end:                                  # fenêtres de 90 jours (limite de volume de l'API)
        t1 = min(t0 + timedelta(days=90), end)
        vals = mt5.calendar_value_history(t0, t1, "", cur)
        for v in (vals if vals is not None else []):
            e = meta.get(v.event_id)
            if e is None: continue
            f = lambda x: None if x == -9223372036854775808 else x / 1e6
            rows.append(dict(time=int(v.time), currency=cur, event_id=v.event_id, name=e.name, importance=e.importance,
                             frequency=e.frequency, sector=e.sector, actual=f(v.actual_value), forecast=f(v.forecast_value),
                             previous=f(v.prev_value), revised_previous=f(v.revised_prev_value), impact=v.impact_type))
        t0 = t1
    print(cur, len(ev), "événements définis")
df = pd.DataFrame(rows).drop_duplicates(["event_id", "time"]).sort_values("time")
df.to_csv(Path(a.out) / "calendar_events.csv", index=False)
print(f"{len(df)} publications -> {a.out}/calendar_events.csv (importance: {df.importance.value_counts().to_dict()})")
mt5.shutdown()
