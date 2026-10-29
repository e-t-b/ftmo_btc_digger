# ftmo_btc_digger

Recherche et bot MT5 pour le FTMO 2-Step (compte démo). Aucune donnée brute ni identifiant dans le dépôt.

- `export_mt5.py` : export M1 (bid + spread) depuis le terminal MT5.
- `backtest.py`, `strategies.py`, `engine.py` : règles simples + walk-forward, DSR/PSR.
- `ml_fx.py`, `features_v2.py` : modèles LightGBM (triple barrière, coûts bid/ask).
- `session_bt.py` : ouverture de session (OPR), cassure/rebond/fausse cassure, TP 2R, limites jour/semaine.
- `bot_mt5.py` : bot (démo uniquement, ordres seulement avec `--send-orders`).
- `ftmo_sim.py` : Monte Carlo du challenge.

État : aucun avantage validé hors échantillon à ce jour (forex, BTC). Ne pas acheter le challenge avant validation.
