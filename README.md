# ftmo_btc_digger

Recherche et bot MT5 pour le FTMO 2-Step (compte démo). Aucune donnée brute ni identifiant dans le dépôt.

- `export_mt5.py` : export M1 (bid + spread) depuis le terminal MT5.
- `backtest.py`, `strategies.py`, `engine.py` : règles simples + walk-forward, DSR/PSR.
- `ml_fx.py`, `features_v2.py` : modèles LightGBM (triple barrière, coûts bid/ask).
- `session_bt.py` : ouverture de session (OPR), cassure/rebond/fausse cassure, TP 2R, limites jour/semaine.
- `bot_mt5.py` : bot (démo uniquement, ordres seulement avec `--send-orders`).
- `ftmo_sim.py` : Monte Carlo du challenge.

## Refonte v2 (branche `refonte-v2`)

- `docs/protocole_jalon_A.md` : protocole pré-enregistré (données, découpage, stratégie, coûts, risque, critères).
- `v2/` : chargement des M1 FTMO, coûts, breakout de session, portefeuille + module de risque, simulateur FTMO, jalon A.
- `tests_v2/` : tests de la mécanique (`python -m pytest tests_v2`).
- `reports_v2/jalon_a_dev.md` : résultat du jalon A — **NO-GO** (espérance hors échantillon −0,064 R).
- `docs/protocole_jalon_A_intraday.md`, `v2/intraday.py`, `reports_v2/jalon_a_intraday_dev.md` : hypothèses intraday H1-H3 (momentum de fin de séance, BTCUSD, comblement de gap) — **NO-GO**, aucun semestre tradé.

Données attendues dans `data/m1/<SYMBOLE>.npz` + `<SYMBOLE>_spec.json` (non versionnées).

État : aucun avantage validé hors échantillon à ce jour (forex, BTC, indices : breakout de session, momentum de fin de séance, comblement de gap). Ne pas acheter le challenge avant validation.
