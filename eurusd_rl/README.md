# eurusd_rl — agent intraday EURUSD (PPO), recherche locale uniquement

Aucun broker n'est connecté et aucun ordre n'est envoyé par ce paquet.

## Données

- Export MT5 FTMO-Demo, EURUSD M1 BID + spread de barre, du 2015-01-02 au 2026-10-29.
- Fichier compressé sans perte : `../data_eurusd/EURUSD_M1_2015_2026.npz.xz` (hors dépôt), SHA-256 vérifié au chargement.
- Rapport qualité : [`reports/data_quality.md`](reports/data_quality.md) (`python -m eurusd_rl.qc`).
- Heure serveur = heure de New York + 7 h (vérifié : pic NFP à 15:30 serveur sur toute la période).

## Périodes (UTC)

| Rôle | Période |
|---|---|
| Entraînement | 2015-01-05 → 2025-06-30 |
| Calibration (sélection des checkpoints) | 2025-07-01 → 2025-12-31 |
| Validation de recherche (3 blocs) | 2026-01 → 04, 2026-05 → 07, 2026-08 → 10-28 |
| Test protégé | données forward ≥ 2026-11-02, consultées une fois après gel et autorisation |

L'historique 2026-05 → 2026-10 a déjà été vu par d'anciennes expériences EURUSD (`ml_fx.py`, `session_bt.py`) :
il sert de validation, pas de test.

## Règles et coûts

Tout est dans [`config/default.yaml`](config/default.yaml) : FTMO 2-Step Swing 10 000 $ (objectif 10 %/5 %,
perte journalière 5 % base solde 00:00 Prague équité flottante incluse, perte totale 10 % statique, levier 1:30),
limites internes (0,5 %/trade, arrêt jour −1,5 %, semaine −4 %, global −8 %, 3 trades/jour),
fenêtre d'entrée 02:00-16:00 NY, clôture forcée 16:45 NY, commission 3,50 $/lot/côté (à confirmer),
spread conservateur (profil p75 2022-2024 par quart d'heure, observé depuis 2022, plancher 1 pt).

## Utilisation

```bash
python -m pytest eurusd_rl/tests -q                       # obligatoire avant tout entraînement
python -m eurusd_rl.train --run runs/A_baseline_s1 --seed 1 --updates 300
python -m eurusd_rl.train --run runs/A_baseline_s1 --updates 600 --resume
python -m eurusd_rl.report --run runs/A_baseline_s1 --references
```

Chaque run contient `config.json` (config complète, seed, hash des données, version de la bibliothèque de stratégies),
`progress.jsonl`, `validation_history.jsonl` (calibration), `latest.pt`, `best_validation.pt`
(meilleur score robuste sur la calibration), `evaluation.json`. Les checkpoints se chargent avec `weights_only=True`.

## Critères de promotion

Voir la section `promotion` de la config. Un PnL supérieur ne suffit jamais ; le premier champion à battre est la
meilleure stratégie explicite de la bibliothèque (étape B).
