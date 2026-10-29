# Rapport qualité — EURUSD M1 (FTMO MT5)

Fichier : SHA-256 `2fc008d46678332ce68dba20af54fa283df49c02e17a8ee20158e5b1f721a56e`

- Lignes : 4 380 155 ; UTC 2015-01-04 22:00:00 → 2026-10-28 20:54:00
- Doublons : 0 ; non monotones : 0 ; OHLC incohérents : 0 ; prix ≤ 0 : 0
- Lignes le week-end (serveur) : 0 ; tick volume nul : 0
- Trous > 5 min en semaine : 1718 dont hors rollover : 81
- Fuseau : pic NFP à 15:30-15:31 serveur dans 52% des premiers vendredis (attendu : majorité)
- |rendement 1 min| : médiane 0.51 bp, p99,99 28.1 bp, max 164.6 bp

## Par année

| Année | Lignes | Jours | Spread médian | Spread p95 | Part spread 0 | Tick vol. médian |
|---|---|---|---|---|---|---|
| 2015 | 369986 | 258 | 8.0 | 16.0 | 0.0% | 65.0 |
| 2016 | 373151 | 260 | 7.0 | 11.0 | 0.1% | 80.0 |
| 2017 | 370355 | 259 | 7.0 | 10.0 | 13.7% | 58.0 |
| 2018 | 371934 | 259 | 3.0 | 9.0 | 0.0% | 51.0 |
| 2019 | 371516 | 259 | 8.0 | 10.0 | 0.0% | 26.0 |
| 2020 | 371763 | 261 | 50.0 | 50.0 | 0.0% | 25.0 |
| 2021 | 371672 | 260 | 50.0 | 50.0 | 0.0% | 19.0 |
| 2022 | 367830 | 259 | 2.0 | 3.0 | 0.0% | 41.0 |
| 2023 | 367606 | 259 | 2.0 | 4.0 | 0.0% | 39.0 |
| 2024 | 370173 | 260 | 2.0 | 4.0 | 0.0% | 43.0 |
| 2025 | 369727 | 259 | 2.0 | 5.0 | 24.4% | 34.0 |
| 2026 | 304442 | 214 | 0.0 | 5.0 | 61.2% | 43.0 |

## Trous hors rollover (10 plus longs)

- 2018-12-24 20:59:00 → 2018-12-26 06:00:00 (1981 min)
- 2018-12-31 22:59:00 → 2019-01-02 06:00:00 (1861 min)
- 2019-12-24 20:59:00 → 2019-12-26 00:00:00 (1621 min)
- 2019-12-31 22:59:00 → 2020-01-02 00:05:00 (1506 min)
- 2016-03-23 17:29:00 → 2016-03-23 20:33:00 (184 min)
- 2016-10-13 15:41:00 → 2016-10-13 17:52:00 (131 min)
- 2022-10-05 22:06:00 → 2022-10-06 00:11:00 (125 min)
- 2022-09-20 22:06:00 → 2022-09-21 00:08:00 (122 min)
- 2022-09-15 22:06:00 → 2022-09-16 00:08:00 (122 min)
- 2022-09-07 22:06:00 → 2022-09-08 00:08:00 (122 min)

## Conséquences retenues

- Prix BID seuls : l'ask est reconstruit avec le modèle de spread conservateur (`costs.py`).
- Spread historique non homogène (2020-2021 à 50 pts constants, zéros fréquents en 2025-2026) : non utilisé comme feature ; coût = max(profil p75 2022-2024, observé depuis 2022, 1 pt).
- Tick volume non homogène entre années : uniquement en ratio à sa propre moyenne glissante passée.
- Aucune entrée pendant l'heure du rollover ; clôture forcée à 16:45 NY.
- Jours partiels exclus : 2015-01-02 et 2026-10-29 (jour d'export).
