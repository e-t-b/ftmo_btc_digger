# Jalon A — hypothèses intraday H1-H3 (période de développement)

Date : 2026-10-30. Protocole : `docs/protocole_jalon_A_intraday.md`, commité avant tout test (commit `ac0ba26`).
Résultats bruts : `reports_v2/jalon_a_intraday_dev.json`. Reproduire : `python -m v2.jalon_intraday`.
Instruments disponibles au moment du test : US100, US500, GER40 (H1, H3) et BTCUSD (H2). L'export complet
du US30 n'était pas encore disponible ; il pourra être ajouté comme test complémentaire.

## Décision : NO-GO pour H1, H2 et H3

Le walk-forward n'a tradé **aucun semestre**, sur aucun instrument : la meilleure configuration
d'entraînement n'a jamais atteint une statistique t de 1. Aucun trade hors échantillon, donc aucun critère rempli.
La période finale (2024-07 → 2026-10) n'a pas été lancée et reste vierge.

## Résultats en échantillon (période de développement, moyenne des 8 configurations, après coûts)

| Hypothèse | Instrument | Trades / config | R net moyen | Meilleure config | Configs positives |
|---|---|---|---|---|---|
| H1 momentum fin de séance | US100 | 580 | −0,035 | +0,006 | 2 / 8 |
| H1 | US500 | 570 | −0,079 | −0,032 | 0 / 8 |
| H1 | GER40 | 525 | −0,073 | −0,042 | 0 / 8 |
| H2 momentum BTCUSD | BTCUSD | 1 253 | −0,093 | −0,063 | 0 / 8 |
| H3 comblement de gap | US100 | 347 | −0,064 | −0,057 | 0 / 8 |
| H3 | US500 | 357 | −0,091 | −0,070 | 0 / 8 |
| H3 | GER40 | 313 | −0,040 | −0,012 | 0 / 8 |

Même en choisissant après coup la meilleure des 8 configurations, seule une variante de H1 sur US100 est
au-dessus de zéro, à +0,006 R, ce qui n'est pas distinguable de zéro.

## Pourquoi : l'effet n'existe pas dans ces données, même sans coûts

Vérification directe sur les prix, sans le moteur ni les coûts (2021-10 → 2024-06 ; BTCUSD 2017-06 → 2024-06) :

| Instrument | Corrélation début / fin de séance | Gain moyen de la règle H1 sans coûts | Corrélation gap / première heure |
|---|---|---|---|
| US500 | −0,08 | −0,9 bp (t = −0,8) | +0,04 |
| US100 | −0,06 | −1,8 bp (t = −1,5) | −0,03 |
| GER40 | −0,12 | −1,3 bp (t = −1,9) | −0,04 |
| BTCUSD | −0,05 | +1,2 bp (t = +0,8) | −0,12 |

- **H1** : sur 2021-2024, la fin de séance a plutôt tendance à **corriger** le mouvement du début de journée.
  L'effet publié (Gao et al., 2018 ; Baltussen et al., 2021) porte sur des périodes antérieures et n'apparaît pas ici.
- **H2** : le gain brut est faible et non significatif ; les coûts du BTCUSD (≈ 0,13 R par trade) le rendent négatif.
- **H3** : le gap n'annonce pas de retour vers la clôture de la veille dans la première heure.

## Vérifications

- 10 tests unitaires (`tests_v2/`) dont 3 sur les setups H1 et H3 (sens, seuil, objectif au comblement).
- Le calcul direct sur les prix (tableau ci-dessus) et le moteur concordent : négatif sur les trois indices.
- Calage horaire contrôlé : pic de volume à 09:30 New York (US500, US100) et 09:00 Francfort (GER40).

## Piste observée après coup (non validée)

La corrélation négative de H1 (−0,06 à −0,12) suggère un léger **retour à la moyenne** en fin de séance.
C'est une observation faite *après* avoir vu les données : elle ne peut pas être adoptée telle quelle.
Elle ne vaut que −1 à −2 bp par trade avant coûts, alors que le spread et le glissement d'un indice coûtent
environ 1 à 1,5 bp aller-retour. Si elle est testée, ce doit être dans un nouveau protocole pré-enregistré,
et sa validation ne pourra se faire que sur la période finale.
