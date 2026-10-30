# Catalogue de stratégies intraday S1-S15 — rapport (période de développement)

Date : 2026-10-30. Protocole : `docs/protocole_catalogue.md` (commit `ef12a61`, avant tout test ; correctif S15
`fa563d4`). Résultats bruts : `reports_v2/catalogue_dev.json`. Reproduire : `python -m v2.catalogue_run`.

## Décision : NO-GO, aucune survivante

Aucune des 19 stratégies de la famille n'est significative après correction de Holm (α = 5 %).
La meilleure p-valeur est 0,22 (S2), pour un seuil corrigé de 0,0026. Conformément au protocole, la période finale
(2024-07 → 2026-10) n'a pas été lancée.

## Résultats hors échantillon (walk-forward, portefeuille autonome par stratégie, risque 0,5 %)

| Stratégie | Trades | Espérance (R) | PF | t | p | Seuil Holm | DD max | Semestres tradés |
|---|---|---|---|---|---|---|---|---|
| S2 Volatility breakout | 172 | +0,100 | 1,21 | 0,78 | 0,217 | 0,0026 | −7,0 % | 3 / 14 |
| S10 Dérive avant la Fed | 46 | +0,042 | 1,22 | 0,52 | 0,300 | 0,0028 | −2,0 % | sans sélection |
| S11 Tournant de mois | 394 | +0,017 | 1,04 | 0,31 | 0,380 | 0,0029 | −9,7 % | sans sélection |
| S3 Cassure plus haut / bas de la veille | 637 | −0,014 | 0,97 | −0,24 | 0,595 | 0,0031 | −16,1 % | 9 / 66 |
| S8 IBS horaire | 79 | −0,042 | 0,92 | −0,30 | 0,619 | 0,0033 | −7,3 % | 2 / 6 |
| S13 Fixing de Londres | 322 | −0,018 | 0,92 | −0,59 | 0,723 | 0,0036 | −5,9 % | 5 / 52 |
| S4 Donchian H1 | 437 | −0,037 | 0,92 | −0,69 | 0,756 | 0,0038 | −16,0 % | 7 / 66 |
| S7 Turtle Soup | 286 | −0,103 | 0,84 | −1,13 | 0,871 | 0,0042 | −25,1 % | 7 / 66 |
| S14 Dérive après NFP / CPI | 8 | −0,787 | 0,15 | −2,92 | 0,998 | 0,0050 | −3,1 % | 1 / 58 |
| S12 Saisonnalité horaire | 6 043 | −0,093 | 0,77 | −8,24 | 1,000 | 0,0056 | −278 % | 59 / 66 |
| S1 Range asiatique → Londres | 0 | — | — | — | 1 | — | — | 0 / 52 |
| S5 Retour au VWAP | 0 | — | — | — | 1 | — | — | 0 / 6 |
| S6 Bollinger en range | 0 | — | — | — | 1 | — | — | 0 / 58 |
| S15 Paires | 0 | — | — | — | 1 | — | — | 0 / 15 |

Déjà testées et comptées dans la famille : breakout de session v2 (p = 0,91), H1, H2, H3 (aucun trade hors
échantillon, p = 1). S9 : réservée à la période finale, non testée (p = 1 en développement).

## Lecture

- **Les trois seules espérances positives (S2, S10, S11) sont très loin d'être significatives** (t entre 0,3 et 0,8).
  Avec 19 stratégies testées, en trouver deux ou trois légèrement positives est exactement ce que le hasard produit.
- **S1, S5, S6, S15 n'ont jamais été tradées** : aucune configuration n'atteignait t ≥ 1 en entraînement.
- **S12 est fortement négative** : choisir l'heure la plus marquée parmi 9 à 24 créneaux trouve presque toujours un
  |t| ≥ 2 par hasard, puis le trade quotidien paie les coûts sans avantage. C'est une illustration directe du
  problème des tests multiples.
- S14 n'a tradé qu'un semestre : le seuil de 30 trades d'entraînement et t ≥ 1 n'est presque jamais atteint.

## Vérifications

- Réimplémentation indépendante de S11 (US100, 136 trades) et de S2 (US100, 418 trades) : écart maximal 0 R.
- Calendriers vérifiés sur les prix : les jours NFP montrent un mouvement médian de 17 bps à 08:30 New York sur
  USDJPY, contre 1,6 bp les autres jours.
- Moteur commun aux autres études (mêmes coûts, module de risque, simulateur FTMO) déjà validé.

## Écarts au protocole (transparence)

- Un test de fumée, avant le run, a calculé par erreur des moyennes sur toute la période, période finale comprise :
  une configuration par stratégie sur EURUSD, US500 et la paire US100/US500, S9 comprise (S9 : −0,022 R sur US500).
  Aucune règle de stratégie n'a été changée en conséquence. Ce coup d'œil affaiblit légèrement la valeur de la période
  finale comme test totalement aveugle pour ces combinaisons ; il ne peut pas avoir favorisé une stratégie,
  puisque la sélection est mécanique et qu'aucune n'a survécu.
- Correctif S15 (extrêmes de minute de la série synthétique) décidé après ce test de fumée et documenté au §3 bis.

## Bilan d'ensemble à ce stade

Dans cette série, 18 stratégies intraday ont été testées sur les données FTMO (14 du catalogue, le breakout de
session, H1, H2, H3), en plus des essais antérieurs du dépôt (OPR, LightGBM, deep learning). Aucune ne montre
d'avantage hors échantillon après coûts. L'infrastructure est fiable ; ce sont les sources d'avantage qui manquent.
