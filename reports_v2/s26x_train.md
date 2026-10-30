# S26 étendu — exploration sur l'entraînement (Dukascopy 2013-2021-08, FTMO 2021-09 → 2024-06)

Période finale FTMO (2024-07 →) non utilisée. Reproduire : `python -m v2.s26x`. Résultats : `reports_v2/s26x_train.json`.
Toutes les variantes : stop à mi-chemin entre l'entrée et l'ouverture, mouvement ≤ 1 ATR, ATR ≥ 0,8 % du prix.

| Variante | Source | Trades/an | R | t | Gain/an | DD max |
|---|---|---|---|---|---|---|
| A — 3 indices, décision à +2 h (S26b) | Dukascopy | 63 | +0,096 | 1,81 | +3,0 % | −7,3 % |
| | FTMO | 97 | +0,101 | 1,39 | +4,8 % | −7,5 % |
| **B — 3 indices, décision à +1, +2 ou +3 h** | Dukascopy | 126 | **+0,098** | **2,42** | +6,1 % | −10,8 % |
| | FTMO | 179 | **+0,080** | 1,42 | +7,1 % | −11,1 % |
| C — 6 indices, +2 h | Dukascopy | 115 | +0,048 | 1,27 | +2,8 % | −13,7 % |
| | FTMO | 175 | −0,004 | −0,09 | +0,2 % | −13,6 % |
| D — 6 indices, +1/+2/+3 h | Dukascopy | 222 | +0,038 | 1,31 | +4,3 % | −18,8 % |
| | FTMO | 326 | +0,004 | 0,09 | +0,9 % | −21,2 % |

Indices ajoutés (variante D) : JP225 −0,14 R / −0,10 R, UK100 −0,03 R / −0,19 R, US30 +0,04 R / −0,01 R
(Dukascopy / FTMO). **L'effet « journée de tendance » n'existe que sur US500, US100 et GER40.**
Les plafonds par bloc d'indices US (variantes E, F) n'améliorent rien.

## Risque par trade (variante B, Monte Carlo FTMO 2-Step, 5 000 trajectoires)

| Risque | Réussite Dukascopy | Réussite FTMO | Durée médiane |
|---|---|---|---|
| 0,50 % | 28 % | 32 % | 7 à 8 mois |
| **0,75 %** | **40 %** | **41 %** | **5 mois** |
| 1,00 % | 32 % | 31 % | 5 mois |

Au-delà de 0,75 %, les échecs par perte journalière ou totale l'emportent.

Correctif : les Monte Carlo Dukascopy des rapports `dukascopy.md` et `s26b_train.json` complétaient la série
de journées par des zéros jusqu'en 2024, ce qui sous-estimait la réussite. Les chiffres ci-dessus sont corrects.
