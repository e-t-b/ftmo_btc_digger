# M1 — sélection de stratégie selon le contexte (méta-modèle) : rapport

Date : 2026-10-30. Protocole : `docs/protocole_meta.md` (commit `8349c73`, avant tout calcul).
Résultats : `reports_v2/meta_dev.json`.
Reproduire : `python -m v2.meta` puis `python -m v2.meta_eval base`, `stress`, `final`.

## Décision : NO-GO

| | Trades | R net moyen | t |
|---|---|---|---|
| Réservoir non filtré (26 stratégies, semestres 2018-S1 → 2024-S1) | 58 314 | −0,079 | — |
| Trades choisis par le modèle | 15 377 | −0,039 | −3,6 |
| Après module de risque (portefeuille) | 6 587 | −0,066 | −4,0 |
| Choisis par le modèle, coûts stressés | 15 153 | −0,076 | −7,1 |

Monte Carlo FTMO : 4 % de réussite. p = 1,0 contre un seuil de Holm de 0,0038 (famille 33).

## Lecture

- Le modèle apprend **un peu** : il réduit la perte moyenne de moitié (−0,079 → −0,039 R) en écartant les pires
  contextes, et 3 semestres sur 13 deviennent positifs (2020-S1, 2022-S1, 2022-S2).
- Mais il ne trouve **aucun contexte durablement gagnant** : choisir le moins mauvais parmi des stratégies perdantes
  donne une stratégie moins perdante, pas une stratégie gagnante.
- Le portefeuille est encore plus négatif que la moyenne des trades : les plafonds de risque gardent surtout les trades
  du forex, plus nombreux et plus coûteux.

## Écarts au protocole

- Évaluation découpée en processus séparés (limite mémoire de 7 Go) ; résultat identique, code dans `v2/meta_eval.py`.
- Stress mesuré au niveau des trades (R net moyen), sans repasser par le portefeuille. Sans effet sur la décision.
