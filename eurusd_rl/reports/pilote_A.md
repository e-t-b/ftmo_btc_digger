# Rapport — Étape A (baseline PPO), 2026-10-29

## 1. Ce qui a été testé

| Run | Contenu | Arrêt |
|---|---|---|
| `A_baseline_s1` | PPO compact (25 319 paramètres) : GRU + attention sur M5 (48 barres) et H1 (24 barres), contexte (11), état du compte (10). Récompense = variation d'équité en R − 10 R par violation. Seed 1. | Arrêté à la mise à jour 125/300 (aucun apprentissage) |
| `A2_s1` | A1 + perte de politique sur les seuls pas à choix réel + biais initial WAIT = +4 (exploration étalée sur la journée). Seed 1. | Arrêté à la mise à jour 53/300 (effondrement vers WAIT) |
| `sanity_planted_d4` | Config A2 sur prix réels 2023-06 → 2024-09 auxquels on ajoute une dérive quotidienne de 4 pips/h dont le sens est donné à l'agent. Diagnostic de la chaîne d'apprentissage. | 100 mises à jour |

## 2. Données et périodes

EURUSD M1 FTMO-Demo (SHA-256 `2fc008d4…`), 2015-01-05 → 2026-10-28. Entraînement 2015-01-05 → 2025-06-30 ;
calibration 2025-07-01 → 2025-12-31 (sélection des checkpoints) ; validation de recherche 2026-01-01 → 2026-10-28 en trois blocs.
Coûts : spread conservateur, commission 3,50 $/lot/côté, glissement 0,2 pip (0,5 pip en plus au stop). Stress : spread × 1,5 et glissements × 2.
Règles : 0,5 %/trade, TP 4R, stop = 1 ATR M15 (≥ 3 pips et ≥ 3 × coût aller-retour), 3 trades/jour, arrêts internes −1,5 %/−4 %/−8 %.

## 3. Résultats chiffrés

Calibration pendant l'entraînement d'A1 (évaluation déterministe, comptes mensuels) :

| Mise à jour | Trades | R total | R/trade | Gagnants | DD max |
|---|---|---|---|---|---|
| 25 | 129 | −37,4 | −0,29 | 19 % | −8,8 % |
| 50 | 0 | — | — | — | 0 |
| 75 | 288 | −71,0 | −0,25 | 23 % | −12,5 % |
| 100 | 320 | −88,1 | −0,28 | 32 % | −8,3 % |
| 125 | 0 | — | — | — | 0 |

Évaluation complète (compte continu) du meilleur checkpoint d'A1 (mise à jour 25) et des références :

| Politique | Période | Trades | R/trade | PF | Rendement | DD max | Violations |
|---|---|---|---|---|---|---|---|
| A1 @25 | calibration | 69 | −0,24 | 0,74 | −8,2 % | −18,8 % | 0 |
| A1 @25 | validation | 85 | −0,19 | 0,79 | −8,0 % | −11,9 % | 0 |
| A1 @25 | validation, coûts stressés | 36 | −0,45 | — | −8,2 % | — | 0 |
| Aléatoire (actions valides) | validation | 82 | −0,21 | 0,48 | −8,4 % | −9,0 % | 0 |
| WAIT | toutes | 0 | — | — | 0 | 0 | 0 |

- A1 @25 est dégénéré : **100 % SELL**. Chaque bloc de validation se termine sur l'arrêt global interne (−8 %), qui borne mécaniquement la perte.
- Sur la calibration, A1 @25 atteint +10 % au jour 12, puis perd 18,8 % depuis ce pic : une réussite de phase 1 par pure chance d'une politique « toujours vendre ».
- A1 @125 (dernier checkpoint) et A2 : 0 trade en mode déterministe (WAIT permanent ; A2 : entropie 0,14 → 0,002 en 50 mises à jour).

Diagnostic sur signal injecté (calibration 2024-07 → 2024-09) :

| Mise à jour | Trades | R total | R/trade | Gagnants | DD max |
|---|---|---|---|---|---|
| 40 | 36 | +40,4 | +1,12 | 61 % | −2,7 % |
| 60 | 97 | +131,7 | +1,36 | 53 % | −3,2 % |
| 80 | 190 | +161,7 | +0,85 | 43 % | −5,7 % |
| 100 | 102 | +89,8 | +0,88 | 42 % | −8,0 % |

Règle oracle (entrer dans le sens de l'indice à la première décision du jour) : +1,00 R/trade avec 4 pips/h ;
+0,25 R/trade avec 2 pips/h ; **−0,32 R/trade avec 1 pip/h** (14 pips de dérive journalière connue à l'avance ne suffisent pas avec ces stops et ces coûts).

## 4. Comparaison au champion

Aucun champion n'existe encore. Aucun candidat ne bat WAIT (0 R) ni ne s'approche des critères de promotion :
**A1 et A2 sont rejetés.**

## 5. Problèmes détectés

1. Sur données réelles, PPO ne trouve aucune entrée dont l'espérance couvre les coûts : récompense d'entraînement plate (~−20 R par lot de 4 096 pas), puis repli vers WAIT ou politique dégénérée à un seul sens.
2. La chaîne d'apprentissage est **capable** d'apprendre : sur le signal injecté, elle atteint et dépasse la règle oracle en 40-60 mises à jour.
   L'échec sur données réelles s'explique donc d'abord par l'absence de signal exploitable dans ces features, pas par un défaut de la chaîne.
3. Correction de mon diagnostic intermédiaire :
   - J'avais avancé que 97 % des pas n'avaient qu'une action valide. C'était la part des actions WAIT, pas des pas sans choix ; mesurée dans A2, la part des pas à choix réel est de 72 à 90 %.
   - J'avais aussi lu un KL d'environ 10⁻⁵ comme un signe de non-apprentissage. Le KL reste du même ordre (10⁻⁴) dans le run qui apprend : ce n'est pas un indicateur d'échec dans cette configuration.
4. Instabilité : sur le signal injecté, le dernier checkpoint (100) est nettement moins bon que le meilleur (60). Il faudra conserver la sélection par calibration et ajouter de la régularisation.
5. La barre est haute : le coût aller-retour (~1,3 pip) représente environ 0,15-0,3 R par trade avec des stops de 4 à 8 pips.
6. L'arrêt global interne à −8 % borne chaque évaluation. Une politique perdante finit toujours à −8 % : à comparer en R/trade et en PF, pas seulement en rendement.

## 6. Décision

**Étape A sur données réelles : abandonnée en l'état** (pas de nouvelles seeds pour cette architecture : l'échec est structurel, pas de variance).
Chaîne technique (données, environnement, tests, PPO, checkpoints, évaluation) : **validée et conservée**.

## 7. Expérience suivante proposée

Étape B, en commençant sans apprentissage :

1. Construire la bibliothèque de stratégies explicites (section 7 du cahier des charges) avec leurs tests de causalité.
2. Mesurer chaque stratégie seule sur l'entraînement : fréquence, espérance nette en R, stabilité par année, sensibilité aux coûts.
3. Ne passer au renforcement que si au moins une stratégie a une espérance nette positive et stable sur l'entraînement **et** la calibration.
   Le meilleur candidat devient le champion à battre.
4. Usage du renforcement envisagé ensuite : arbitrer WAIT contre le sens proposé par les stratégies (filtre), plutôt que découvrir les entrées sur features brutes.

## 8. Test protégé

Non consulté : les données forward (≥ 2026-11-02) n'existent pas encore et ne sont chargées par aucun script.
