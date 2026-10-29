# Rapport — Étage 1 : prédicteur supervisé (LightGBM), 2026-10-29

## 1. Ce qui a été testé

Une couche prédictive entraînée à anticiper le résultat du trade que l'environnement prendrait. Protocole fixé dans `predictor.py` avant mesure.

- **Lignes** : décisions où l'entrée est permise, aux quarts d'heure, jusqu'au 2025-12-31 (159 415 lignes).
- **Features** : 174 features causales (`features_pred.py`) :
  - contexte ;
  - 12 bougies M5 et 6 bougies H1 ;
  - rendements de 15 min à 5 jours ;
  - RSI et position dans le range ;
  - état des 12 stratégies de la bibliothèque aux échelles M5, H1 et H4 ;
  - GBPUSD, US500 et GER40.
- **Cibles** : trade en achat et en vente, cible 2R ou 4R, stop d'1 ATR M15, coûts complets. Étiquettes identiques à l'environnement (test sur 120 trades réels).
- **Modèles** : 4 modèles LightGBM, hyperparamètres fixes.
- **Validation** : walk-forward annuel purgé, prédictions hors échantillon de 2016 à 2025-06.

## 2. Porte (pré-enregistrée)

- Corrélation de Spearman ≥ 0,8 entre le décile de score et le R réalisé.
- Meilleur dixième :
  - R net ≥ +0,05 ;
  - t ≥ 3 sur les moyennes journalières ;
  - ≥ 300 trades ;
  - ≥ 65 % d'années positives.
- Confirmation sur la calibration (2025-07 → 2025-12).

## 3. Résultats chiffrés (hors échantillon 2016 → 2025-06)

| Modèle | AUC | Spearman déciles | R net moyen (toutes) | Meilleur dixième R net | Années + | Calibration (meilleur dixième) |
|---|---|---|---|---|---|---|
| Achat 2R | 0,540 | 0,95 | −0,214 | −0,151 | 0 % | −0,226 (n = 435) |
| Vente 2R | 0,521 | 0,54 | −0,178 | −0,176 | 0 % | −0,073 (n = 697) |
| Achat 4R | 0,613 | 0,92 | −0,219 | −0,143 | 0 % | −0,269 (n = 431) |
| Vente 4R | 0,594 | 0,22 | −0,181 | −0,177 | 0 % | −0,122 (n = 798) |

**Aucun modèle ne passe la porte.**

**Diagnostic sans coûts**, avec les mêmes trades et les mêmes prédictions :

| Modèle | Brut, toutes les lignes | Brut, meilleur dixième |
|---|---|---|
| Achat 2R | −0,019 | +0,003 |
| Vente 2R | +0,021 | +0,021 |
| Achat 4R | −0,019 | +0,012 |
| Vente 4R | +0,027 | +0,026 |

## 4. Comparaison au champion

Aucun champion.

## 5. Diagnostic

1. **Le modèle apprend, mais pas la direction.** Les variables dominantes sont :
   - l'heure et le temps restant avant la clôture forcée ;
   - la volatilité (rang de l'ATR) ;
   - la position dans le range du jour.

   Il apprend *quand* un trade peut structurellement atteindre sa cible (assez de temps, assez de mouvement), et quand le coût pèse moins en R (volatilité forte, donc stop plus large). D'où des déciles ordonnés (Spearman jusqu'à 0,95) et une AUC de 0,54 à 0,61.
2. **L'avantage brut du meilleur dixième est quasi nul** (0 à +0,03 R par trade), contre un coût d'environ 0,15 à 0,2 R. Le classement réduit la perte sans la renverser.
3. Le léger brut positif des ventes (+0,02 R) reflète la baisse d'EURUSD sur la période, pas une prédiction.
4. Avertissement statistique : sur les rendements bruts du meilleur dixième, le t calculé sur les moyennes journalières est élevé (+4 à +5) alors que la moyenne par trade est quasi nulle. Cela traduit une forte hétérogénéité entre jours (jours à nombreux signaux contre jours à signal isolé), non élucidée. Je ne m'appuie que sur les moyennes par trade, et la porte nette n'est franchie sous aucune mesure.

## 6. Décision

Étage 1 : **porte non franchie**. Combiner les 174 signaux disponibles ne produit pas d'information directionnelle exploitable sur EURUSD 2016-2025. L'étage 2 (modèle d'exécution) n'a rien à exécuter.

## 7. Suite

La seule source d'information encore non testée est le calendrier économique (fichier toujours en attente).

## 8. Test protégé

Non consulté ; données 2026 non lues.
