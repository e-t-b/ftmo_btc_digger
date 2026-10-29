# Rapport — Étage 1 bis : prédicteur deep learning sur séquences longues, 2026-10-29

## 1. Ce qui a été testé

Un réseau qui lit directement l'historique des bougies et doit prédire si un achat (ou une vente) avec TP 2:1 touchera sa cible avant son stop : c'est le signal « forte hausse en prévision » destiné au modèle d'exécution. Protocole fixé dans `dl_predictor.py` avant mesure.

- **Entrées** : M5 × 96 (8 h), M15 × 96 (24 h), H1 × 120 (5 j), H4 × 60 (10 j), D1 × 60 (3 mois), plus le contexte.
- **Architecture** : un encodeur GRU + attention par unité de temps, fusion, deux probabilités (achat, vente) et deux rendements futurs auxiliaires (60 et 240 min).
- **Lignes, étiquettes et coûts** : identiques à l'étage 1.
- **Validation** : walk-forward purgé en 3 plis (tests 2021-2022, 2023-2024, 2025-01 → 06), arrêt précoce.
- **Comparaison** : LightGBM réentraîné sur exactement les mêmes plis.

## 2. Résultats (hors échantillon 2021 → 2025-06, 65 290 lignes)

| Modèle | AUC | Spearman déciles | Meilleur dixième R net | Gagnants (meilleur dixième) | Porte |
|---|---|---|---|---|---|
| Deep, achat | 0,541 | 0,88 | −0,174 | 39,3 % | non |
| LightGBM, achat | 0,544 | 0,86 | −0,153 | 40,3 % | non |
| Deep, vente | 0,532 | 0,19 | −0,211 | 37,9 % | non |
| LightGBM, vente | 0,522 | 0,32 | −0,177 | 38,8 % | non |

R net moyen de toutes les lignes : −0,217 (achat), −0,180 (vente).
Seuil de rentabilité d'un TP 2:1 après coûts : environ 40 % de trades gagnants. Les deux modèles y sont tout juste, et seulement sur leur meilleur dixième.

## 3. Diagnostic

1. **Le deep learning ne fait pas mieux que LightGBM**, et même légèrement moins bien sur le meilleur dixième (−0,17 R contre −0,15 R à l'achat ; −0,21 R contre −0,18 R à la vente).
2. **Surapprentissage immédiat** : sur les deux premiers plis, la perte de validation est minimale dès la première époque puis remonte. Le réseau apprend vite le bruit de l'entraînement, faute de motif généralisable dans les séquences de prix.
3. Comme LightGBM, il ordonne les opportunités côté achat (Spearman 0,88), grâce aux mêmes informations de contexte (heure, volatilité, temps restant), mais pas assez pour couvrir le coût.

## 4. Décision

Porte non franchie. Lire de plus longues séquences avec un réseau profond n'apporte pas d'information directionnelle supplémentaire sur EURUSD 2015-2025.

## 5. Test protégé

Non consulté ; données 2026 non lues.
