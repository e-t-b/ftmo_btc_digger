# Rapport — Étape 1 : carte de prévisibilité EURUSD (première partie), 2026-10-29

## 1. Ce qui a été testé

44 études d'événements pré-enregistrées (`predictability.py`, docstring daté avant mesure). Aucune stratégie, aucun paramètre ajusté.

| Famille | Contenu |
|---|---|
| E1 | Rendement moyen de chaque heure (heure de New York) et des 4 sessions |
| E2 | Fixing de Londres 16:00 (tous les jours ; fins de mois signées à l'inverse du mouvement du mois) |
| E3 | Jour de la semaine |
| E4 | Continuation ou retournement après un grand mouvement H1 ou D1 |
| E5 | Comblement du gap du lundi |
| E6 | Momentum temporel 20 j → 5 j et 5 j → 1 j |

Critère « exploitable » (les trois à la fois) : |moyenne| ≥ 2 pips par occurrence ; |t| ≥ 3,25 (Bonferroni à 5 % sur 44 tests) ; même signe dans ≥ 70 % des années.

## 2. Données et périodes

Prix BID M1, entraînement 2015-01-05 → 2025-06-30. Les données ne sont chargées que jusqu'au 2025-12-31 : la validation 2026 n'est jamais lue.

## 3. Résultats chiffrés

**Aucun effet exploitable.** Effets statistiquement significatifs (|t| ≥ 3,25) :

| Effet | Moyenne | t | Années même signe | Interprétation |
|---|---|---|---|---|
| Heure 18 h NY | +1,26 p | +9,9 | 100 % | **Artefact majoritaire** : sur 2022-2025, +1,63 p sur le BID contre +0,66 p sur le prix moyen bid/ask. Le spread s'élargit au rollover puis se resserre. Le résidu tombe pendant le spread maximal (~2,5 p) : non exploitable. |
| Heure 07 h NY | −1,31 p | −5,8 | 91 % | **Effet réel** : identique sur le BID et le prix moyen (−1,64 p, t −4,5 sur 2022-2025). L'euro baisse en fin de matinée européenne, cohérent avec l'effet documenté de dépréciation d'une devise pendant les heures de son propre marché. Taille ≈ coût aller-retour (~1,3 p). |
| Heure 16 h NY | −0,47 p | −4,2 | 55 % | Rollover : signe instable ; 17 h NY est un artefact pur (−0,73 p BID, +0,05 p prix moyen). |
| Heure 06 h NY | −0,71 p | −3,3 | 91 % | Même famille que 07 h NY ; trop petit seul. |
| Session « Asie » 17-03 NY | +1,62 p | +3,3 | 73 % | Contient l'artefact de 18 h NY. |

Autres résultats notables (non significatifs après correction) :
- lundi +5,6 p (t 2,8 ; l'ouverture du lundi se fait au spread le plus large, artefact probable) ;
- comblement du gap du lundi +3,7 p (t 2,2 ; 55 % des années seulement) ;
- fixing de Londres et fins de mois : rien (t < 1,5) ;
- grands mouvements : ni continuation ni retournement (|t| ≤ 1,3) ;
- momentum temporel : rien (|t| < 1).

Détail complet par année : `predictability_eurusd.json`.

## 4. Comparaison au champion

Sans objet (aucune stratégie).

## 5. Problèmes détectés

1. Les mesures sur le BID créent des artefacts autour du rollover (17-19 h NY). Toute étude future doit utiliser un prix moyen bid/ask (fiable seulement depuis 2022 dans nos données) ou exclure ces heures.
2. Le seul effet réel identifié (« baisse de l'euro en fin de matinée européenne », 06-08 h NY, environ −2 pips sur deux heures) a la taille de nos coûts. Regrouper 06 h et 07 h après avoir vu les résultats serait un choix *a posteriori* : il faudrait le confirmer sur un échantillon indépendant.

**Vérification de signe sur la calibration (2025-07 → 2025-12, 131 jours)** : 06 h NY −0,92 p (t −1,3),
07 h NY −1,16 p (t −1,4). Même signe que l'entraînement ; non significatif seul (puissance faible).

## 6. Décision

Première partie de la carte : **aucun effet exploitable avec la structure de coûts actuelle**.

Une hypothèse est retenue pour confirmation indépendante (pas pour trader) : vente EURUSD 06:00 → 08:00 NY. Son signe est confirmé sur la calibration ; elle se testera ensuite sur GBPUSD (même mécanisme attendu pendant les heures de Londres) et, plus tard, en forward. Avec un coût aller-retour de 1,3 pip, même confirmé, l'avantage net serait d'environ 0,5 pip par jour : insuffisant pour un challenge.

## 7. Suite

Deuxième partie de la carte, dès réception des données :
- annonces macro : dérive après publication et écart entre réel et consensus ;
- avance-retard avec GBPUSD, US500 et GER40.

C'est là que se trouve la seule information réellement extérieure au graphique EURUSD.

## 8. Test protégé

Non consulté. Validation 2026 : non lue.
