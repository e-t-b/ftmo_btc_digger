# Rapport — Étape B, partie 1 : stratégies explicites seules (sans apprentissage), 2026-10-29

## 1. Ce qui a été testé

Bibliothèque `B1-2026-10-29` (`strategies.py`) : 11 stratégies directionnelles + 1 filtre, paramètres fixés a priori
(aucune optimisation). Chaque stratégie est exécutée seule dans le même environnement que l'agent : entrée au signal si aucune
position, stop à l'invalidation de la stratégie (au moins le stop minimal de coûts, signal ignoré au-delà de 40 pips),
TP en multiple de R, clôture forcée à 16:45 NY. Trois variantes par stratégie : TP 4R, TP 2R, TP 4R + filtre
volatilité/session. Soit 33 essais.

Tests (5 nouveaux, 27 au total, tous verts) : absence de look-ahead sur **toutes** les sorties de la bibliothèque
(perturbation du futur), opening range de Londres avec changement d'heure, confirmation des swings après k barres,
détection et réaction FVG, cohérence des invalidations.

## 2. Données et périodes

Entraînement uniquement : 2015-01-05 → 2025-06-30, comptes réinitialisés chaque mois (pour que l'arrêt global interne
ne tronque pas l'historique). Coûts identiques à l'étape A ; stress : spread × 1,5 et glissements × 2.
La calibration n'a pas été consultée (aucune variante n'a passé la porte). Validation et test protégé : non utilisés.

## 3. Porte de passage (pré-enregistrée avant la mesure)

≥ 300 trades ; espérance nette ≥ +0,05 R **et** t ≥ 3,0 (correction pour ~33 essais) ; ≥ 65 % d'années positives ;
espérance positive sous coûts stressés ; puis espérance positive sur la calibration.

## 4. Résultats chiffrés (entraînement, TP 4R)

| Stratégie | Signaux/jour | Trades | Net R/trade | t | PF | Années + | Stress R/trade | **Brut R/trade (sans coûts)** | t brut |
|---|---|---|---|---|---|---|---|---|---|
| trend_pullback | 28,3 | 5 685 | −0,18 | −7,4 | 0,79 | 0 % | −0,23 | +0,001 | +0,04 |
| squeeze_breakout | 0,9 | 1 922 | −0,15 | −3,6 | 0,82 | 0 % | −0,28 | +0,020 | +0,46 |
| range_meanrev | 3,1 | 2 845 | −0,16 | −4,5 | 0,82 | 0 % | −0,26 | +0,065 | +1,81 |
| liquidity_sweep | 3,5 | 3 522 | −0,24 | −7,8 | 0,73 | 0 % | −0,33 | +0,016 | +0,52 |
| mtf_momentum | 6,8 | 4 034 | −0,23 | −7,9 | 0,74 | 0 % | −0,30 | −0,005 | −0,19 |
| vwap_reclaim | 3,7 | 4 091 | −0,17 | −6,0 | 0,80 | 9 % | −0,24 | +0,003 | +0,11 |
| session_level_reject | 1,6 | 2 991 | −0,23 | −6,5 | 0,76 | 0 % | −0,36 | +0,047 | +1,36 |
| orb | 1,6 | 3 908 | −0,15 | −5,2 | 0,82 | 9 % | −0,22 | +0,012 | +0,41 |
| break_retest | 0,9 | 2 137 | −0,22 | −6,0 | 0,73 | 9 % | −0,31 | −0,027 | −0,73 |
| fvg_reaction | 8,1 | 4 943 | −0,24 | −9,1 | 0,74 | 0 % | −0,31 | +0,021 | +0,86 |
| fib_confluence | 5,2 | 2 472 | −0,20 | −5,6 | 0,76 | 18 % | −0,27 | +0,007 | +0,21 |

Variantes TP 2R : toutes négatives (−0,18 à −0,28 R, t de −6 à −13).
Variantes avec filtre : toutes négatives (−0,10 à −0,30 R) ; les moins mauvaises sont `orb` (−0,10, t −2,5) et
`fib_confluence` (−0,13, t −2,3). Détail complet : `stage_b_strategies.json` et `stage_b_gross.json`.

**Aucune des 33 variantes ne passe la porte.**

## 5. Problèmes détectés / diagnostic

1. **Espérance brute nulle** : sans aucun coût, les 11 stratégies sont à 0 ± 0,07 R par trade, aucune significative
   (meilleur t = 1,8 pour `range_meanrev`, attendu par hasard sur 11 essais). À l'échelle M5-M15 avec des stops de 5 à
   9 pips, EURUSD 2015-2025 se comporte comme une marche aléatoire pour ces configurations.
2. Le résultat net (−0,15 à −0,25 R) est simplement le coût : 1,3 pip aller-retour pour des stops médians de 5 à 9 pips.
3. La constance est frappante : 0 % d'années positives pour la plupart des stratégies, ce n'est pas une affaire de régime.
4. Conséquence pour le renforcement : un filtre PPO « WAIT ou sens de la stratégie » n'aurait à filtrer que des signaux
   d'espérance brute nulle ; l'étape A a déjà montré que PPO ne trouve pas d'avantage conditionnel dans ces features.

## 6. Décision

Porte non franchie : **pas de passage au renforcement sur cette bibliothèque**. Aucun champion. Bibliothèque, tests et
backtest de règles conservés (réutilisables tels quels pour d'autres échelles).

## 7. Expériences suivantes possibles

Le levier principal identifié est le rapport coût / risque : à 1,3 pip de coût, un stop de 7 pips perd ~0,2 R par trade
avant toute question de signal ; un stop de 25-40 pips ne perd que ~0,03-0,05 R.

- **B2 (dans le cahier des charges actuel)** : mêmes familles de stratégies à l'échelle H1/H4, stops en ATR H1
  (≈ 15-30 pips), détention jusqu'à la clôture forcée de 16:45 NY. Même porte, nouveaux essais comptés.
- **B3 (exige de modifier le cahier des charges)** : horizon swing (positions sur plusieurs jours, permis par le compte
  FTMO Swing), stops en ATR D1. Contredit la règle « aucune position overnight ».
- Calendrier économique : expérience isolée, prévue par le cahier des charges, si l'export est disponible.

## 8. Test protégé

Non consulté.
