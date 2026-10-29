# Jalon A — rapport (période de développement)

Date : 2026-10-30. Protocole : `docs/protocole_jalon_A.md`, commité avant le premier test (commit `0ccc7ce`).
Résultats bruts : `reports_v2/jalon_a_dev.json`. Reproduire : `python -m v2.jalon_a`.

## Décision : NO-GO

La stratégie de breakout de session (étage 1) ne remplit pas les critères du jalon A.
4 critères sur 7 échouent, dont l'espérance, qui est négative. Conformément au protocole, la période
finale (2024-07 → 2026-10) **n'a pas été lancée** : elle reste vierge pour une prochaine itération.

| Critère | Seuil | Obtenu (walk-forward hors échantillon, 2018-01 → 2024-06) | Statut |
|---|---|---|---|
| Nombre de trades | ≥ 300 | 382 | OK |
| Espérance après coûts | ≥ 0,20 R | −0,064 R | Échec |
| Profit factor | ≥ 1,3 | 0,82 | Échec |
| Drawdown maximal | ≤ 6 % | −20,0 % | Échec |
| Réussite Monte Carlo FTMO | ≥ 50 % | 0,02 % | Échec |
| Espérance sous stress des coûts | > 0 | −0,096 R | Échec |
| Robustesse (voisins) | majorité > 0 | USDJPY et US100 : OK | OK |

Résultat total sur la période : −12,1 % du compte, à 0,5 % de risque par trade.

## Pourquoi ça échoue

1. **Sur le forex, la cassure n'a pas d'avantage, même avant coûts.** Sur toute la période de développement et
   en moyenne sur les 32 configurations, le R brut (avant coûts) vaut −0,03 sur EURUSD, −0,09 sur AUDUSD et
   +0,02 sur USDJPY. Après coûts : −0,07, −0,14 et −0,03. Aucune configuration EURUSD ou AUDUSD n'est positive,
   même en choisissant la meilleure après coup.
2. **Le walk-forward a donc presque toujours refusé de trader.** 7 semestres tradés sur 49 :
   EURUSD, AUDUSD et BTCUSD jamais (statistique t d'entraînement < 1), USDJPY 5 fois sur 13, US100 2 fois sur 2.
   Les semestres tradés ont perdu : USDJPY −0,088 R (250 trades), US100 −0,020 R (132 trades).
3. **US100 est le seul signal intéressant, mais il ne passe pas le hors échantillon.** Sur 2021-09 → 2024-06,
   les 32 configurations sont toutes positives en échantillon (+0,09 R en moyenne, meilleure t = 2,3).
   Sur ses deux semestres hors échantillon (2023-S2, 2024-S1), l'espérance tombe à −0,02 R.
   L'historique M1 FTMO du US100 ne couvre que 5 ans, ce qui ne suffit pas à trancher.
4. **Trailing vs objectif fixe 4R (information)** : USDJPY +0,039 R avec trailing contre −0,041 R avec 4R ;
   US100 +0,097 R contre +0,101 R. Pas de différence décisive.

## Vérifications effectuées

- **Mécanique** : 7 tests unitaires à résultat connu (`tests_v2/`) : cassure, filtre de tendance, stop, gap,
  trailing, absence de look-ahead, règles FTMO.
- **Réimplémentation indépendante** : 1 200 sessions tirées au hasard (EURUSD, US100, BTCUSD, configurations
  aléatoires) recalculées ligne par ligne par un code séparé : 1 200 accords, écart maximal 0 R.
- **Calage horaire** : le volume du US100 saute exactement à 09:30 New York, celui de l'EURUSD à 08:00 Londres,
  et les mouvements des NFP tombent à 08:30 New York.
- **Cohérence comptable** : −0,064 R × 382 trades × 0,5 % = −12,2 % ; résultat mesuré −12,1 %.

## Données et limites

- Spread de barre MT5 non fiable avant 2022 : le coût de spread de 2015-2021 est un profil estimé sur
  2022-2024 (protocole §4). Les conclusions ne dépendent pas du coût : le forex est négatif avant coûts.
- Commission forex (3,5 $/lot/côté) et crypto (3,25 bps/côté), leviers Swing : hypothèses à confirmer sur un
  trade démo. Elles n'influencent pas la conclusion.
- Le Monte Carlo est presque toujours « non conclu » (99 %) : avec 7 semestres tradés sur 49, la plupart des
  journées rééchantillonnées sont sans trade.

## Ce que ça implique pour la suite

Le module de risque, le moteur, le simulateur FTMO et le pipeline de validation sont réutilisables tels quels.
C'est le générateur de setups qui doit changer. Toute nouvelle hypothèse doit faire l'objet d'un nouveau
protocole pré-enregistré, testée sur la période de développement, et la période finale reste réservée.
