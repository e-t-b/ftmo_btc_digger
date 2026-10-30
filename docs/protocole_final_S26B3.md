# Protocole pré-enregistré — Test final unique : S26-B3 sur la période finale FTMO

Rédigé et commité le 2026-10-30, **avant** toute exécution sur la période finale (2024-07-01 → 2026-10-28),
qui n'a été lue par aucune stratégie de ce projet. Ce test est lancé **une seule fois**. Aucune règle ne peut être
modifiée après avoir vu son résultat ; un nouvel essai nécessiterait une nouvelle période jamais vue.

## Stratégie (code : `v2/s26x.py`, fonction `setups`, variante B)

- Instruments : US500.cash, US100.cash (séance 09:30-16:00 New York), GER40.cash (09:00-17:30 Berlin).
- Décisions à ouverture + 1 h, + 2 h, + 3 h (clôture de la minute précédente) ; premier signal du jour retenu,
  un trade au plus par instrument et par jour.
- Signal : |prix − ouverture| ≥ 0,5 × ATR14 D1 et ≤ 1,0 × ATR14 D1 ; prix dans les 20 % extrêmes du range de la
  séance, dans le sens du mouvement ; ATR14 D1 ≥ 0,8 % du prix d'ouverture.
- Entrée au marché à la minute suivante ; stop à mi-chemin entre l'entrée et l'ouverture ; pas d'objectif ;
  sortie forcée 15:58 New York / 17:28 Berlin.
- Coûts : modèle FTMO de `v2/config.py` (inchangé).

## Risque

0,75 % du solde initial par trade (0,375 % sous 96 % du solde, arrêt sous 93 %) ; stop journalier interne −2 % ;
risque ouvert ≤ 1,5 % ; pas de plafond par bloc. Compte 10 000 $, FTMO 2-Step.

## Critères de décision (tous requis)

1. Espérance > 0 R sur la période finale ;
2. Profit factor ≥ 1,1 ;
3. Réussite Monte Carlo FTMO (phase 1 puis 2, 20 000 trajectoires, blocs de 5 jours) ≥ 40 % sur les journées
   de la période finale ;
4. Drawdown maximal ≤ 10 % (limite FTMO).

**GO** → 1 à 2 mois sur le compte d'essai gratuit FTMO avant tout achat de challenge. **NO-GO** → arrêt de la
piste du bot intraday sur ces données.

Informations publiées sans effet sur la décision : résultat par indice, par heure de décision, par semestre,
et S26 d'origine (décision unique à +2 h, stop à l'ouverture, risque 0,5 %).
