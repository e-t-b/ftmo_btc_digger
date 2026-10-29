# Protocole pré-enregistré — Jalon A (refonte v2)

Rédigé et commité le 2026-10-30, **avant** le premier backtest de la stratégie v2.
Toute modification ultérieure de ce fichier doit être commitée séparément, datée et justifiée ;
un changement de règle après avoir vu un résultat invalide le jalon pour la période concernée.

## 1. Données

Source : terminal MT5 du compte démo FTMO (prix BID M1 + spread de barre), export `export_mt5.py`.

| Instrument | M1 disponible (UTC) | Session tradée | Ouverture locale |
|---|---|---|---|
| EURUSD | 2015-01 → 2026-10 | Londres | 08:00 Europe/London |
| USDJPY | 2015-01 → 2026-10 | Tokyo | 09:00 Asia/Tokyo |
| AUDUSD | 2015-01 → 2026-10 | Tokyo | 09:00 Asia/Tokyo |
| US100.cash | 2021-09 → 2026-10 | New York | 09:30 America/New_York |
| BTCUSD | 2017-04 → 2026-10 | New York | 09:30 America/New_York |

Heure serveur : GMT+2, GMT+3 pendant l'heure d'été américaine. Tout est converti en UTC.
Jour FTMO (perte journalière) : minuit Europe/Prague.

Constats du contrôle qualité :
- Spread de barre non homogène : 2015-2019 élevé, 2020-2021 à 50 points constants sur le forex,
  zéros fréquents sur EURUSD en 2025-2026, 100 points constants sur BTCUSD en 2026.
- US100 : avant septembre 2021, une seule barre par jour (inutilisable en intraday).

## 2. Découpage temporel

- **Période de développement** : début des données → 2024-06-30.
- **Période finale (holdout)** : 2024-07-01 → 2026-10-28. Non touchée tant que les critères de
  développement ne sont pas remplis. Elle n'est **pas lancée du tout** si le jalon échoue en développement,
  pour rester vierge pour une itération suivante.
- **Walk-forward** dans la période de développement : fenêtre d'entraînement croissante (expanding),
  pas de test de 6 mois (semestres civils). Premier semestre de test :
  - forex : 2018-S1 (entraînement 2015-01 → 2017-12) ;
  - BTCUSD : 2020-S2 (entraînement 2017-04 → 2020-06) ;
  - US100 : 2023-S2 (entraînement 2021-09 → 2023-06).

## 3. Stratégie étage 1 (breakout de session)

Pour chaque instrument et chaque jour ouvré :
1. **Range d'ouverture (OR)** : plus haut / plus bas BID des N premières minutes après l'ouverture locale.
2. **Tendance de fond** : clôture D1 de la veille (jour UTC complet précédent) comparée à son EMA(L)
   calculée sur les clôtures D1 jusqu'à la veille. Long seulement si clôture > EMA, short seulement si < EMA.
3. **Volatilité** : ratio = (H_OR − L_OR) / ATR14 D1 (jusqu'à la veille). Setup valide si lo ≤ ratio ≤ hi.
4. **Entrée** : première barre M5 clôturée au-delà de l'OR dans le sens de la tendance, dans les 180 min
   qui suivent la fin de l'OR. Exécution au marché à l'ouverture de la minute suivante (ask pour un long).
   Un seul trade par instrument et par jour, pas de réentrée.
5. **Stop initial** : bord opposé de l'OR. Il définit 1R. Trade ignoré si le stop est inférieur à 3 fois
   le coût aller-retour estimé.
6. **Sortie** : trailing stop chandelier, mis à jour à chaque clôture M5 :
   stop_long = max(stop, plus haut depuis l'entrée − k × ATR14(M15)) (symétrique en short).
   Pas d'objectif fixe. Clôture forcée à 16:45 New York (forex, BTCUSD) ou 15:55 New York (US100),
   donc avant le rollover et le reset FTMO.
7. Les stops sont vérifiés sur chaque barre M1 (BID pour un long, ASK reconstruit pour un short) ;
   un gap au-delà du stop est exécuté à l'ouverture de la barre.

Grille (32 configurations) : N ∈ {30, 60} min ; L ∈ {20, 50} ; lo ∈ {0, 0,15} ; hi ∈ {0,5, 1,0} ; k ∈ {2,0, 3,5}.

Variante de référence (information, jamais sélectionnable) : objectif fixe à 4R à la place du trailing.

## 4. Coûts

- **Spread** par minute = max(profil p75 par quart d'heure serveur, spread observé si la date est dans la
  fenêtre de confiance, plancher). Profil estimé sur 2022-01-01 → 2024-06-30 (dans le développement).

| Instrument | Fenêtre de confiance | Plancher | Commission / côté | Glissement entrée et sortie temporelle | Glissement supplémentaire sur stop |
|---|---|---|---|---|---|
| EURUSD | 2022 → 2024 | 2 points (0,2 pip) | 3,5 $ / lot | 0,2 pip | 0,5 pip |
| USDJPY | 2022 → 2026 | 2 points | 3,5 $ / lot | 0,2 pip | 0,5 pip |
| AUDUSD | 2022 → 2026 | 2 points | 3,5 $ / lot | 0,2 pip | 0,5 pip |
| US100 | 2022 → 2026 | 0,5 point | 0 | 0,5 point | 1,5 point |
| BTCUSD | 2022 → 2025 | 10 $ | 3,25 bps | 1 bp | 3 bps |

- Pas de swap : toutes les positions sont clôturées avant 17:00 New York.
- Hypothèses à confirmer sur un trade démo : commission forex et crypto du compte Swing.

## 5. Sélection walk-forward

Pour chaque instrument et chaque fenêtre d'entraînement : configuration de la grille qui maximise la
statistique t du R net moyen (moyenne / écart-type × √n), parmi celles ayant au moins 60 trades.
Si la meilleure statistique t est < 1,0, l'instrument n'est **pas tradé** au semestre suivant.
Le semestre de test utilise la configuration choisie, sans la voir.

## 6. Module de risque (identique backtest / live)

Compte 10 000 $ FTMO 2-Step Swing.
- Risque par trade : 0,5 % du solde initial ; 0,25 % si l'equity est sous 96 % du solde initial ;
  arrêt total sous 93 %.
- Stop journalier interne : equity (flottant compris) ≤ equity de début de jour FTMO − 2 % du solde initial
  → clôture de tout et aucune entrée jusqu'au jour FTMO suivant.
- Risque ouvert cumulé ≤ 1,5 % ; au plus 2 positions exposées du même côté du dollar
  (EURUSD/AUDUSD long = dollar short ; USDJPY long = dollar long).
- Taille arrondie au pas de volume ; trade ignoré si inférieure au minimum.
- Marge : somme des notionnels / levier ≤ equity, avec levier 1:30 forex, 1:15 indices, 1:1 crypto (hypothèses Swing).

## 7. Simulateur FTMO

Phase 1 : +10 % ; phase 2 : +5 % ; perte journalière : equity ≤ solde de début de jour − 5 % du solde
initial ; perte maximale : equity ≤ 90 % du solde initial (statique) ; 4 jours de trading minimum par phase.

Monte Carlo : bootstrap par blocs de 5 jours ouvrés des journées hors échantillon du portefeuille
(chaque journée garde ses trades en R et son pire flottant intrajournalier), 20 000 trajectoires,
horizon 250 jours ouvrés par phase. Réussite = phase 1 puis phase 2.

## 8. Critères du jalon A (portefeuille walk-forward hors échantillon, période de développement)

Tous doivent être remplis :
1. au moins 300 trades ;
2. espérance ≥ 0,20 R par trade après coûts ;
3. profit factor ≥ 1,3 ;
4. drawdown maximal ≤ 6 % du solde initial ;
5. probabilité de réussite Monte Carlo (phase 1 puis 2) ≥ 50 % ;
6. espérance > 0 avec spread × 1,5 et glissement × 2 ;
7. robustesse : pour chaque instrument tradé, la configuration la plus souvent retenue et la majorité
   de ses 5 voisines (un seul paramètre changé) ont une espérance > 0 sur toute la période de développement.

Si 1-7 sont remplis : test unique sur la période finale, pipeline figé. GO si espérance > 0 et
profit factor ≥ 1,1 sur la période finale. Sinon NO-GO.
