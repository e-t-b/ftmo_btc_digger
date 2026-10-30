# Protocole pré-enregistré — Catalogue de stratégies intraday (S1-S15)

Rédigé et commité le 2026-10-30, **avant** tout backtest des stratégies S1 à S15.
Cadre commun repris de `docs/protocole_jalon_A.md` et `docs/protocole_jalon_A_intraday.md` : données FTMO M1,
modèle de coûts, module de risque, simulateur FTMO, période de développement (→ 2024-06-30) et période finale
(2024-07-01 → 2026-10-28). Intraday uniquement : toute position est fermée dans la journée.

## 1. Famille testée et correction pour tests multiples

La famille compte **19 stratégies** : S1-S15 ci-dessous + les 4 déjà testées (breakout de session v2, H1, H2, H3).
Pour chaque stratégie, statistique = t du R effectif des trades hors échantillon (walk-forward, portefeuille
autonome de la stratégie, module de risque appliqué) ; p-valeur unilatérale (loi normale). Une stratégie sans
trade hors échantillon a p = 1. **Correction de Holm** au risque global α = 5 % sur les 19 p-valeurs.

**Sélection en développement (« survivantes »)** : significative après Holm, espérance > 0 avec spread × 1,5 et
glissement × 2, et (si la stratégie a une grille) configuration la plus retenue et majorité de ses voisines
positives sur la période de développement.

**Portefeuille des survivantes** (une seule simulation, module de risque commun) : au moins 300 trades,
profit factor ≥ 1,3, drawdown maximal ≤ 6 %, réussite Monte Carlo FTMO ≥ 50 %.

**Période finale** : lancée une seule fois, seulement s'il existe au moins une survivante, sur les survivantes et
sur S9. GO pour une stratégie si espérance > 0 et profit factor ≥ 1,1 sur la période finale ; S9, qui n'a pas de
test en développement, doit en plus y atteindre p < 0,05 / 19. Sans survivante : NO-GO, la période finale reste vierge.

## 2. Conventions communes

Instruments : FX4 = EURUSD, GBPUSD, USDJPY, AUDUSD ; IDX = US100, US500, GER40 ; BTC = BTCUSD.
(US30 non inclus : export indisponible au moment du protocole.)

| Groupe | Fenêtre d'entrée (heure locale) | Clôture forcée |
|---|---|---|
| FX4 | 07:00-16:00 Europe/London | 16:45 New York |
| US100, US500 | 09:30-16:00 New York | 15:58 New York |
| GER40 | 09:00-17:30 Europe/Berlin | 17:28 Europe/Berlin |
| BTCUSD | 09:30-16:00 New York | 15:58 New York |

- Jours ouvrés uniquement. Au plus 1 trade par stratégie, par instrument et par jour.
- Signal sur barre clôturée ; entrée au marché à l'ouverture de la minute suivante (ask pour un long).
- Stop vérifié chaque minute (BID long, ASK short), gap exécuté à l'ouverture ; objectif = ordre limite.
- Trade ignoré si le stop est inférieur à 3 fois le coût aller-retour.
- ATR14 D1 et veille = jours UTC terminés avant la décision ; « tendance » = clôture D1 de la veille vs EMA20 D1.
- ATR14 M15 et ATR14 H1 sur grille UTC, connus à la clôture de leur barre.
- Walk-forward : fenêtre croissante, semestres civils ; premier semestre de test FX4 2018-S1, BTC 2020-S2,
  indices 2023-S2 ; sélection par statistique t avec au moins 60 trades d'entraînement (30 pour S14) ;
  instrument non tradé au semestre si t < 1. Les stratégies sans paramètre (S9, S10, S11) n'ont pas de sélection :
  tous leurs trades de la période de développement sont hors échantillon. S12 se réestime à chaque semestre.

## 3. Stratégies

**S1 — Range asiatique cassé à Londres (FX4).** Range BID de 00:00 à R_fin (Londres). Première clôture M5
au-delà du range entre R_fin et R_fin + 3 h, dans les deux sens. Filtre : 0,1 ≤ range / ATR D1 ≤ 1,0.
Grille : R_fin ∈ {07:00, 08:00} ; stop ∈ {bord opposé, milieu du range} ; sortie ∈ {clôture forcée, objectif 1,5 R}.

**S2 — Volatility breakout (IDX, BTC).** P = range (haut − bas) de la veille. Niveaux = ouverture de séance ± k·P.
Première clôture M1 au-delà d'un niveau, jusqu'à 60 min avant la clôture. Stop = entrée ∓ s·P. Sortie : clôture forcée.
Grille : k ∈ {0,3 ; 0,6} ; s ∈ {0,5 ; 1,0} ; filtre ∈ {aucun, tendance}.

**S3 — Cassure du plus haut / plus bas de la veille (FX4, IDX, BTC).** Première clôture M5 au-delà du plus haut
ou du plus bas de la veille, dans les 4 h suivant le début de fenêtre (FX : 07:00-11:00 Londres).
Grille : stop ∈ {0,25 ; 0,5} × ATR D1 ; sortie ∈ {clôture forcée, trailing 3 × ATR M15} ; filtre ∈ {aucun, tendance}.

**S4 — Canal Donchian H1 (FX4, IDX, BTC).** Clôture H1 au-dessus du plus haut (ou sous le plus bas) des N barres
H1 précédentes ; signal entre début de fenêtre + 60 min et clôture − 60 min. Stop = a × ATR H1 ; trailing
chandelier 3 × ATR H1. Grille : N ∈ {20, 50} ; a ∈ {1,5 ; 2,5} ; filtre ∈ {aucun, prix vs EMA200 H1}.

**S5 — Retour au VWAP (IDX).** VWAP de séance (prix typique pondéré par le volume tick) ; σ = écart-type de
(clôture − VWAP) sur les minutes de la séance. À partir de l'ouverture + 30 min, première clôture M5 avec
|écart| > z·σ → position vers le VWAP. Objectif = VWAP au signal ; stop = s·σ au-delà de l'entrée.
Grille : z ∈ {2,0 ; 2,5} ; s ∈ {1,0 ; 2,0} ; fin de fenêtre ∈ {12:00 locale, clôture − 60 min}.

**S6 — Bollinger en régime de range (FX4, IDX).** Barres M15 ; ADX14 < a ; clôture sous la bande basse
BB(20, b) → long, au-dessus de la bande haute → short. Objectif = bande médiane au signal ; stop = c × ATR M15.
Fenêtre indices : ouverture + 30 min → clôture − 60 min. Grille : b ∈ {2,0 ; 2,5} ; a ∈ {20, 25} ; c ∈ {1,0 ; 2,0}.

**S7 — Fausse cassure de la veille, « Turtle Soup » (FX4, IDX, BTC).** Dans les W premières heures de fenêtre :
le prix dépasse le plus haut de la veille d'au moins x × ATR D1, puis une clôture M5 revient sous ce plus haut
→ short (symétrique au plus bas). Stop = extrême de la séance + 0,05 × ATR D1.
Grille : x ∈ {0 ; 0,1} ; W ∈ {2 h, 4 h} ; sortie ∈ {objectif 2 R, clôture forcée}.

**S8 — IBS horaire (IDX).** Barres H1 alignées sur l'ouverture de séance ; IBS = (clôture − bas) / (haut − bas)
de la barre terminée. IBS < q → long ; IBS > 1 − q → short ; signal entre ouverture + 60 min et
clôture − 60 min − durée. Sortie après la durée ; stop = 1 × ATR H1.
Grille : q ∈ {0,1 ; 0,2} ; durée ∈ {1 h, 2 h} ; filtre ∈ {aucun, tendance}.

**S9 — Retournement de fin de séance (IDX).** Règle H1 inversée, configuration fixe (seuil 0, entrée à
clôture − 30 min, stop 0,5 × ATR D1). **Observée après coup** : pas de test en développement, période finale seulement.

**S10 — Dérive avant la Fed (US100, US500).** Jour de décision FOMC : long à 09:31 New York, sortie 13:55,
stop 0,5 × ATR D1. Sans paramètre. Dates : `data/calendar/fomc.csv` (federalreserve.gov).

**S11 — Tournant de mois (IDX).** Dernier jour ouvré du mois et 3 premiers du mois suivant : long à
l'ouverture + 1 min, sortie à la clôture forcée, stop 0,5 × ATR D1. Sans paramètre.

**S12 — Saisonnalité horaire (FX4, IDX, BTC).** Pour chaque semestre, sur la fenêtre d'entraînement : rendement
de chaque heure locale de la fenêtre de trading (BTC : 24 heures UTC). Heure retenue = |t| maximal ; tradée au
semestre suivant seulement si |t| ≥ 2, dans le sens du rendement moyen, de l'heure pleine à +60 min, stop 1 × ATR H1.

**S13 — Fixing de Londres (FX4).** r = variation entre 16:00 − p et 15:59 (Londres). Si |r| ≥ thr × ATR D1 :
position opposée à r à l'ouverture de 16:01, sortie après la durée, stop 0,25 × ATR D1.
Grille : p ∈ {30, 60} min ; thr ∈ {0 ; 0,1} ; durée ∈ {30, 60} min.

**S14 — Dérive après NFP / CPI (FX4, IDX).** r5 = variation entre la clôture de 08:29 et celle de 08:34 (New York).
Si |r5| ≥ m × ATR D1 : position dans le sens de r5 à 08:35, stop = clôture de 08:29, sortie après X minutes.
Grille : événements ∈ {NFP, NFP + CPI} ; m ∈ {0 ; 0,1} ; X ∈ {60, 150}. Dates : `data/calendar/` (bls.gov).

**S15 — Paires en retour à la moyenne (US100/US500 ; EURUSD/GBPUSD).** Série synthétique S = A / B sur les
minutes communes ; coût de S = S × (spread A / A + spread B / B), glissements additionnés ; P&L d'une position
neutre en notionnel ≈ notionnel × variation relative de S ; marge des deux jambes (levier divisé par 2).
Sur les clôtures M5 de la fenêtre : z = (S − moyenne) / écart-type des N dernières clôtures M5.
|z| > z0 → position vers la moyenne ; stop au niveau correspondant à |z| = z0 + 1,5.
Grille : N ∈ {48, 96} ; z0 ∈ {2,0 ; 2,5} ; sortie ∈ {objectif = moyenne au signal, 60 min}.

## 4. Coûts additionnels

GBPUSD : comme EURUSD (fenêtre de confiance 2022 → 2024, plancher 2 points, 3,5 $/lot/côté, glissement 0,2 pip,
0,5 pip sur stop). Autres instruments : inchangés.
