# Protocole pré-enregistré — Confirmation sur l'historique indépendant Dukascopy (indices 2013-2021)

Rédigé et commité le 2026-10-30, **avant** d'exécuter la moindre stratégie sur ces données.
Seuls contrôles déjà faits : format, couverture, calage horaire (volume à 09:30 New York) et dates FOMC (pic à 14:00).

## 1. Données

Dukascopy, minutes BID, horodatage UTC, téléchargées via `dukascopy-node` (usa500idxusd, usatechidxusd, deuidxeur) :
DK_US500 et DK_US100 de 2013-01 à 2021-09, DK_GER40 de 2013-09 à 2021-09.
**Fenêtre de test : début → 2021-08-31**, pour n'avoir aucun recouvrement avec les minutes FTMO (indices à partir de 2021-09).
Aucune stratégie n'a jamais vu ces données. Il n'y a pas de période d'entraînement : les règles et configurations sont
figées ci-dessous à partir des seuls résultats de développement FTMO.

## 2. Coûts

Pas de spread dans les données Dukascopy. Coût de spread par minute = profil p75 FTMO par quart d'heure (même
instrument FTMO, fenêtre 2022-01 → 2024-06), exprimé **en fraction du prix** (profil en points / prix médian FTMO de la
fenêtre), puis multiplié par le prix Dukascopy ; plancher idem en fraction du prix. Glissements, commission (0),
stops : identiques aux instruments FTMO correspondants, convertis de la même façon en fraction du prix.
Quart d'heure « serveur » = heure UTC + 2 h (+3 h pendant l'heure d'été américaine), comme FTMO.

## 3. Hypothèses confirmatoires (5) — les seules qui comptent pour la décision

Stratégies de la famille positives en développement sur les indices, règles inchangées :

| Code | Stratégie | Configuration | Instruments |
|---|---|---|---|
| S2 | Volatility breakout | k = 0,3 ; s = 0,5 ; filtre de tendance (configuration la plus retenue en développement) | US500, US100, GER40 |
| S10 | Dérive avant la Fed | sans paramètre ; dates `data/calendar/fomc_2013_2020.csv` (réunions programmées) | US500, US100 |
| S11 | Tournant de mois | sans paramètre | US500, US100, GER40 |
| S25 | Veille de jour férié | sans paramètre ; jour férié = jour de semaine sans séance (≥ 50 % des minutes) | US500, US100, GER40 |
| S26 | Journée de tendance | sans paramètre | US500, US100, GER40 |

Pour chaque stratégie : portefeuille autonome (module de risque commun, 0,5 %), statistique t du R effectif, p unilatérale.
**Correction de Holm sur ces 5 p-valeurs, α = 5 %.**
**Confirmée** si : significative après Holm **et** espérance > 0 avec spread × 1,5 et glissements × 2.

Une stratégie confirmée devient candidate. Elle devra encore passer : portefeuille des candidates (≥ 300 trades sur
FTMO + Dukascopy, PF ≥ 1,3, drawdown ≤ 6 %, Monte Carlo FTMO ≥ 50 %), puis la période finale FTMO (2024-07 → 2026-10).

## 4. Analyse exploratoire (information uniquement, aucune décision)

Toutes les autres stratégies applicables aux indices, configuration `grid[0]` ou sans paramètre :
S3, S4, S5, S6, S7, S8, S9, S16, S17, S21, S22, S23, S24, S27, S28, plus la paire S15 (US100/US500).
Résultats publiés à titre de contrôle (les stratégies négatives en FTMO doivent l'être ici aussi si les deux sources concordent).
