# Protocole pré-enregistré — Lot 2 (S16-S20), stratégies publiées reprises sans paramétrage

Rédigé et commité le 2026-10-30, **avant** tout backtest de S16-S20. Cadre commun : `docs/protocole_catalogue.md`.

## 1. Correction cumulative

La famille passe à **24 stratégies** (19 précédentes + 5). Holm à α = 5 % sur les 24 p-valeurs ; les p-valeurs des
19 précédentes sont reprises telles quelles. Chaque nouveau lot augmentera la famille de la même façon.
Les 5 stratégies sont prises **telles que publiées**, sans grille : pas de sélection walk-forward, tous leurs trades
de la période de développement comptent comme hors échantillon. Leur période de découverte chevauche souvent notre
période de développement : seule la période finale (après publication) constitue un test réellement indépendant.

**Candidat** = significatif après Holm sur la famille cumulée + espérance > 0 sous stress des coûts. Les candidats
sont ensuite combinés et soumis aux critères de portefeuille du protocole du catalogue (300 trades, PF ≥ 1,3,
DD ≤ 6 %, Monte Carlo ≥ 50 %), puis à la période finale.
**Piste** (information, sans décision) : p < 0,05 non corrigée et espérance > 0 sous stress.

## 2. Stratégies

**S16 — ORB 5 minutes** (Zarattini & Aziz, 2023 ; SSRN 4416622). US100, US500 (ouverture 09:30 New York),
GER40 (09:00 Berlin). Première bougie de 5 minutes de la séance : sens = sens de la bougie (doji ignoré).
Entrée à l'ouverture de la deuxième bougie ; stop = autre extrémité de la première bougie ; objectif = 10 R ;
sinon sortie à la clôture forcée (15:58 New York, 17:28 Berlin).

**S17 — Noise Area + trailing VWAP** (Zarattini, Aziz & Barbon, 2024, « Beat the Market »). US100, US500, GER40.
σ(m) = moyenne, sur les 14 séances précédentes, de |clôture à la minute m de la séance / ouverture − 1|.
Bornes : haute = max(ouverture, clôture veille) × (1 + σ), basse = min(ouverture, clôture veille) × (1 − σ).
Décisions uniquement aux clôtures de :00 et :30, à partir de ouverture + 30 min. Hors position : clôture > borne haute
→ long ; < borne basse → short. En position longue : clôture < max(borne haute, VWAP) → sortie ; si de plus
clôture < borne basse → retournement short (symétrique en short). Exécution à l'ouverture de la minute suivante.
VWAP de séance pondéré par le volume tick. Sortie forcée en fin de séance.
Adaptation au risque FTMO (non publiée, fixée ici) : stop d'urgence touché en continu à 1 largeur de bruit
(σ × ouverture) de l'entrée ; 1 R = cette distance ; entrée ignorée si 1 R < 3 fois le coût aller-retour.

**S18 — Gotobi, long avant le fixing** (arXiv 2301.13204). USDJPY. Jours Gotobi : 5, 10, 15, 20, 25 et 30
(dernier jour du mois s'il est plus court) ; si week-end, jour ouvré précédent (jours fériés japonais non gérés).
Long à 08:00 heure de Tokyo (après le reset FTMO dans les deux saisons), sortie à 09:55. Stop 0,25 × ATR D1.

**S19 — Gotobi, short après le fixing.** Mêmes jours. Short à 09:56 heure de Tokyo, sortie à 12:00.
Stop 0,25 × ATR D1. (L'étude conditionne ce trade au profit réalisé avant 9:00 ; version inconditionnelle retenue.)

**S20 — BTC 21:00 → 23:00 UTC** (Quantpedia : heures 22:00 et 23:00 UTC les plus fortes, 2015-2022).
Long tous les jours (week-ends compris) à 21:00 UTC, sortie à 23:00 UTC. Stop 0,5 × ATR D1.
Le trade peut franchir le reset FTMO de minuit à Prague en été ; il est rattaché au jour de son entrée.
