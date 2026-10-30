# Protocole pré-enregistré — M1 : sélection de stratégie selon le contexte (méta-modèle)

Rédigé et commité le 2026-10-30, **avant** toute construction du jeu de données ou entraînement.
Cadre : protocoles précédents (coûts, module de risque, simulateur FTMO, période de développement → 2024-06-30,
période finale réservée). M1 compte comme **une** stratégie de plus : famille cumulée **33**, Holm à 5 %.

## 1. Réservoir de trades candidats

Tous les trades, sur la période de développement, des stratégies déjà testées :
- catalogue S1-S8, S10, S11, S13-S15, chacune dans sa **première configuration de grille** (`grid[0]`, choix
  arbitraire fixé ici pour ne pas sélectionner une configuration au vu des résultats) ;
- lots 2-5 : S16-S28.
Exclues : S9 (réservée à la période finale), S12 (adaptative), breakout de session et H1-H3 (autre moteur).

## 2. Variables de contexte (toutes connues à la clôture de la minute qui précède l'entrée)

Identifiants : stratégie, instrument, sens. Temps : heure UTC, minute du jour, jour de semaine, mois.
Jour UTC précédent : ATR14 D1 / prix, sens de la tendance (clôture vs EMA20 D1) × sens du trade,
range de la veille / ATR, écart prix actuel − clôture de la veille (en ATR, signé par le sens du trade).
Intrajournalier : ATR14 M15 / ATR D1, ATR14 H1 / ATR D1, rendement des 15, 60 et 240 dernières minutes en ATR
(signés par le sens du trade). Trade : distance du stop en ATR D1, coût estimé en R, objectif fixe (oui/non).
Calendrier : jour de NFP, CPI, FOMC. Historique propre : R net moyen et nombre des 50 derniers trades de la même
stratégie sur le même instrument **clôturés avant** l'entrée.

## 3. Modèle et walk-forward

Cible : R net du trade, borné à [−3, 10]. LightGBM (régression), paramètres fixés : 300 arbres, taux 0,03,
15 feuilles, 200 observations minimum par feuille, sous-échantillonnage 0,8 (lignes et colonnes), graine 7 ;
stratégie, instrument et sens en variables catégorielles.
Semestres de test civils de 2018-S1 à 2024-S1. Pour chaque semestre : entraînement sur tous les trades **clôturés
avant** le début du semestre (fenêtre croissante), au moins 2 000 trades sinon le semestre n'est pas tradé.
Règle de décision fixe : on prend un trade si la prédiction est > 0.

## 4. Évaluation et critères

Trades retenus → module de risque commun (0,5 %, stop journalier interne, plafonds) → métriques, statistique t,
p-valeur unilatérale, Monte Carlo FTMO. Stress : mêmes trades retenus, coûts × 1,5 (spread) et × 2 (glissements).
Information : même chose sans filtre (tous les trades du réservoir sur les mêmes semestres), pour mesurer l'apport.

**Candidat** si : significatif après Holm sur 33 ; espérance > 0 sous stress ; ≥ 300 trades ; PF ≥ 1,3 ;
drawdown ≤ 6 % ; Monte Carlo ≥ 50 %. Seulement alors : test unique sur la période finale, pipeline figé
(modèle réentraîné sur tout le développement, mêmes paramètres). Sinon NO-GO, période finale intacte.
