# Protocole pré-enregistré — Lot 4 (S24-S26)

Rédigé et commité le 2026-10-30, **avant** tout backtest de S24-S26. Cadre : protocoles du catalogue et des lots 2-3.
Stratégies sans paramétrage ; famille cumulée : **30**.

**S24 — Continuation après la décision de la Fed (US100, US500).** Jours FOMC (`data/calendar/fomc.csv`).
r = variation entre la clôture de 13:59 et celle de 14:29 (New York). Position dans le sens de r à l'ouverture de 14:30 ;
stop = clôture de 13:59 ; sortie à la clôture forcée (15:58). Fondement : réaction lente des prix à l'annonce.

**S25 — Veille de jour férié (US100, US500, GER40).** Jour ouvré suivi d'un jour de semaine sans séance pour
l'instrument (jour férié de la bourse, connu à l'avance), dans la période où les minutes de l'instrument sont
disponibles. Long à l'ouverture + 1 min, sortie à la clôture forcée, stop 0,5 × ATR D1. Fondement : effet
pré-férié documenté sur les indices actions (Ariel, 1990).

**S26 — Journée de tendance (US100, US500, GER40, BTCUSD).** À ouverture + 120 min : si |prix − ouverture| ≥
0,5 × ATR D1 et si le prix est dans les 20 % extrêmes du range de la séance dans le sens du mouvement → position
dans ce sens à la minute suivante ; stop = ouverture de séance ; sortie à la clôture forcée.
Règle formulée ici (pas de source unique) : hypothèse de continuation des journées de tendance.
