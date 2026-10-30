# Protocole pré-enregistré — Lot 3 (S21-S23), cassures de Toby Crabel

Rédigé et commité le 2026-10-30, **avant** tout backtest de S21-S23. Cadre : `docs/protocole_catalogue.md` et
`docs/protocole_lot2.md` (stratégies sans paramétrage, Holm sur la famille cumulée, qui passe à **27**).
Source : T. Crabel, *Day Trading with Short Term Price Patterns and Opening Range Breakout* (1990), règles reprises
de oxfordstrat.com (« stretch ») ; NR7 = range de la veille le plus étroit des 7 derniers jours.
Instruments : FX4, IDX, BTCUSD ; fenêtres et clôtures forcées du catalogue (§2). Jours D1 = jours UTC.

**S21 — Stretch breakout.** Bruit d'un jour = min(haut − ouverture, ouverture − bas) du jour UTC ;
stretch = 2 × moyenne du bruit des 10 jours précédents. O = ouverture de la fenêtre (première minute).
Première clôture M1 au-delà de O + stretch (long) ou O − stretch (short), jusqu'à 60 min avant la fin de fenêtre ;
entrée à la minute suivante ; stop = l'autre niveau ; sortie à la clôture forcée.

**S22 — Stretch breakout les jours NR7.** S21 uniquement si la veille est un NR7.

**S23 — ORB 30 minutes les jours NR7.** Si la veille est un NR7 : range des 30 premières minutes de la fenêtre ;
première clôture M5 au-delà, dans les 3 h suivantes ; stop = bord opposé ; sortie à la clôture forcée.

Validité : jours retenus par le filtre de couverture du catalogue (≥ 50 % des minutes de la fenêtre).
