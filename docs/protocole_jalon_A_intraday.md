# Protocole pré-enregistré — Jalon A, hypothèses intraday H1-H3

Rédigé et commité le 2026-10-30, **avant** tout backtest de H1, H2 ou H3.
Reprend le cadre de `docs/protocole_jalon_A.md` (données, coûts, module de risque, simulateur FTMO,
découpage développement / période finale), sauf mention contraire ci-dessous.
Contrainte du projet : intraday uniquement, toute position est ouverte et fermée dans la même journée FTMO.

## 1. Instruments et séances

| Instrument | M1 utilisable | Séance cash (locale) | Hypothèses |
|---|---|---|---|
| US100.cash | 2021-09 → | 09:30-16:00 America/New_York | H1, H3 |
| US500.cash | 2021-09 → | 09:30-16:00 America/New_York | H1, H3 |
| US30.cash | selon export | 09:30-16:00 America/New_York | H1, H3 |
| GER40.cash | 2022-01 → | 09:00-17:30 Europe/Berlin | H1, H3 |
| BTCUSD | 2017-04 → | 09:30-16:00 America/New_York (fenêtres horaires, marché 24/7) | H2 |

Clôture de la veille (P_c) : clôture BID de la dernière minute strictement avant l'heure de clôture de la
séance précédente ayant des données (indices) ; pour BTCUSD, prix à 16:00 New York la veille calendaire.
ATR14 D1 : jours UTC terminés avant l'ouverture (comme le protocole initial).

## 2. Hypothèses

**H1 — momentum de fin de séance (indices).**
r1 = prix BID à ouverture + 30 min / P_c − 1. Setup si |r1| ≥ thr × ATR14 D1 / P_c.
Entrée au marché à l'ouverture de la minute E avant la clôture, dans le sens de r1.
Stop = entrée ∓ m × ATR14 D1. Sortie à l'ouverture de la minute « clôture − 2 min ».
Grille : thr ∈ {0 ; 0,2} ; E ∈ {30 ; 60} min ; m ∈ {0,25 ; 0,5}.

**H2 — même règle sur BTCUSD**, sur les horaires New York ci-dessus. Même grille.

**H3 — comblement de gap à l'ouverture (indices).**
gap = ouverture BID de la première minute de séance − P_c. Setup si lo ≤ |gap| / ATR14 D1 ≤ hi.
Entrée au marché à l'ouverture de la deuxième minute de séance, à l'opposé du gap.
Objectif = P_c (comblement complet) ; stop = entrée ± |gap| ; sortie temporelle à ouverture + X.
Grille : lo ∈ {0,2 ; 0,3} ; hi ∈ {0,8 ; 1,2} ; X ∈ {30 ; 60} min.

Dans les trois cas : un trade au plus par instrument, par hypothèse et par jour ; stop vérifié sur chaque
minute (BID pour un long, ASK reconstruit pour un short) ; gap au-delà du stop exécuté à l'ouverture ;
trade ignoré si le stop est inférieur à 3 fois le coût aller-retour.

## 3. Coûts des indices ajoutés

| Instrument | Fenêtre de confiance du spread | Plancher | Commission | Glissement entrée / sortie | Supplément sur stop |
|---|---|---|---|---|---|
| US500.cash | 2022 → 2026 | 0,25 point | 0 | 0,25 point | 0,75 point |
| US30.cash | 2022 → 2026 | 1,5 point | 0 | 1,5 point | 4 points |
| GER40.cash | 2022-02 → 2026 | 1,0 point | 0 | 1,0 point | 3 points |

US100 et BTCUSD : inchangés. Profil de spread estimé sur 2022-01-01 → 2024-06-30. Levier indices 1:15.

## 4. Walk-forward

Fenêtre d'entraînement croissante, semestres de test civils. Premier semestre de test :
indices 2023-S2 ; BTCUSD 2020-S2. Sélection : meilleure statistique t du R net avec au moins 60 trades
d'entraînement ; instrument non tradé au semestre si t < 1,0.

## 5. Critères (chaque hypothèse séparément, puis la combinaison des hypothèses retenues)

1. au moins 300 trades hors échantillon ;
2. **espérance > 0 avec une statistique t ≥ 2,0** (remplace « ≥ 0,20 R » : les sorties étant surtout
   temporelles, l'espérance en R dépend de la largeur du stop ; décision prise avant tout test) ;
3. profit factor ≥ 1,3 ;
4. drawdown maximal ≤ 6 % ;
5. probabilité de réussite Monte Carlo (phase 1 puis 2) ≥ 50 % ;
6. espérance > 0 avec spread × 1,5 et glissement × 2 ;
7. robustesse : configuration la plus retenue et majorité de ses voisines positives sur la période de développement.

Trois hypothèses sont testées : une hypothèse qui passe « de justesse » doit être lue en tenant compte de ces
essais multiples. Période finale (2024-07-01 → 2026-10-28) : lancée une seule fois, uniquement pour les
hypothèses qui remplissent 1-7, pipeline figé. GO si espérance > 0 et profit factor ≥ 1,1.
