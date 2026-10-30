# Recherche de stratégies intraday — bilan des lots 2 à 4 (et famille cumulée)

Date : 2026-10-30. Protocoles : `docs/protocole_lot2.md`, `docs/protocole_lot3.md`, `docs/protocole_lot4.md`, chacun
commité avant son test. Famille cumulée et p-valeurs : `reports_v2/famille.json`. Reproduire : `python -m v2.lot_run lot2`
(puis `lot3`, `lot4`).

## Résultat : aucun candidat sur 30 stratégies

Seuil de Holm pour la première stratégie : p ≤ 0,05 / 30 ≈ 0,0017 (t ≈ 2,9). Meilleure p obtenue : 0,091 (S26).
Aucune « piste » au sens du protocole (p < 0,05 non corrigée et positive sous stress).
La période finale n'a été lancée pour aucune stratégie.

| Stratégie | Lot | Trades | Espérance (R) | t | p |
|---|---|---|---|---|---|
| S26 Journée de tendance (indices, BTC) | 4 | 416 | +0,042 | 1,33 | 0,091 |
| S2 Volatility breakout | catalogue | 172 | +0,100 | 0,78 | 0,217 |
| S19 Gotobi, short après le fixing | 2 | 676 | +0,024 | 0,69 | 0,245 |
| S10 Dérive avant la Fed | catalogue | 46 | +0,042 | 0,52 | 0,300 |
| S25 Veille de jour férié | 4 | 30 | +0,071 | 0,44 | 0,330 |
| S11 Tournant de mois | catalogue | 394 | +0,017 | 0,31 | 0,380 |
| S18 Gotobi, long avant le fixing | 2 | 676 | +0,005 | 0,15 | 0,439 |
| S17 Noise Area + VWAP (Zarattini et al.) | 2 | 2 128 | −0,004 | −0,14 | 0,556 |
| S16 ORB 5 minutes (Zarattini & Aziz) | 2 | 1 963 | −0,075 | −1,40 | 0,919 |
| S24 Continuation après la Fed | 4 | 42 | −0,351 | −1,41 | 0,921 |
| S22 Stretch Crabel, jours NR7 | 3 | 673 | −0,032 | −1,32 | 0,907 |
| S23 ORB 30 min, jours NR7 | 3 | 1 156 | −0,082 | −1,51 | 0,935 |
| S21 Stretch breakout (Crabel) | 3 | 6 461 | −0,024 | −2,64 | 0,996 |
| S20 BTC 21:00 → 23:00 UTC | 2 | 1 380 | −0,202 | −13,9 | 1,000 |

Les 16 autres stratégies de la famille (catalogue, breakout de session, H1-H3, S9) : voir `reports_v2/catalogue_dev.md`.

## Ce que montrent les données

- **La distribution des p-valeurs est celle d'une absence d'avantage, coûts compris.** Sans avantage réel, on attendrait
  1 à 2 stratégies sous p = 0,05 sur 30 par pur hasard ; on n'en observe aucune, et la majorité est négative.
- **Les stratégies publiées ne survivent pas aux coûts CFD.** L'ORB 5 minutes de Zarattini & Aziz est négative ici,
  ce que confirme une réplication indépendante sur CFD indices (gain brut reproduit, net nul). La « Noise Area »
  (Sharpe 1,33 publié sur SPY, 2007-2024) est neutre sur 2021-2024 : seul US100 y est positif (+0,07 R).
- **Les rares signaux presque positifs sont sur les indices**, surtout US100 et US500 : S26, S2, S10, S11, S25.
  Mais FTMO ne fournit que 5 ans de minutes sur les indices (2021-2026), ce qui limite fortement la puissance des tests.
- **Données BTCUSD** : avant 2022, la minute du rollover FTMO contient des mèches aberrantes (jusqu'à −20 %).
  Toute stratégie BTC qui traverse ce moment doit être testée sur 2022+ uniquement.

## Prochaine étape proposée : un historique indices plus long

Un historique minute 2013-2021 des indices (Dukascopy, gratuit) donnerait environ 8 ans de données **jamais vues**
pour les stratégies indices de la famille. Ce serait un test indépendant, comparable à une seconde période finale
située dans le passé. Le téléchargement est bloqué depuis les environnements de Claude ; le fichier
`telecharger_dukascopy_indices.bat` (dossier `test_claude`) le fait depuis Windows (Node.js requis).
Avant ce test, un protocole devra figer la liste des stratégies indices évaluées (toutes, pas seulement les meilleures),
pour ne pas choisir après coup.
