# Rapport — Étapes B2 (intraday H1) et B3 (swing H4), stratégies seules, 2026-10-29

## 1. Ce qui a été testé

Les mêmes logiques que l'étape B (bibliothèque `B2-2026-10-29`, sorties M5 identiques à B1), avec des unités de temps relevées :

| | B2 — intraday | B3 — swing |
|---|---|---|
| Barre d'entrée / structure / tendance / niveaux | H1 / H4 / D1 / veille | H4 / D1 / W1 / semaine précédente |
| Émission du signal | à la clôture H1 | au plus 60 min après la clôture H4 |
| Détention | jusqu'à 16:45 NY | plusieurs jours, week-ends compris, swaps FTMO (−9,74 $/lot/nuit à l'achat, +0,42 $ à la vente, triple le mercredi) |
| Durée maximale | — | 10 jours **calendaires** (voir écart de protocole) |
| Plafond de stop | 60 pips | 150 pips |
| Stratégies | 11 | 8 (ORB, VWAP et range asiatique exclus) |
| Variantes | TP 4R, TP 2R → 22 essais | TP 4R, TP 2R → 16 essais |

Nouveaux tests (33 au total, tous verts) :
- absence de look-ahead aux échelles M5, H1 et H4 ;
- émission unique et péremption des signaux ;
- swaps ordinaires, triple du mercredi et crédit à la vente ;
- gap de week-end à travers le stop.

## 2. Données et périodes

Entraînement 2015-01-05 → 2025-06-30 ; comptes mensuels (B2) ou trimestriels (B3).
La calibration n'a pas été consultée, puisqu'aucune variante n'a passé la porte. Validation et test protégé : non utilisés.

## 3. Porte (pré-enregistrée)

| Critère | Seuil |
|---|---|
| Nombre de trades | ≥ 300 (B2) / ≥ 150 (B3) |
| Espérance nette | ≥ +0,05 R |
| t-stat | ≥ 3,2 (environ 70 essais cumulés depuis l'étape B) |
| Années positives | ≥ 65 % |
| Coûts stressés | espérance positive |

## 4. Résultats chiffrés (entraînement, TP 4R)

### B2 — intraday H1 : 0 variante sur 22 ne passe

| Stratégie | Trades | Net R | t | Brut R (t) | Stop médian |
|---|---|---|---|---|---|
| range_meanrev | 346 | −0,001 | −0,02 | +0,063 (+1,1) | 22,7 p |
| mtf_momentum | 671 | −0,002 | −0,04 | +0,049 (+1,1) | 28,6 p |
| squeeze_breakout | 286 | −0,038 | −0,85 | −0,005 (−0,1) | 42,1 p |
| break_retest | 884 | −0,050 | −1,41 | +0,015 (+0,4) | 26,5 p |
| liquidity_sweep | 1 962 | −0,069 | −2,22 | +0,019 (+0,6) | 18,7 p |
| fib_confluence | 337 | −0,076 | −1,30 | −0,013 (−0,2) | 34,0 p |
| vwap_reclaim | 2 192 | −0,082 | −2,95 | −0,009 (−0,3) | 23,4 p |
| fvg_reaction | 1 008 | −0,107 | −2,38 | +0,002 (+0,1) | 18,8 p |
| orb | 1 690 | −0,113 | −2,90 | −0,016 (−0,4) | 15,4 p |
| trend_pullback | 2 435 | −0,120 | −4,71 | −0,053 (−2,1) | 24,8 p |
| session_level_reject | 1 278 | −0,160 | −3,91 | −0,043 (−1,0) | 15,0 p |

### B3 — swing H4 : 0 variante sur 16 ne passe

| Stratégie | Trades | Net R | t | Brut R (t) | Stop médian | Durée médiane |
|---|---|---|---|---|---|---|
| liquidity_sweep (niveaux hebdo) | 401 | **+0,088** | +0,98 | +0,166 (+1,8) | 38,7 p | 1 j |
| mtf_momentum | 152 | +0,012 | +0,11 | +0,059 (+0,5) | 79,1 p | 5 j |
| fib_confluence | 84 | −0,035 | −0,24 | +0,010 (+0,1) | 59,2 p | 4 j |
| squeeze_breakout | 79 | −0,142 | −1,51 | −0,102 (−1,1) | 103,2 p | 8 j |
| trend_pullback | 397 | −0,143 | −2,05 | −0,086 (−1,2) | 65,2 p | 3 j |
| fvg_reaction | 161 | −0,171 | −1,33 | −0,107 (−0,8) | 37,9 p | 1 j |
| break_retest (niveaux hebdo) | 216 | −0,278 | −3,14 | −0,227 (−2,5) | 64,8 p | 3 j |
| range_meanrev | 86 | −0,365 | −2,68 | −0,303 (−2,2) | 48,6 p | 2 j |

Variantes TP 2R : aucune ne passe ; même hiérarchie.
Détail complet : `stage_b2_strategies.json` et `stage_b3_strategies.json`.

## 5. Problèmes détectés / diagnostic

1. **Le levier des coûts a fonctionné, mais il n'y avait rien dessous.**
   - En B2, les stops passent de 5-9 pips à 15-42 pips : le coût par trade est à peu près divisé par deux (environ −0,06 R au lieu de −0,2 R).
   - Pourtant l'espérance **brute** reste nulle (de −0,05 à +0,06 R, aucune significative).
2. **B3, le seul net positif (`liquidity_sweep` hebdomadaire, +0,09 R) n'est pas significatif** (t = 1,0).
   - Il est positif de 2015 à 2019 (de +0,05 à +0,57 R par an), mais négatif en 2022, 2024 et 2025 (−0,16 à −0,22 R).
   - Il s'érode donc au lieu de se confirmer.
   - Son pendant, la cassure-retest des niveaux hebdomadaires, est négatif (brut −0,23 R, t −2,5). Cela suggère qu'en swing les extrêmes de la semaine précédente ont tendance à tenir plutôt qu'à céder, mais ce n'est qu'une piste, pas une preuve : sur environ 38 essais, un ou deux |t| > 2 sont attendus par hasard.
3. **B3 manque de puissance** : plusieurs stratégies ont moins de 100 trades en 10,5 ans.
4. **Swaps** bien modélisés : −733 $ sur les 401 trades de `liquidity_sweep` (environ −0,04 R par trade), payés par les achats.
5. **Écart de protocole** : la durée maximale annoncée était de « 10 jours de bourse » ; l'implémentation compte 10 jours **calendaires** (environ 8 jours de bourse, détention observée jusqu'à 12 jours calendaires à cause des week-ends). L'impact est mineur, car la durée médiane est de 1 à 5 jours. À corriger si B3 est poursuivie.

## 6. Décision

Porte non franchie en B2 comme en B3 : **pas de passage au renforcement, pas de champion**.

Au total, sur 71 variantes explicites et trois échelles (M5, H1, H4), aucune n'a montré d'avantage robuste, ni brut ni net, sur EURUSD 2015-2025. S'y ajoute l'agent PPO de l'étape A.

## 7. Suites possibles

1. **Test indépendant d'une seule hypothèse**, à pré-enregistrer : « en swing, les rejets des extrêmes de la semaine précédente gagnent et les cassures de ces extrêmes perdent ».
   - Données : l'export GBPUSD 2015-2026 déjà présent, échantillon indépendant de la sélection faite sur EURUSD.
   - Coût faible ; porte stricte car c'est un seul essai.
   - L'érosion récente sur EURUSD rend un succès peu probable.
2. **Expérience « calendrier économique »** isolée, prévue par le cahier des charges, si l'export est disponible.
3. **Ne pas engager de challenge payant** en l'état : aucune stratégie testée n'a d'espérance positive démontrée.

## 8. Test protégé

Non consulté.
