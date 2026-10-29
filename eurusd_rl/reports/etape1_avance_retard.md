# Rapport — Étape 1, partie 2a : avance-retard (GBPUSD, US500, GER40 → EURUSD), 2026-10-29

## 1. Ce qui a été testé

24 tests pré-enregistrés (`leadlag.py`, docstring daté avant mesure). Question posée : le mouvement récent d'un autre instrument annonce-t-il le prochain mouvement d'EURUSD ?

- **Horizons** : 1, 5, 15 et 60 minutes.
- **Deux familles** :
  - L1 : tous les instants ;
  - L2 : seulement après un grand mouvement de l'instrument (≥ 2 écarts-types passés).
- **Mesure** : rendement d'EURUSD sur l'horizon suivant, signé par le sens du mouvement de l'instrument.

Critère « exploitable » : |moyenne| ≥ 2 pips, |t| ≥ 3,1 (Bonferroni à 5 % sur 24 tests), signe stable ≥ 70 % des années.

## 2. Données et périodes

Exports FTMO-Demo, prix BID M1 (hash vérifiés). GBPUSD depuis 2015, US500.cash et GER40.cash depuis fin 2017.
Mesure sur l'entraînement (→ 2025-06), signe vérifié sur la calibration (2025-07 → 2025-12). Heures de rollover exclues ; 2026 non lu.

## 3. Résultats chiffrés

**Aucun effet exploitable.** Effets statistiquement significatifs (|t| ≥ 3,1) :

| Effet | Moyenne par occurrence | t | Années même signe | Calibration |
|---|---|---|---|---|
| GBPUSD 1 min → EURUSD 1 min | +0,005 pip | +6,0 | 91 % | −0,003 (t −1,1) |
| GER40 1 min → EURUSD 1 min | +0,009 pip | +6,2 | 60 % | −0,001 (t −0,3) |
| GBPUSD 5 min → EURUSD 5 min | −0,019 pip | −4,6 | 82 % | −0,003 (t −0,2) |

Les plus grands effets moyens restent inférieurs au pip :
- GBPUSD après grand mouvement, 60 min : +0,80 p (t 2,6, calibration de signe opposé) ;
- US500 sur 60 min : −0,23 p (t −2,6).

Détail : `leadlag_eurusd.json`.

## 4. Comparaison au champion

Sans objet.

## 5. Diagnostic

Les liens entre marchés sont réels mais minuscules : de l'ordre du centième de pip, soit environ 100 fois moins que le coût aller-retour (~1,3 pip). Ils ne se confirment pas sur la calibration. L'information croisée est intégrée dans le prix d'EURUSD en moins d'une minute : c'est le domaine des acteurs à très basse latence, pas d'un bot MT5 de détail.

## 6. Décision

Avance-retard : **aucune piste exploitable**. Rien à transmettre aux étapes suivantes.

## 7. Suite

Dernière piste de la carte : les annonces macroéconomiques (calendrier en attente).

## 8. Test protégé

Non consulté.
