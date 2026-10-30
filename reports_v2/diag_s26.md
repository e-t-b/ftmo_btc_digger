# Diagnostic de S26 (journée de tendance) — d'où viennent les pertes ?

Données : Dukascopy 2013-2021-08 (841 trades) et FTMO développement 2021-09 → 2024-06 (337 trades), US500, US100, GER40.
Période finale FTMO non utilisée. Reproduire : `python -m v2.diag_s26`.
Règle rappelée : à ouverture + 2 h, si |prix − ouverture| ≥ 0,5 ATR D1 et prix dans les 20 % extrêmes du range,
on suit le mouvement ; stop = ouverture de séance ; sortie à la clôture.

## Où sont les pertes

| Sortie | Trades | R moyen | Total |
|---|---|---|---|
| Clôture (temps) | 1 006 (85 %) | +0,23 | +234 R |
| Stop à l'ouverture | 166 (14 %) | −1,02 | −170 R |
| Gap au-delà du stop | 6 | −1,03 | −6 R |

## Ce que S26 fait mal (effets présents sur les deux sources)

1. **Le stop est trop loin.** Il est à l'ouverture, donc 1 R = tout le mouvement du matin (0,5 à 1+ ATR).
   Les trades gagnants ne reculent presque jamais autant (recul maximal médian 0,30 R, 75e centile 0,53 R).
   Le risque engagé est donc surdimensionné par rapport au mouvement réellement en jeu : l'espérance par R est diluée.
2. **Les pertes sont des trades qui n'ont jamais fonctionné**, pas des gains rendus : les trades stoppés ont atteint
   au mieux +0,11 R (médiane). Moins de 1 % des perdants étaient passés par +1 R.
3. **Les entrées en sur-extension perdent** : mouvement > 1 ATR au moment de l'entrée → −0,013 R (Dukascopy), −0,067 R (FTMO).
4. **Les journées calmes ne paient pas** : ATR < 0,8 % du prix → +0,015 R (Dukascopy), −0,115 R (FTMO).
5. **Le gain vient en fin de séance** : +0,01 R après 2 h de position, +0,04 à +0,07 R au bout de 4 h.

Non retenus (incohérents entre les deux sources) : gap d'ouverture, tendance D1, sens long/short, jour de la semaine.

## Effet estimé des corrections (approximation à partir du recul maximal)

| Stop | R moyen Dukascopy | R moyen FTMO | Gagnants |
|---|---|---|---|
| Ouverture (actuel, 1 R) | +0,047 | +0,066 | 54 % |
| À mi-chemin (0,5 R) | +0,105 | +0,105 | 49 % |

Filtre « mouvement ≤ 1 ATR et ATR ≥ 0,8 % du prix » : +0,059 R (Dukascopy) et +0,094 R (FTMO), au lieu de +0,044 et +0,064.

Attention : resserrer le stop n'augmente pas l'avantage en points. Il double le gain **par unité de risque**,
ce qui compte pour FTMO puisque le risque par trade est fixé. La significativité statistique reste la même.
