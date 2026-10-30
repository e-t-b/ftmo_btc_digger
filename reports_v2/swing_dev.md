# Étude swing — résultat du développement (protocole docs/protocole_swing.md, amendements 1-2)

Développement : forex 2000-2017, crypto 2011/2016-2017, argent 2008-2017, indices Dukascopy 2013-2017.

| Stratégie | Trades | R net | t | p | Stress | Hors crypto (R) |
|---|---|---|---|---|---|---|
| T1 Donchian 55/20 | 1043 | +0,33 | 1,35 | 0,089 | +0,28 | **−0,10** |
| T2 Donchian 20/10 | 2096 | +0,13 | 0,91 | 0,18 | +0,10 | **−0,12** |
| T3 SMA50/200 + cassure | 1004 | +0,02 | 0,55 | 0,29 | −0,00 | −0,04 |
| M1 RSI(2) indices | 210 | +0,09 | 2,46 | 0,007 | +0,06 | +0,09 |

Holm sur la famille du projet (37 hypothèses) : **aucune stratégie retenue** → pas de test final (règle 3).
(M1 passerait un Holm limité aux 4 hypothèses swing : information seulement.)

Lecture :
- Le suivi de tendance n'est positif que grâce à BTC/ETH 2011-2017 (trades de +50 à +100 R pendant la bulle).
  Sur forex, indices, argent et pétrole, il perd (−0,04 à −0,12 R par trade).
- M1 (RSI2 sur indices US/Japon) montre un petit effet (+0,15 R sur US500/US100/US30/JP225), cohérent avec la
  littérature, mais trop rare et trop faible pour le challenge : à 1 % de risque, +2,8 %/an, réussite Monte Carlo
  26 % en 500 jours ouvrés, durée médiane ~2 ans (information, développement).
