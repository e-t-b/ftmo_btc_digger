# Test final unique — S26-B3 sur la période finale FTMO (2024-07 → 2026-10)

Protocole : `docs/protocole_final_S26B3.md` (commit `1492d13`, avant exécution). Résultats : `reports_v2/final_s26b3.json`.

## Décision selon le protocole : NO-GO

| Critère | Seuil | Résultat | |
|---|---|---|---|
| Espérance | > 0 | +0,072 R (403 trades) | OK |
| Profit factor | ≥ 1,1 | 1,15 | OK |
| Réussite Monte Carlo FTMO | ≥ 40 % | 37,9 % | Échec |
| Drawdown maximal | ≤ 10 % | −18,4 % | Échec |

Résultat brut à 0,75 % de risque : +21,5 % en 2,3 ans (≈ +9 %/an), 44 % de trades gagnants.

## Ce que montre la période finale

- **L'avantage est réel** : troisième confirmation indépendante après Dukascopy 2013-2021 et FTMO 2021-2024
  (+0,07 à +0,10 R à chaque fois). S26 d'origine (information) : +0,083 R, t = 2,03, drawdown −4 %.
- **Mais il ne passe pas les limites FTMO** à 0,75 % : une série de pertes au second semestre 2024 creuse un
  drawdown de −18 %, au-delà de la perte maximale de 10 %.
- Information sans effet sur la décision : US100 +0,18 R, GER40 +0,09 R, US500 −0,09 R ; décision à +2 h +0,18 R,
  +1 h +0,07 R, +3 h −0,08 R. Ces chiffres ne peuvent pas servir à modifier les règles : ce serait les ajuster sur
  la seule période qui restait aveugle.
