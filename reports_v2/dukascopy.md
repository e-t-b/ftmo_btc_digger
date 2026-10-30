# Confirmation sur l'historique indépendant Dukascopy (indices, 2013 → 2021-08)

Date : 2026-10-30. Protocole : `docs/protocole_dukascopy.md` (commit `8e9c746`, avant tout test).
Résultats : `reports_v2/dukascopy.json`. Reproduire : `python -m v2.dukascopy`
(données `data/dukascopy/DK_*.npz`, non versionnées, issues de `telecharger_dukascopy_indices.bat`).

## Décision : aucune stratégie confirmée (Holm sur 5, α = 5 %)

| Stratégie | FTMO dév. (R, trades) | Dukascopy (R, trades) | t | p | Seuil Holm | Stress |
|---|---|---|---|---|---|---|
| S26 Journée de tendance | +0,042 (416) | **+0,044 (841)** | 1,73 | 0,041 | 0,010 | +0,024 |
| S2 Volatility breakout | +0,100 (172) | +0,025 (2 485) | 0,88 | 0,189 | 0,013 | — |
| S25 Veille de jour férié | +0,071 (30) | +0,058 (159) | 0,82 | 0,208 | 0,017 | — |
| S11 Tournant de mois | +0,017 (394) | −0,067 (1 176) | −2,19 | 0,986 | 0,025 | — |
| S10 Dérive avant la Fed | +0,042 (46) | −0,297 (131) | −5,58 | 1,000 | 0,050 | — |

## Lecture

- **S26 est le seul signal qui se reproduit** : même espérance sur deux sources indépendantes (+0,042 R en FTMO
  2021-2024, +0,044 R en Dukascopy 2013-2021), positive sur les trois indices et 7 années sur 9.
  Combinées (méthode de Stouffer), les deux preuves donnent environ t ≈ 2,2. C'est probablement un effet réel.
  **Mais il est trop faible pour le challenge** : environ +2 % par an à 0,5 % de risque, réussite Monte Carlo < 1 %.
- **S2** : faible et incohérent (positif sur US100 et GER40, négatif sur US500).
- **S10** : la dérive avant la Fed a disparu (−3 bps en moyenne sur 2013-2021, conforme à la littérature, qui la
  documente surtout avant 2011). **S11** se retourne. Les deux étaient du bruit en développement.
- **Contrôle exploratoire** : les stratégies négatives en FTMO le sont aussi ici (S5, S6, S8, S9, S15, S27, S28 avec
  t < −2,9). Les deux sources concordent : le moteur et les coûts ne biaisent pas les résultats.

## Limites

Spread Dukascopy non fourni : coût = profil FTMO 2022-2024 en fraction du prix (1,1 bp médian sur US500/US100,
0,65 bp sur GER40). Les CFD Dukascopy et FTMO ne sont pas identiques, mais suivent le même sous-jacent.
