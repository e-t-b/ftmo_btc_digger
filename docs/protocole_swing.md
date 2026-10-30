# Protocole pré-enregistré — Étude swing (positions sur plusieurs jours, compte FTMO Swing)

Rédigé et commité le 2026-10-30 **avant** tout calcul de performance sur les données D1. Seul un test mécanique
du moteur (EURUSD, T1, trades antérieurs à 2006, sans lecture de la performance) a été exécuté pour vérifier le code.
Paramètres machine : `docs/protocole_swing.json`.

## Question

Le passage au swing (FTMO Swing : positions de nuit et de week-end autorisées, levier 1:30) donne-t-il une
stratégie qui passe le challenge 2-Step avec une probabilité utile, là où l'intraday a échoué ?

## Données

Historique D1 FTMO MT5 (`export_d1.py`), un fichier par symbole, de la première barre disponible au 2026-10-28.
Univers : forex (12 paires), indices cash (10), crypto (BTC, ETH), pétrole (2), argent, gaz naturel, cuivre.
**Or exclu** (choix antérieur de l'utilisateur). Barres du samedi/dimanche écartées.

- **Développement** : de la première barre au **2017-12-31**.
- **Période finale** : **2018-01-01 → 2026-10-28**, lue une seule fois, après commit du choix du développement.
  (La période intraday 2024-07 → 2026-10 déjà consommée pour S26 portait sur d'autres règles et d'autres données.)

## Exécution et coûts

Signal sur la clôture du jour, entrée à l'ouverture du jour suivant ; stop vérifié sur le haut/bas du jour ;
gap au-delà du stop → sortie à l'ouverture. Spread = max(spread de la barre, médiane récente en fraction du prix),
glissement d'un demi-spread à chaque exécution.

Swaps : le portage historique est inconnu (les swaps actuels reflètent les taux de 2026). **Cas de base** : coût
symétrique = moyenne (long, short) des swaps actuels si elle est négative (marge du courtier), appliquée chaque nuit,
triple le jour indiqué par MT5. **Stress** : swaps actuels tels quels et spread × 1,5.

## Stratégies (paramètres standards publiés, aucun réglage)

- **T1** Donchian 55/20 (Turtle 2) : cassure de clôture du plus haut/bas 55 j, stop 2 × ATR20, sortie sur cassure 20 j opposée.
- **T2** Donchian 20/10 (Turtle 1) : idem 20/10.
- **T3** Tendance SMA50/SMA200 + cassure 20 j dans le sens de la tendance, stop suiveur 3 × ATR20, sortie au croisement inverse.
- **M1** RSI(2) indices seulement, longs : clôture > SMA200 et RSI2 < 10, sortie clôture > SMA5 ou après 10 jours,
  stop catastrophe 2,5 × ATR20.
T1-T3 sur tout l'univers ; une position au plus par symbole et par stratégie.

## Portefeuille et risque

Risque par trade fixe (fraction du solde initial), risque ouvert total ≤ 4 % (trades refusés au-delà).
Règles FTMO du projet : risque réduit de moitié sous 96 %, arrêt sous 93 %. Le pire de la journée suppose que
toutes les positions touchent leur extrême du jour en même temps (prudent).

## Développement : sélection

1. Par stratégie : espérance par trade (R net), test t unilatéral, p-value ajoutée à la famille cumulée du projet
   (`reports_v2/famille.json`) ; **Holm à 5 % sur toute la famille**. Sont aussi publiés : stress, blocs de 2 ans
   positifs, détail par symbole.
2. Portefeuille des stratégies retenues ; risque par trade choisi dans {0,25 %, 0,5 %, 0,75 %, 1 %} : plus forte
   réussite Monte Carlo FTMO (10 000 trajectoires, blocs de 5 jours, horizon 500 jours ouvrés) parmi les risques
   dont le drawdown intrajournalier de développement reste ≤ 8 %.
3. Si aucune stratégie ne survit à Holm : **pas de test final**, conclusion négative.

## Période finale : critères (tous requis)

1. Espérance > 0 R ; 2. PF ≥ 1,2 ; 3. réussite Monte Carlo FTMO ≥ 40 % (20 000 trajectoires) ;
4. drawdown intrajournalier ≤ 10 %.
Publiés sans effet sur la décision : durée médiane pour réussir, stress coûts/swaps, résultat par stratégie et par année.

**GO** → essai sur le compte gratuit FTMO. **NO-GO** → arrêt de la piste swing sur ces règles.

## Amendement 1 (2026-10-30, avant tout calcul de performance)

Constat sur la disponibilité des données (aucun résultat de stratégie calculé) : les indices cash FTMO ne commencent
qu'en décembre 2017 (US30/AUS200 février 2019, HK50 décembre 2018), donc aucune donnée de développement pour M1
et pour les indices dans T1-T3. Correctifs :
- **Indices en développement : barres D1 Dukascopy** (US500, US100, GER40, US30, UK100, JP225 ; 2013 → 2017,
  `v2/swing_dk.py`), jour serveur FTMO, spread et swaps du symbole FTMO correspondant. Elles ne servent **jamais**
  en période finale (où seuls les symboles FTMO sont utilisés).
- **Commissions FTMO** ajoutées au coût (forex 3,5 $/lot/côté ≈ 3e-5 du prix ; crypto 3,25 pb/côté).
- Plancher de spread : au moins le plancher de `v2/config.py` quand il existe ; crypto ≥ 1e-4 du prix.
- Contrôle qualité : BTC/ETH présentent des journées à très grande amplitude (2011-2013, cohérentes avec la
  volatilité de l'époque) ; conservées, résultat publié par symbole.
- Pétrole US, gaz naturel, indices FTMO : période finale seulement (données trop courtes avant 2018).
