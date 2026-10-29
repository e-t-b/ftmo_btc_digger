# Test en temps réel d'une IA de lecture macro sur EURUSD — protocole pré-enregistré

Enregistré le 2026-10-29 à 23:05 (Paris), avant le premier avis. Toute modification ultérieure de ce fichier est visible
dans l'historique Git et invalide la partie du test qui la suit.

## Objet

Mesurer si un agent Claude, qui lit chaque matin l'actualité macroéconomique et le contexte de marché, rend un avis
directionnel sur EURUSD meilleur que le hasard **après coûts**. Le test se fait en temps réel parce qu'un modèle de langage
ne peut pas être évalué honnêtement sur un historique qu'il a vu pendant son entraînement.

## Mode

**Papier uniquement.** Aucun ordre n'est envoyé, aucun compte n'est connecté. L'avis est seulement consigné.

## Avis quotidien

- Fréquence : un avis par jour ouvré (lundi à vendredi), produit par une tâche planifiée vers 08:21 heure de Paris,
  avant l'ouverture de Londres.
- Fichier : `forward_llm/decisions/AAAA-MM-JJ.json`, poussé dans le dépôt pendant l'exécution. L'horodatage du commit fait
  foi ; un avis poussé après 09:00 heure de Paris est exclu de la notation.
- Contenu :
  - `decision_time_utc` : heure UTC à laquelle l'avis est figé ;
  - `direction` : `long`, `short` ou `none` ;
  - `confidence` : entier de 0 à 100 ;
  - `rationale` : quelques phrases ;
  - `sources` : liens consultés ;
  - `events_today` : publications importantes prévues dans la journée.
- L'agent **ne lit pas** les avis précédents ni leurs résultats : on mesure une procédure fixe, pas un apprentissage en ligne.
- Les jours sans cotation complète (jours fériés FTMO) sont ignorés.

## Notation (identique au moteur de recherche, `eurusd_rl/labels.py`)

- Données : exports MT5 EURUSD M1 du compte FTMO-Demo, fournis périodiquement.
- Entrée : ouverture de la première barre M1 qui commence au moins 5 minutes après `decision_time_utc` et au plus tôt
  à 02:00 heure de New York ; achat à l'ask (bid + spread de coût + 0,2 pip), vente au bid − 0,2 pip.
- Stop : 1 ATR(14) M15 calculé sur les barres closes avant l'entrée, au moins 3 pips et 3 fois le coût aller-retour.
- Cible : 2R. Stop prioritaire si stop et cible sont touchés dans la même barre.
- Sortie forcée : 16:45 heure de New York.
- Coûts : commission 3,50 $/lot/côté, glissements 0,2 pip (0,5 pip en plus au stop), spread de coût de `costs.py`.
- Résultat : R net par avis directionnel ; `none` = 0 R (compté à part).

## Références

Sur les mêmes jours et avec les mêmes règles : direction au hasard (moyenne de toutes les directions possibles),
toujours acheter, toujours vendre.

## Critères (fixés maintenant)

- **Point d'étape à 60 avis directionnels** :
  - continuer si le R net moyen est > 0 et meilleur que les trois références ;
  - arrêter si le R net moyen est ≤ −0,1 (aucune chance réaliste de démontrer un avantage).
- **Décision finale à 250 avis directionnels** (≈ 1 an) :
  - avantage démontré seulement si R net moyen ≥ +0,10, t ≥ 3, et résultats positifs sur au moins 3 des 4 trimestres ;
  - la confiance annoncée doit être croissante avec le résultat (R moyen plus élevé pour confiance ≥ 60 que pour < 60).
- Aucun résultat intermédiaire ne justifie de passer en ordres réels ou démo sans nouvelle autorisation explicite.
