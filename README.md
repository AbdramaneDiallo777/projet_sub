# Scoring de risque crédit avec explicabilité et équité

Ce projet a pour objectif de construire un modèle de scoring de crédit capable de prédire le risque de défaut de paiement, tout en restant explicable et en analysant son comportement vis-à-vis de variables sensibles comme l’âge.

Le projet couvre les étapes suivantes :

- exploration et diagnostic du dataset,
- nettoyage et feature engineering,
- entraînement de plusieurs modèles,
- interprétation des prédictions avec SHAP,
- analyse d’équité avec Fairlearn,
- exposition du modèle via une API FastAPI.

---

## Objectif métier

L’objectif est de développer un système de scoring de risque qui aide à évaluer la probabilité qu’un client ne respecte pas ses obligations de remboursement. En plus de la performance prédictive, le projet met l’accent sur :

- la transparence des décisions,
- l’interprétabilité des variables influentes,
- la robustesse du modèle,
- l’équité entre groupes d’âge.

---

## Problématique

Le dataset utilisé est le célèbre dataset "Give Me Some Credit". Il contient des variables liées au comportement financier d’un client ainsi qu’une cible binaire :

- 1 : défaut de paiement dans les 2 prochaines années,
- 0 : pas de défaut.

Le challenge est double :

1. obtenir un bon score de prédiction malgré un déséquilibre de classes,
2. s’assurer que le modèle reste compréhensible et ne discrimine pas de façon injuste certains groupes.

---

## Structure du dépôt

```text
scoring_credit/
├── data/                     # Données brutes et nettoyées
│   ├── cs-training.csv
│   ├── train_clean.csv
│   └── test_clean.csv
├── reports/                  # Graphiques, modèles et tableaux exportés
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── shap_importance.png
│   ├── fairness_by_age.png
│   ├── model_xgb.pkl
│   └── fairness_metrics_table.csv
├── src/                      # Scripts Python organisés par étape
│   ├── 01_exploration.py
│   ├── 02_feature_engineering.py
│   ├── 03_modelisation.py
│   ├── 04_explicabilite.py
│   ├── 05_equite.py
│   ├── api.py
│   └── __init__.py
├── notebooks/               # Espace pour expérimentations ou analyses complémentaires
├── requirements.txt         # Dépendances Python
├── README.md                # Documentation du projet
└── .gitignore               # Fichiers à ignorer dans Git
```

---

## Pipeline du projet

### 1. Exploration des données
Le script [src/01_exploration.py](src/01_exploration.py) permet de :

- charger le dataset,
- examiner les types de colonnes,
- détecter les valeurs manquantes,
- analyser la distribution de la cible,
- identifier les outliers et les colonnes critiques.

### 2. Feature engineering
Le script [src/02_feature_engineering.py](src/02_feature_engineering.py) traite :

- les âges impossibles,
- les codes d’erreur 96/98,
- les valeurs extrêmes sur les ratios,
- les valeurs manquantes,
- la création de variables pertinentes comme `TotalPastDueIncidents`, `HasPastDueHistory`, `IncomePerDependent` et `AgeGroup`.

### 3. Modélisation
Le script [src/03_modelisation.py](src/03_modelisation.py) entraîne et compare :

- une régression logistique comme baseline,
- un modèle XGBoost plus performant.

Les métriques évaluées incluent :

- AUC-ROC,
- précision,
- rappel,
- matrice de confusion,
- courbes ROC.

### 4. Explicabilité
Le script [src/04_explicabilite.py](src/04_explicabilite.py) utilise SHAP pour :

- mesurer l’importance des variables,
- visualiser les effets globaux,
- expliquer les prédictions pour un client précis.

### 5. Équité
Le script [src/05_equite.py](src/05_equite.py) analyse les écarts de décision selon les tranches d’âge via Fairlearn. Il mesure notamment :

- le taux de sélection par groupe,
- le recall par groupe,
- les écarts de parité démographique,
- la robustesse du modèle face à une légère perturbation des données.

### 6. API REST
Le fichier [src/api.py](src/api.py) expose le modèle via FastAPI. L’API permet de soumettre un profil client et d’obtenir :

- la probabilité de défaut,
- la décision associée,
- les facteurs principaux influençant la prédiction.

---

## Données utilisées

Le projet part du dataset Kaggle :

- Give Me Some Credit

La variable cible est :

- `SeriousDlqin2yrs`

Elle indique si le client a connu un défaut de paiement dans les 2 années suivantes.

Variables clés :

- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfDependents`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfTimes90DaysLate`

---

## Prérequis

Avant de lancer le projet, il faut avoir :

- Python 3.9 ou plus,
- pip installé,
- un environnement virtuel recommandé.

---

## Installation

Dans le dossier du projet :

```bash
pip install -r requirements.txt
```

Sous Windows, dans PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Lancement du projet

### Étape 1 : exploration

```bash
python src/01_exploration.py
```

### Étape 2 : feature engineering

```bash
python src/02_feature_engineering.py
```

### Étape 3 : modélisation

```bash
python src/03_modelisation.py
```

### Étape 4 : explicabilité

```bash
python src/04_explicabilite.py
```

### Étape 5 : équité

```bash
python src/05_equite.py
```

### Étape 6 : API

```bash
uvicorn src.api:app --reload
```

Ensuite, ouvrir :

- http://127.0.0.1:8000/docs

pour tester l’API Swagger.

### Étape 7 (optionnelle) : tests automatisés

```bash
pytest tests/
```

---

## Résultats attendus

Le projet produit des fichiers dans le dossier [reports](reports) :

- cartes de performance des modèles,
- graphiques SHAP,
- tableaux d’importance des variables,
- métriques d’équité,
- modèles entraînés (.pkl).

---

## Stack technique

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Fairlearn
- FastAPI
- Uvicorn
- Imbalanced-learn

---

## Points forts du projet

- analyse complète du dataset,
- pipeline de nettoyage et de préparation,
- comparaison de plusieurs modèles,
- interprétation de la prédiction avec SHAP,
- analyse d’équité sur les tranches d’âge,
- fabrication d’une API exploitable en production.

---

## Limites / pistes d’amélioration

- explorer d’autres modèles (LightGBM, CatBoost, stacking),
- intégrer un monitoring des performances en production,
- ajouter une interface front-end simple pour démonstration,
- objectiver le compromis équité/performance avec l'équipe métier avant
  toute mise en production (voir section corrections ci-dessous).

---

## Corrections apportées suite à revue de code

Une revue a identifié plusieurs problèmes méthodologiques ; voici les
corrections apportées :

1. **Fuite de données (data leakage) corrigée.** Auparavant, l'imputation
   des valeurs manquantes et le capping des outliers étaient calculés sur
   tout le dataset avant le split train/test. Désormais, le split est
   fait en premier, et toutes les statistiques (médianes, seuils de
   percentile) sont calculées uniquement sur le train (voir
   `src/preprocessing.py`, fonctions `fit_cleaning_params` /
   `apply_cleaning`), puis appliquées telles quelles au test et à l'API.

2. **Mitigation de l'inéquité par âge**, et non plus simple constat.
   `src/05_equite.py` applique désormais `fairlearn.postprocessing
   .ThresholdOptimizer` (contrainte `equalized_odds`, objectif
   `balanced_accuracy_score`) sur le modèle déjà entraîné. Résultat sur
   ce dataset : demographic parity difference 0.40 → 0.05, equalized odds
   difference 0.40 → 0.08 (tolérance usuelle < 0.10), au prix d'une
   baisse de recall global (0.78 → 0.73) — un compromis à valider avec
   l'équipe métier avant production. L'API utilise cette version corrigée
   pour la décision finale, tout en gardant la probabilité brute et
   l'explication SHAP du modèle original pour la transparence.

3. **Seuil de décision optimisé** au lieu du seuil arbitraire de 0.5 :
   recherché par validation croisée sur le train (maximisation du
   F1-score sur la classe "défaut"), sauvegardé dans
   `reports/decision_threshold.json`.

4. **Versions des dépendances figées** dans `requirements.txt`
   (`pandas==2.2.3` notamment : une version plus récente de pandas casse
   `fairlearn.postprocessing.ThresholdOptimizer`). `imbalanced-learn` a
   été retiré car jamais utilisé (le déséquilibre est déjà géré via
   `class_weight` / `scale_pos_weight`).

5. **`.gitignore` réel ajouté** (mentionné dans le README initial mais
   absent) : exclut `.venv/`, `__pycache__/`, les CSV volumineux et les
   modèles `.pkl` du suivi git.

6. **Chemins robustes** : tous les scripts utilisent désormais des
   chemins relatifs à leur propre emplacement (`Path(__file__)`) au lieu
   de chemins relatifs au dossier de lancement — ils fonctionnent
   maintenant depuis n'importe quel répertoire de travail.

7. **Logique de preprocessing dédupliquée** entre le pipeline
   d'entraînement et l'API (`src/preprocessing.py` partagé), pour
   garantir que l'API applique exactement le même nettoyage que
   l'entraînement.

8. **Tests automatisés ajoutés** (`tests/`) : tests unitaires du
   preprocessing (dont un test anti-régression sur le data leakage) et
   tests de l'API (`pytest tests/`).

Point non corrigé, documenté comme limite assumée : la robustesse au
bruit reste à la limite du seuil visé (~5% de décisions qui changent
avec une perturbation de ±5% des variables), ce qui n'a pas justifié de
changement de modèle à ce stade.

---

## Auteur

Projet réalisé dans le cadre d’une étude sur le scoring de crédit et l’intelligence artificielle responsable.

---

## Licence

Ce projet est fourni à titre éducatif et de démonstration. L’usage en production doit être validé par une équipe métier et conforme à la réglementation sur le traitement des données personnelles.

