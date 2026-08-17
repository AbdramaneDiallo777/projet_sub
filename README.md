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
- tester des seuils de décision plus adaptés au contexte métier,
- améliorer la gestion du déséquilibre avec d’autres méthodes,
- intégrer un monitoring des performances en production,
- ajouter une interface front-end simple pour démonstration.

---

## Auteur

Projet réalisé dans le cadre d’une étude sur le scoring de crédit et l’intelligence artificielle responsable.

---

## Licence

Ce projet est fourni à titre éducatif et de démonstration. L’usage en production doit être validé par une équipe métier et conforme à la réglementation sur le traitement des données personnelles.

