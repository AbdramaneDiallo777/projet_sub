# Scoring de risque crédit avec explicabilité et équité

Ce projet est un **prototype pédagogique** de scoring de crédit. Il estime le risque de défaut à partir du dataset *Give Me Some Credit*, explique les prédictions avec SHAP et étudie les écarts de décision entre tranches d’âge avec Fairlearn. Il n’est pas destiné à prendre automatiquement des décisions de crédit en production.

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
├── reports/                  # Rapports et artefacts générés localement
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   ├── shap_importance.png
│   ├── fairness_by_age.png
│   ├── cleaning_params.json
│   ├── decision_threshold.json
│   ├── fairness_metrics_table.csv
│   └── *.pkl                 # Modèles/mitigator générés, non versionnés
├── src/                      # Scripts Python organisés par étape
│   ├── 01_exploration.py
│   ├── 02_feature_engineering.py
│   ├── 03_modelisation.py
│   ├── 04_explicabilite.py
│   ├── 05_equite.py
│   ├── api.py
│   └── preprocessing.py
├── tests/                    # Tests du preprocessing et de l'API
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

- Python 3.11 ou plus,
- pip installé,
- le fichier `data/cs-training.csv` (source : dataset Kaggle *Give Me Some Credit*).

---

## Installation

Depuis le dossier `scoring_credit`, sous Windows/PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sous macOS/Linux, activez l’environnement avec `source .venv/bin/activate` après sa création, puis installez les dépendances avec `pip install -r requirements.txt`.

---

## Exécution complète

Lancez les commandes depuis le dossier `scoring_credit`. Les étapes doivent être exécutées dans l’ordre : les scripts réutilisent les données et artefacts produits par les étapes précédentes.

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

### Étape 6 : lancer l’API

L’API a besoin des artefacts créés par les étapes 2 et 3 (`train_clean.csv`, `test_clean.csv`, `cleaning_params.json`, `model_xgb.pkl` et `decision_threshold.json`). L’étape 5 génère en plus `threshold_optimizer.pkl`, utilisé pour la décision corrigée par groupe d’âge ; sans cet artefact, l’API revient au seuil global optimisé.

```bash
uvicorn src.api:app --reload
```

La documentation interactive est disponible à l’adresse [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Tester l’API

`GET /` vérifie l’état du service. `POST /predict` attend les champs suivants ; les noms de variables contenant des tirets doivent être conservés tels quels :

```json
{
   "RevolvingUtilizationOfUnsecuredLines": 0.3,
   "age": 45,
   "NumberOfTime30-59DaysPastDueNotWorse": 0,
   "DebtRatio": 0.4,
   "MonthlyIncome": 5000,
   "NumberOfOpenCreditLinesAndLoans": 8,
   "NumberOfTimes90DaysLate": 0,
   "NumberRealEstateLoansOrLines": 1,
   "NumberOfTime60-89DaysPastDueNotWorse": 0,
   "NumberOfDependents": 1
}
```

La réponse fournit la probabilité brute du modèle XGBoost, la décision, la méthode de décision utilisée, la tranche d’âge et les cinq contributions SHAP principales. La mitigation Fairlearn ne modifie ni la probabilité ni son explication ; elle ajuste uniquement la décision finale. Les valeurs SHAP sont celles du modèle brut et ne constituent pas, à elles seules, une justification réglementaire.

### Tests automatisés

```bash
pytest tests/
```

Les tests de preprocessing peuvent être exécutés indépendamment. Les tests de l’API sont ignorés tant que les artefacts du modèle n’ont pas été générés par le pipeline.

---

## Résultats attendus

Les scripts produisent des fichiers dans le dossier [reports](reports) :

- cartes de performance des modèles,
- graphiques SHAP,
- tableaux d’importance des variables,
- métriques d’équité,
- modèles et mitigateurs sérialisés (`.pkl`). Ces artefacts sont générés localement et exclus de Git ; après un clone, relancez les étapes du pipeline avant de démarrer l’API.

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

---

## Points forts du projet

- analyse complète du dataset,
- pipeline de nettoyage et de préparation,
- comparaison de plusieurs modèles,
- interprétation de la prédiction avec SHAP,
- analyse d’équité sur les tranches d’âge,
- API de démonstration documentée avec FastAPI/Swagger.

---

## Limites / pistes d’amélioration

- explorer d’autres modèles (LightGBM, CatBoost, stacking),
- intégrer un monitoring des performances en production,
- ajouter une interface front-end simple pour démonstration,
- objectiver le compromis équité/performance avec l'équipe métier avant
  toute mise en production (voir section corrections ci-dessous).

---

## Corrections apportées suite à revue de code

Les choix méthodologiques importants sont documentés ci-dessous. Les
chiffres de performance et d’équité dépendent des artefacts et doivent
être recalculés après tout changement de données ou de dépendances.

1. **Fuite de données (data leakage) corrigée.** Auparavant, l'imputation
   des valeurs manquantes et le capping des outliers étaient calculés sur
   tout le dataset avant le split train/test. Désormais, le split est
   fait en premier, et toutes les statistiques (médianes, seuils de
   percentile) sont calculées uniquement sur le train (voir
   `src/preprocessing.py`, fonctions `fit_cleaning_params` /
   `apply_cleaning`), puis appliquées telles quelles au test et à l'API.

2. **Mitigation expérimentale de l'inéquité par âge**, et non plus simple constat.
   `src/05_equite.py` applique désormais `fairlearn.postprocessing
   .ThresholdOptimizer` (contrainte `equalized_odds`, objectif
   `balanced_accuracy_score`) sur le modèle déjà entraîné. Les métriques
   avant/après sont enregistrées dans les rapports ; la mitigation peut
   réduire certains écarts au prix d'une baisse de performance. Elle ne
   garantit ni l'absence de discrimination ni la conformité réglementaire,
   et son adéquation doit être évaluée avec le contexte métier.

3. **Seuil de décision optimisé** au lieu du seuil arbitraire de 0.5 :
   recherché par validation croisée sur le train (maximisation du
   F1-score sur la classe "défaut"), sauvegardé dans
   `reports/decision_threshold.json`.

4. **Versions des dépendances figées** dans `requirements.txt` pour
   stabiliser l’environnement. Le déséquilibre des classes est géré via
   `class_weight` / `scale_pos_weight`.

5. **Artefacts générés exclus de Git** : les modèles sérialisés ne sont
   pas inclus dans le dépôt ; le pipeline doit être relancé localement
   pour les recréer.

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

La robustesse aux perturbations reste une limite à surveiller. Les
résultats du test sont calculés par le script d’équité et sauvegardés
dans les rapports ; ils ne doivent pas être considérés comme une garantie
de stabilité en production.

---

## Auteur

Projet réalisé dans le cadre d’une étude sur le scoring de crédit et l’intelligence artificielle responsable.

---

## Licence

Ce projet est fourni à titre éducatif et de démonstration. L’usage en production doit être validé par une équipe métier et conforme à la réglementation sur le traitement des données personnelles.

