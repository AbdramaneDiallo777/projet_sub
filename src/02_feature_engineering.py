"""
Étape 2 — Feature engineering & nettoyage complet
Projet : Scoring de risque explicable et robuste
Dataset : Give Me Some Credit (cs-training.csv)

Traite dans l'ordre tous les problèmes détectés en étape 1 :
    1. age = 0 (aberrant)
    2. Codes d'erreur 96/98 sur les colonnes "days late"
    3. Outliers extrêmes sur RevolvingUtilizationOfUnsecuredLines et DebtRatio
    4. Valeurs manquantes sur MonthlyIncome et NumberOfDependents
    5. Split train/test

IMPORTANT (correction data leakage) :
    Le split train/test est fait AVANT tout calcul de statistique de
    nettoyage (médianes, seuils de percentile). Ces statistiques sont
    calculées UNIQUEMENT sur le train (fit_cleaning_params), puis
    appliquées telles quelles au test (apply_cleaning). Sans cela, les
    valeurs du test set influencent le nettoyage du train set, ce qui
    biaise optimistement l'évaluation finale du modèle.

Comment lancer (depuis n'importe quel dossier) :
    python src/02_feature_engineering.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import (
    fit_cleaning_params,
    apply_cleaning,
    add_engineered_features,
    save_cleaning_params,
)

pd.set_option("display.width", 120)

# ----------------------------------------------------------------
# 0. Chemins robustes (indépendants du dossier depuis lequel on lance)
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_PATH = DATA_DIR / "cs-training.csv"

# ----------------------------------------------------------------
# 1. Chargement
# ----------------------------------------------------------------
df = pd.read_csv(DATA_PATH, index_col=0)

print("=" * 60)
print(f"Dataset chargé : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print("=" * 60)

# ----------------------------------------------------------------
# 2. Problème n°1 — age = 0
# ----------------------------------------------------------------
print("\n--- Traitement : age = 0 ---")
nb_age_zero = (df["age"] == 0).sum()
print(f"Lignes avec age = 0 : {nb_age_zero}")

# Un âge de 0 est impossible pour un emprunteur -> on supprime ces lignes.
# C'est un simple filtrage (pas une statistique calculée sur les données),
# donc aucun risque de fuite à le faire avant le split.
df = df[df["age"] > 0].copy()
print(f"Lignes restantes après suppression : {df.shape[0]}")

# ----------------------------------------------------------------
# 3. Split train/test — AVANT tout calcul de statistique de nettoyage
# ----------------------------------------------------------------
print("\n--- Split train/test (avant nettoyage, pour éviter le data leakage) ---")

TARGET = "SeriousDlqin2yrs"
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = X_train.assign(**{TARGET: y_train})
test_df = X_test.assign(**{TARGET: y_test})

print(f"Train : {train_df.shape[0]} lignes ({y_train.mean()*100:.2f}% défauts)")
print(f"Test  : {test_df.shape[0]} lignes ({y_test.mean()*100:.2f}% défauts)")

# ----------------------------------------------------------------
# 4. Fit des statistiques de nettoyage — UNIQUEMENT sur le train
# ----------------------------------------------------------------
print("\n--- Calcul des statistiques de nettoyage (sur le train uniquement) ---")
cleaning_params = fit_cleaning_params(train_df)

for col, med in cleaning_params["late_cols_medians"].items():
    nb_erreur = train_df[col].isin([96, 98]).sum()
    print(f"{col} : {nb_erreur} codes d'erreur (train) -> médiane de remplacement = {med}")

for col, seuil in cleaning_params["outlier_caps"].items():
    print(f"{col} : seuil de capping (99.5e percentile, train) = {seuil:.2f}")

print(f"NumberOfDependents : médiane (train) = {cleaning_params['dependents_median']}")
print("MonthlyIncome : médianes par tranche d'âge calculées sur le train")

# ----------------------------------------------------------------
# 5. Application du nettoyage (train ET test, avec les stats du train)
# ----------------------------------------------------------------
print("\n--- Application du nettoyage (train et test) ---")
train_df = apply_cleaning(train_df, cleaning_params)
test_df = apply_cleaning(test_df, cleaning_params)

print(f"Valeurs manquantes restantes (train) : {train_df.isnull().sum().sum()}")
print(f"Valeurs manquantes restantes (test)  : {test_df.isnull().sum().sum()}")

# ----------------------------------------------------------------
# 6. Feature engineering — nouvelles variables (déterministe, sans fuite)
# ----------------------------------------------------------------
print("\n--- Création de nouvelles variables ---")
train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)
print("Nouvelles colonnes : TotalPastDueIncidents, HasPastDueHistory, "
      "IncomePerDependent, AgeGroup")

# ----------------------------------------------------------------
# 7. Sauvegarde des données nettoyées + des paramètres de nettoyage
# ----------------------------------------------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(DATA_DIR / "train_clean.csv", index=False)
test_df.to_csv(DATA_DIR / "test_clean.csv", index=False)

# Les paramètres de nettoyage (médianes, seuils) sont sauvegardés pour être
# réutilisés tels quels par l'API en production (même logique, pas de
# recalcul sur les nouvelles requêtes -> cohérence garantie).
save_cleaning_params(cleaning_params, REPORTS_DIR / "cleaning_params.json")

print("\n" + "=" * 60)
print("Fichiers sauvegardés : data/train_clean.csv, data/test_clean.csv")
print("Paramètres de nettoyage sauvegardés : reports/cleaning_params.json")
print("Prochaine étape : modélisation (src/03_modelisation.py)")
print("=" * 60)
