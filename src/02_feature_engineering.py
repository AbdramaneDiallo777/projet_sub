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

Comment lancer :
    python src/02_feature_engineering.py
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 120)

# ----------------------------------------------------------------
# 1. Chargement
# ----------------------------------------------------------------
DATA_PATH = "data/cs-training.csv"
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

# Un âge de 0 est impossible pour un emprunteur -> on supprime ces lignes
# (généralement 1 seule ligne sur ce dataset, impact négligeable)
df = df[df["age"] > 0].copy()
print(f"Lignes restantes après suppression : {df.shape[0]}")

# ----------------------------------------------------------------
# 3. Problème n°2 — codes d'erreur 96/98 sur les colonnes "days late"
# ----------------------------------------------------------------
print("\n--- Traitement : codes d'erreur 96/98 ---")
late_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

for col in late_cols:
    nb_erreur = df[col].isin([96, 98]).sum()
    print(f"{col} : {nb_erreur} lignes avec code d'erreur (96 ou 98)")

# Ces valeurs 96/98 sont des artefacts connus du dataset (pas de vrais comptages).
# Elles concernent quasi toujours les mêmes lignes sur les 3 colonnes à la fois.
# Stratégie : on les remplace par la valeur médiane réelle (hors codes d'erreur),
# car supprimer ~3000 lignes perdrait trop de signal, notamment sur la classe
# minoritaire (défauts).
for col in late_cols:
    valeurs_valides = df.loc[~df[col].isin([96, 98]), col]
    mediane = valeurs_valides.median()
    df.loc[df[col].isin([96, 98]), col] = mediane
    print(f"{col} : codes d'erreur remplacés par la médiane ({mediane})")

# ----------------------------------------------------------------
# 4. Problème n°3 — outliers extrêmes (Revolving & DebtRatio)
# ----------------------------------------------------------------
print("\n--- Traitement : outliers extrêmes ---")

# RevolvingUtilizationOfUnsecuredLines est un ratio -> devrait rester <= ~2-3
# On cappe au 99.5e percentile plutôt que de supprimer, pour ne pas perdre
# d'information sur les profils à risque (qui ont souvent des ratios élevés)
for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
    seuil = df[col].quantile(0.995)
    nb_impactes = (df[col] > seuil).sum()
    df[col] = df[col].clip(upper=seuil)
    print(f"{col} : {nb_impactes} valeurs cappées au 99.5e percentile ({seuil:.2f})")

# ----------------------------------------------------------------
# 5. Problème n°4 — valeurs manquantes
# ----------------------------------------------------------------
print("\n--- Traitement : valeurs manquantes ---")

# NumberOfDependents : peu de manquants -> imputation simple par la médiane (0 le plus souvent)
mediane_dependents = df["NumberOfDependents"].median()
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(mediane_dependents)
print(f"NumberOfDependents : imputé par la médiane ({mediane_dependents})")

# MonthlyIncome : ~20% de manquants -> imputation simple par la médiane serait trop
# grossière. On impute par la médiane calculée PAR TRANCHE D'ÂGE, ce qui capture
# une partie de la variabilité réelle des revenus selon l'âge.
df["age_bracket"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 70, 120])
mediane_par_tranche = df.groupby("age_bracket", observed=True)["MonthlyIncome"].transform("median")
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(mediane_par_tranche)
# Sécurité : s'il reste des NaN (tranche sans aucune valeur), fallback médiane globale
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df = df.drop(columns=["age_bracket"])
print("MonthlyIncome : imputé par la médiane par tranche d'âge")

print(f"\nValeurs manquantes restantes : {df.isnull().sum().sum()}")

# ----------------------------------------------------------------
# 6. Feature engineering — nouvelles variables
# ----------------------------------------------------------------
print("\n--- Création de nouvelles variables ---")

# Ratio d'incidents de retard total (agrège les 3 colonnes "days late")
df["TotalPastDueIncidents"] = df[late_cols].sum(axis=1)

# Indicateur binaire : a déjà eu au moins un incident de retard
df["HasPastDueHistory"] = (df["TotalPastDueIncidents"] > 0).astype(int)

# Revenu par personne à charge (évite division par zéro)
df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

# Tranche d'âge (utile aussi plus tard pour l'analyse d'équité avec Fairlearn)
df["AgeGroup"] = pd.cut(
    df["age"], bins=[0, 30, 40, 50, 60, 70, 120],
    labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"]
)

print("Nouvelles colonnes : TotalPastDueIncidents, HasPastDueHistory, "
      "IncomePerDependent, AgeGroup")

# ----------------------------------------------------------------
# 7. Split train/test
# ----------------------------------------------------------------
from sklearn.model_selection import train_test_split

X = df.drop(columns=["SeriousDlqin2yrs"])
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- Split train/test ---")
print(f"Train : {X_train.shape[0]} lignes ({y_train.mean()*100:.2f}% défauts)")
print(f"Test  : {X_test.shape[0]} lignes ({y_test.mean()*100:.2f}% défauts)")

# ----------------------------------------------------------------
# 8. Sauvegarde des données nettoyées
# ----------------------------------------------------------------
X_train.assign(SeriousDlqin2yrs=y_train).to_csv("data/train_clean.csv", index=False)
X_test.assign(SeriousDlqin2yrs=y_test).to_csv("data/test_clean.csv", index=False)

print("\n" + "=" * 60)
print("Fichiers sauvegardés : data/train_clean.csv, data/test_clean.csv")
print("Prochaine étape : modélisation (src/03_modelisation.py)")
print("=" * 60)
