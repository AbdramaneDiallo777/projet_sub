"""
Étape 1 — Exploration & nettoyage
Projet : Scoring de risque explicable et robuste
Dataset : Give Me Some Credit (cs-training.csv)

Comment lancer :
    1. Place cs-training.csv dans le dossier data/
    2. python src/01_exploration.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------
# 0. Chemins robustes (indépendants du dossier depuis lequel on lance)
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = DATA_DIR / "cs-training.csv"

# ----------------------------------------------------------------
# 1. Chargement des données
# ----------------------------------------------------------------
df = pd.read_csv(DATA_PATH, index_col=0)  # la première colonne est un index inutile

print("=" * 60)
print("APERÇU DES DONNÉES")
print("=" * 60)
print(f"\nDimensions : {df.shape[0]} lignes x {df.shape[1]} colonnes\n")
print(df.head())
print("\nTypes de colonnes :\n", df.dtypes)

# ----------------------------------------------------------------
# 2. Valeurs manquantes
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("VALEURS MANQUANTES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"nb_manquants": missing, "%_manquants": missing_pct})
print(missing_report[missing_report["nb_manquants"] > 0])

# On s'attend typiquement à des manquants sur :
#   - MonthlyIncome (~20%)
#   - NumberOfDependents (~2-3%)

# ----------------------------------------------------------------
# 3. Distribution de la variable cible
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("DISTRIBUTION DE LA TARGET (SeriousDlqin2yrs)")
print("=" * 60)
target_counts = df["SeriousDlqin2yrs"].value_counts()
target_pct = df["SeriousDlqin2yrs"].value_counts(normalize=True) * 100
print(target_counts)
print(f"\n% de défauts (classe 1) : {target_pct[1]:.2f}%")
print("→ Dataset déséquilibré : à traiter en étape 2 (SMOTE ou class_weight)")

plt.figure(figsize=(5, 4))
sns.countplot(x="SeriousDlqin2yrs", data=df)
plt.title("Distribution de la target")
plt.savefig(REPORTS_DIR / "target_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------
# 4. Statistiques descriptives des variables numériques
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 60)
print(df.describe().T)

# Points de vigilance connus sur ce dataset :
#   - age : parfois valeur 0 (aberrante, à traiter)
#   - RevolvingUtilizationOfUnsecuredLines : peut dépasser 1 (outliers extrêmes)
#   - Number of Times XX days late : valeurs 96/98 = codes d'erreur, pas des vraies valeurs

# ----------------------------------------------------------------
# 5. Détection rapide d'outliers sur quelques colonnes clés
# ----------------------------------------------------------------
cols_a_verifier = [
    "age",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
]

print("\n" + "=" * 60)
print("VALEURS EXTRÊMES SUR VARIABLES CLÉS")
print("=" * 60)
for col in cols_a_verifier:
    if col in df.columns:
        print(f"\n{col} :")
        print(f"  min={df[col].min()}, max={df[col].max()}, "
              f"médiane={df[col].median()}")

# ----------------------------------------------------------------
# 6. Matrice de corrélation (aperçu rapide)
# ----------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "correlation_matrix.png", dpi=120, bbox_inches="tight")
plt.close()

print("\n" + "=" * 60)
print("Graphiques sauvegardés dans reports/")
print("Prochaine étape : feature engineering (src/02_feature_engineering.py)")
print("=" * 60)
