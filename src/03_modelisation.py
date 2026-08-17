"""
Étape 3 — Modélisation
Projet : Scoring de risque explicable et robuste

Entraîne 2 modèles et les compare :
    1. Régression logistique (baseline, interprétable nativement)
    2. XGBoost (plus performant, sera expliqué via SHAP à l'étape 4)

Comment lancer :
    python src/03_modelisation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

# ----------------------------------------------------------------
# 1. Chargement des données nettoyées (issues de l'étape 2)
# ----------------------------------------------------------------
train = pd.read_csv("data/train_clean.csv")
test = pd.read_csv("data/test_clean.csv")

TARGET = "SeriousDlqin2yrs"

# AgeGroup est une variable catégorielle (texte) -> on la retire pour l'instant.
# On garde 'age' (numérique) qui porte déjà cette information pour les modèles.
# AgeGroup sera réutilisée telle quelle à l'étape 5 (Fairlearn) pour analyser
# l'équité par tranche d'âge.
cols_a_exclure = [TARGET, "AgeGroup"]

X_train = train.drop(columns=cols_a_exclure)
y_train = train[TARGET]
X_test = test.drop(columns=cols_a_exclure)
y_test = test[TARGET]

print("=" * 60)
print(f"Train : {X_train.shape[0]} lignes, {X_train.shape[1]} features")
print(f"Test  : {X_test.shape[0]} lignes")
print("=" * 60)

# ----------------------------------------------------------------
# 2. Modèle 1 — Régression logistique (baseline)
# ----------------------------------------------------------------
print("\n--- Entraînement : Régression logistique ---")

# La régression logistique est sensible à l'échelle des variables
# -> on standardise (moyenne 0, écart-type 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(
    class_weight="balanced",  # compense le déséquilibre 6.68% vs 93.32%
    max_iter=1000,
    random_state=42,
)
log_reg.fit(X_train_scaled, y_train)

proba_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]
pred_log_reg = log_reg.predict(X_test_scaled)

auc_log_reg = roc_auc_score(y_test, proba_log_reg)
print(f"AUC-ROC : {auc_log_reg:.4f}")
print("\nRapport de classification :")
print(classification_report(y_test, pred_log_reg, target_names=["Pas de défaut", "Défaut"]))

# ----------------------------------------------------------------
# 3. Modèle 2 — XGBoost
# ----------------------------------------------------------------
print("\n--- Entraînement : XGBoost ---")

# scale_pos_weight = ratio classe_majoritaire / classe_minoritaire
# compense le déséquilibre différemment que class_weight (spécifique à XGBoost)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight calculé : {scale_pos_weight:.2f}")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42,
)
xgb.fit(X_train, y_train)

proba_xgb = xgb.predict_proba(X_test)[:, 1]
pred_xgb = xgb.predict(X_test)

auc_xgb = roc_auc_score(y_test, proba_xgb)
print(f"AUC-ROC : {auc_xgb:.4f}")
print("\nRapport de classification :")
print(classification_report(y_test, pred_xgb, target_names=["Pas de défaut", "Défaut"]))

# ----------------------------------------------------------------
# 4. Comparaison des 2 modèles
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("COMPARAISON DES MODÈLES")
print("=" * 60)
print(f"{'Modèle':<25} {'AUC-ROC':<10}")
print(f"{'Régression logistique':<25} {auc_log_reg:.4f}")
print(f"{'XGBoost':<25} {auc_xgb:.4f}")

# ----------------------------------------------------------------
# 5. Courbes ROC comparées
# ----------------------------------------------------------------
fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_log_reg)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, proba_xgb)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Régression logistique (AUC={auc_log_reg:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbes ROC — comparaison des modèles")
plt.legend()
plt.tight_layout()
plt.savefig("reports/roc_curves.png", dpi=120, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------
# 6. Matrices de confusion
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, pred_log_reg),
    display_labels=["Pas de défaut", "Défaut"],
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Régression logistique")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, pred_xgb),
    display_labels=["Pas de défaut", "Défaut"],
).plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title("XGBoost")

plt.tight_layout()
plt.savefig("reports/confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------
# 7. Sauvegarde des modèles (réutilisés à l'étape 4 - SHAP)
# ----------------------------------------------------------------
joblib.dump(log_reg, "reports/model_logreg.pkl")
joblib.dump(scaler, "reports/scaler.pkl")
joblib.dump(xgb, "reports/model_xgb.pkl")

print("\n" + "=" * 60)
print("Modèles sauvegardés dans reports/ (model_logreg.pkl, model_xgb.pkl)")
print("Graphiques sauvegardés : roc_curves.png, confusion_matrices.png")
print("Prochaine étape : explicabilité SHAP (src/04_explicabilite.py)")
print("=" * 60)
