import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, f1_score,
)
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(DATA_DIR / "train_clean.csv")
test = pd.read_csv(DATA_DIR / "test_clean.csv")

TARGET = "SeriousDlqin2yrs"


cols_a_exclure = [TARGET, "AgeGroup"]

X_train = train.drop(columns=cols_a_exclure)
y_train = train[TARGET]
X_test = test.drop(columns=cols_a_exclure)
y_test = test[TARGET]

print("=" * 60)
print(f"Train : {X_train.shape[0]} lignes, {X_train.shape[1]} features")
print(f"Test  : {X_test.shape[0]} lignes")
print("=" * 60)


print("\n--- Entraînement : Régression logistique ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
log_reg.fit(X_train_scaled, y_train)

proba_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]
pred_log_reg_05 = (proba_log_reg >= 0.5).astype(int)

auc_log_reg = roc_auc_score(y_test, proba_log_reg)
print(f"AUC-ROC : {auc_log_reg:.4f}")
print("\nRapport de classification (seuil 0.5, référence) :")
print(classification_report(y_test, pred_log_reg_05, target_names=["Pas de défaut", "Défaut"]))


print("\n--- Entraînement : XGBoost ---")

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
pred_xgb_05 = (proba_xgb >= 0.5).astype(int)

auc_xgb = roc_auc_score(y_test, proba_xgb)
print(f"AUC-ROC : {auc_xgb:.4f}")
print("\nRapport de classification (seuil 0.5, référence) :")
print(classification_report(y_test, pred_xgb_05, target_names=["Pas de défaut", "Défaut"]))


print("\n--- Recherche du seuil de décision optimal (F1 sur la classe défaut) ---")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba_train_cv = cross_val_predict(
    XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="auc", random_state=42,
    ),
    X_train, y_train, cv=cv, method="predict_proba",
)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_train, proba_train_cv)
f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
best_idx = np.argmax(f1_scores[:-1])  # thresholds a une valeur de moins que precisions/recalls
best_threshold = float(thresholds[best_idx])

print(f"Seuil optimal trouvé (validation croisée train) : {best_threshold:.3f}")
print(f"  (F1 estimé à ce seuil : {f1_scores[best_idx]:.3f}, vs seuil 0.5 par défaut)")

pred_xgb_opt = (proba_xgb >= best_threshold).astype(int)
print("\nRapport de classification XGBoost (seuil optimisé, évalué sur le test) :")
print(classification_report(y_test, pred_xgb_opt, target_names=["Pas de défaut", "Défaut"]))

with open(REPORTS_DIR / "decision_threshold.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "threshold": best_threshold,
            "method": "F1-optimal via validation croisée 5-fold sur le train, "
                      "modèle XGBoost",
        },
        f, indent=2, ensure_ascii=False,
    )
print("Seuil sauvegardé : reports/decision_threshold.json (repris par l'API et l'étape 5)")


print("\n" + "=" * 60)
print("COMPARAISON DES MODÈLES (AUC-ROC)")
print("=" * 60)
print(f"{'Modèle':<25} {'AUC-ROC':<10}")
print(f"{'Régression logistique':<25} {auc_log_reg:.4f}")
print(f"{'XGBoost':<25} {auc_xgb:.4f}")


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
plt.savefig(REPORTS_DIR / "roc_curves.png", dpi=120, bbox_inches="tight")
plt.close()


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(
    confusion_matrix(y_test, pred_log_reg_05),
    display_labels=["Pas de défaut", "Défaut"],
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Régression logistique (seuil 0.5)")

ConfusionMatrixDisplay(
    confusion_matrix(y_test, pred_xgb_opt),
    display_labels=["Pas de défaut", "Défaut"],
).plot(ax=axes[1], cmap="Blues", colorbar=False)
axes[1].set_title(f"XGBoost (seuil optimisé={best_threshold:.2f})")

plt.tight_layout()
plt.savefig(REPORTS_DIR / "confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.close()


joblib.dump(log_reg, REPORTS_DIR / "model_logreg.pkl")
joblib.dump(scaler, REPORTS_DIR / "scaler.pkl")
joblib.dump(xgb, REPORTS_DIR / "model_xgb.pkl")

print("\n" + "=" * 60)
print("Modèles sauvegardés dans reports/ (model_logreg.pkl, model_xgb.pkl)")
print("Graphiques sauvegardés : roc_curves.png, confusion_matrices.png")
print("Prochaine étape : explicabilité SHAP (src/04_explicabilite.py)")
print("=" * 60)
