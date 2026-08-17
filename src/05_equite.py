"""
Étape 5 — Équité & robustesse
Projet : Scoring de risque explicable et robuste

Analyse si le modèle XGBoost traite équitablement les différentes tranches
d'âge (variable identifiée comme sensible à l'étape 4 - SHAP), et teste
sa robustesse face à de petites perturbations des données.

Comment lancer :
    python src/05_equite.py
"""

import sys
import types

# Même contournement qu'à l'étape 4, au cas où Fairlearn dépendrait
# aussi de numba indirectement sur ce PC.
if "numba" not in sys.modules:
    try:
        import numba  # noqa: F401
        import numba.typed  # noqa: F401
    except ImportError:
        def _identity_decorator(*args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

        fake_numba = types.ModuleType("numba")
        fake_numba.njit = _identity_decorator
        fake_numba.jit = _identity_decorator
        fake_numba.vectorize = _identity_decorator
        fake_numba.prange = range

        fake_numba_typed = types.ModuleType("numba.typed")
        fake_numba_typed.List = list
        fake_numba_typed.Dict = dict
        fake_numba.typed = fake_numba_typed

        sys.modules["numba"] = fake_numba
        sys.modules["numba.typed"] = fake_numba_typed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import recall_score, precision_score, accuracy_score

# ----------------------------------------------------------------
# 1. Chargement du modèle et des données
# ----------------------------------------------------------------
xgb = joblib.load("reports/model_xgb.pkl")
test = pd.read_csv("data/test_clean.csv")

TARGET = "SeriousDlqin2yrs"
SENSITIVE_COL = "AgeGroup"  # identifiée comme facteur important par SHAP

X_test = test.drop(columns=[TARGET, SENSITIVE_COL])
y_test = test[TARGET]
sensitive_features = test[SENSITIVE_COL]

y_pred = xgb.predict(X_test)

print("=" * 60)
print("ANALYSE D'ÉQUITÉ PAR TRANCHE D'ÂGE")
print("=" * 60)
print(f"\nTranches d'âge présentes : {sensitive_features.unique().tolist()}")

# ----------------------------------------------------------------
# 2. Métriques par tranche d'âge
# ----------------------------------------------------------------
metrics = {
    "selection_rate": selection_rate,   # % de clients classés "à risque"
    "recall": recall_score,             # % de vrais défauts détectés
    "precision": precision_score,       # fiabilité des alertes
    "accuracy": accuracy_score,
}

metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_features,
)

print("\n--- Métriques par groupe d'âge ---")
print(metric_frame.by_group.round(4))

print("\n--- Écart max entre groupes (worst case) ---")
print(metric_frame.difference().round(4))

# ----------------------------------------------------------------
# 3. Indicateurs d'équité globaux (Fairlearn)
# ----------------------------------------------------------------
# Demographic parity : les groupes doivent avoir un taux similaire de
# clients classés "à risque", indépendamment de leur taux de défaut réel
dp_diff = demographic_parity_difference(
    y_test, y_pred, sensitive_features=sensitive_features
)

# Equalized odds : les groupes doivent avoir un taux similaire de vrais
# positifs ET de faux positifs (plus strict, tient compte du vrai risque)
eo_diff = equalized_odds_difference(
    y_test, y_pred, sensitive_features=sensitive_features
)

print("\n" + "=" * 60)
print("INDICATEURS D'ÉQUITÉ GLOBAUX")
print("=" * 60)
print(f"Demographic parity difference : {dp_diff:.4f}")
print("  (0 = parfaitement équitable ; usuellement on tolère < 0.1)")
print(f"Equalized odds difference     : {eo_diff:.4f}")
print("  (0 = parfaitement équitable ; usuellement on tolère < 0.1)")

if dp_diff > 0.1 or eo_diff > 0.1:
    print("\n⚠️  Écart notable détecté entre groupes d'âge -> à discuter "
          "dans le rapport (le modèle traite différemment certaines "
          "tranches d'âge, ce qui peut poser un problème d'équité).")
else:
    print("\n✅ Écarts dans une fourchette raisonnable entre groupes d'âge.")

# ----------------------------------------------------------------
# 4. Graphique : taux de sélection ("classé à risque") par groupe
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

metric_frame.by_group["selection_rate"].plot(
    kind="bar", ax=axes[0], color="steelblue"
)
axes[0].set_title("Taux de clients classés 'à risque' par tranche d'âge")
axes[0].set_ylabel("Taux de sélection")
axes[0].tick_params(axis="x", rotation=45)

metric_frame.by_group["recall"].plot(
    kind="bar", ax=axes[1], color="indianred"
)
axes[1].set_title("Recall (défauts détectés) par tranche d'âge")
axes[1].set_ylabel("Recall")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("reports/fairness_by_age.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nGraphique sauvegardé : reports/fairness_by_age.png")

metric_frame.by_group.to_csv("reports/fairness_metrics_table.csv")
print("Table sauvegardée : reports/fairness_metrics_table.csv")

# ----------------------------------------------------------------
# 5. Test de robustesse — perturbation des données
# ----------------------------------------------------------------
# On vérifie que le modèle ne change pas radicalement d'avis si on
# perturbe légèrement les variables numériques (+/- 5%). Un modèle
# robuste doit rester globalement stable.
print("\n" + "=" * 60)
print("TEST DE ROBUSTESSE (perturbation +/- 5%)")
print("=" * 60)

np.random.seed(42)
X_test_perturbed = X_test.copy()
numeric_cols = X_test.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    bruit = np.random.uniform(0.95, 1.05, size=len(X_test_perturbed))
    X_test_perturbed[col] = X_test_perturbed[col] * bruit

y_pred_perturbed = xgb.predict(X_test_perturbed)

taux_changement = (y_pred != y_pred_perturbed).mean()
print(f"Taux de prédictions qui changent après perturbation : {taux_changement:.2%}")

if taux_changement < 0.05:
    print("✅ Modèle stable : moins de 5% des décisions changent avec du bruit léger.")
else:
    print("⚠️  Modèle sensible : plus de 5% des décisions changent -> "
          "à mentionner comme limite dans le rapport.")

# ----------------------------------------------------------------
# 6. Résumé pour le rapport
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("RÉSUMÉ POUR LE RAPPORT")
print("=" * 60)
print(f"Demographic parity difference : {dp_diff:.4f}")
print(f"Equalized odds difference     : {eo_diff:.4f}")
print(f"Stabilité face au bruit (+/-5%) : {(1 - taux_changement):.2%} de décisions inchangées")
print("\nFichiers générés : fairness_by_age.png, fairness_metrics_table.csv")
print("Prochaine étape : API FastAPI + rapport final (src/06_api.py)")
print("=" * 60)
