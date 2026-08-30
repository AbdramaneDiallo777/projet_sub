"""
Étape 5 — Équité & robustesse
Projet : Scoring de risque explicable et robuste

Analyse si le modèle XGBoost traite équitablement les différentes tranches
d'âge (variable identifiée comme sensible à l'étape 4 - SHAP), et teste
sa robustesse face à de petites perturbations des données.

AJOUT par rapport à la version initiale :
    Le diagnostic seul ne suffit pas — un écart de demographic parity de
    0.40 est trop important pour être simplement "mentionné dans le
    rapport". On applique donc une MITIGATION via fairlearn.postprocessing
    .ThresholdOptimizer : elle choisit des seuils de décision différents
    par tranche d'âge de façon à satisfaire la contrainte "equalized odds"
    (Hardt et al., 2016), à partir du même modèle XGBoost déjà entraîné
    (pas de ré-entraînement nécessaire). On compare les métriques
    d'équité AVANT / APRÈS mitigation, et l'objet est sauvegardé pour
    être repris par l'API (décision finale équitable, tout en gardant
    la probabilité brute et l'explication SHAP du modèle original).

Comment lancer :
    python src/05_equite.py
"""

import sys
import types
from pathlib import Path

# ------------------------------------------------------------------
# Contournement Windows : sur certains PC, une stratégie de contrôle
# d'application (Smart App Control) bloque le fichier .dll utilisé par
# `numba` pour accélérer certains calculs internes de SHAP/Fairlearn.
# ------------------------------------------------------------------
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

import json
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Warning interne à fairlearn (assignation de dtype dans son code interne,
# sans impact sur les résultats) — voir fairlearn/postprocessing/
# _interpolated_thresholder.py. Filtré pour ne pas polluer la sortie.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="fairlearn.postprocessing.*"
)

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    equalized_odds_difference,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import recall_score, precision_score, accuracy_score

# ----------------------------------------------------------------
# 0. Chemins robustes
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ----------------------------------------------------------------
# 1. Chargement du modèle et des données
# ----------------------------------------------------------------
xgb = joblib.load(REPORTS_DIR / "model_xgb.pkl")
train = pd.read_csv(DATA_DIR / "train_clean.csv")
test = pd.read_csv(DATA_DIR / "test_clean.csv")

TARGET = "SeriousDlqin2yrs"
SENSITIVE_COL = "AgeGroup"  # identifiée comme facteur important par SHAP

X_train = train.drop(columns=[TARGET, SENSITIVE_COL])
y_train = train[TARGET]
sensitive_train = train[SENSITIVE_COL]

X_test = test.drop(columns=[TARGET, SENSITIVE_COL])
y_test = test[TARGET]
sensitive_test = test[SENSITIVE_COL]

y_pred = xgb.predict(X_test)

print("=" * 60)
print("ANALYSE D'ÉQUITÉ PAR TRANCHE D'ÂGE — AVANT MITIGATION")
print("=" * 60)
print(f"\nTranches d'âge présentes : {sensitive_test.unique().tolist()}")

# ----------------------------------------------------------------
# 2. Métriques par tranche d'âge (modèle brut, seuil 0.5)
# ----------------------------------------------------------------
metrics = {
    "selection_rate": selection_rate,
    "recall": recall_score,
    "precision": precision_score,
    "accuracy": accuracy_score,
}

metric_frame = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_test,
)

print("\n--- Métriques par groupe d'âge (avant mitigation) ---")
print(metric_frame.by_group.round(4))
print("\n--- Écart max entre groupes (avant mitigation) ---")
print(metric_frame.difference().round(4))

dp_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_test)
eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_test)

print("\n" + "=" * 60)
print("INDICATEURS D'ÉQUITÉ GLOBAUX — AVANT MITIGATION")
print("=" * 60)
print(f"Demographic parity difference : {dp_diff:.4f}  (tolérance usuelle < 0.1)")
print(f"Equalized odds difference     : {eo_diff:.4f}  (tolérance usuelle < 0.1)")

if dp_diff > 0.1 or eo_diff > 0.1:
    print("\n⚠️  Écart notable détecté entre groupes d'âge -> mitigation appliquée ci-dessous.")
else:
    print("\n✅ Écarts déjà dans une fourchette raisonnable.")

# ----------------------------------------------------------------
# 3. MITIGATION — ThresholdOptimizer (contrainte : equalized odds)
# ----------------------------------------------------------------
# On réutilise le modèle XGBoost déjà entraîné (prefit=True) : on ne le
# ré-entraîne pas, on ajuste seulement les seuils de décision par groupe
# de façon à satisfaire la contrainte d'équité choisie.
print("\n" + "=" * 60)
print("MITIGATION — fairlearn.postprocessing.ThresholdOptimizer")
print("=" * 60)

mitigator = ThresholdOptimizer(
    estimator=xgb,
    constraints="equalized_odds",
    objective="balanced_accuracy_score",  # évite le piège de accuracy_score
    # (par défaut) : sur un dataset déséquilibré (6.7% de défauts),
    # accuracy_score pousse le seuil vers "toujours prédire pas de
    # défaut", ce qui satisfait bien la contrainte d'équité mais rend
    # le modèle inutile (recall proche de 0). balanced_accuracy_score
    # force à considérer les deux classes de façon symétrique.
    predict_method="predict_proba",
    prefit=True,
)
mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)

y_pred_mitigated = mitigator.predict(
    X_test, sensitive_features=sensitive_test, random_state=42
)

metric_frame_mitigated = MetricFrame(
    metrics=metrics, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=sensitive_test,
)

print("\n--- Métriques par groupe d'âge (après mitigation) ---")
print(metric_frame_mitigated.by_group.round(4))
print("\n--- Écart max entre groupes (après mitigation) ---")
print(metric_frame_mitigated.difference().round(4))

dp_diff_mitigated = demographic_parity_difference(
    y_test, y_pred_mitigated, sensitive_features=sensitive_test
)
eo_diff_mitigated = equalized_odds_difference(
    y_test, y_pred_mitigated, sensitive_features=sensitive_test
)

print("\n" + "=" * 60)
print("INDICATEURS D'ÉQUITÉ GLOBAUX — APRÈS MITIGATION")
print("=" * 60)
print(f"Demographic parity difference : {dp_diff_mitigated:.4f}  (avant : {dp_diff:.4f})")
print(f"Equalized odds difference     : {eo_diff_mitigated:.4f}  (avant : {eo_diff:.4f})")

# Impact sur la performance globale (la mitigation a un coût en accuracy
# globale : c'est un arbitrage équité <-> performance à documenter)
acc_avant = accuracy_score(y_test, y_pred)
acc_apres = accuracy_score(y_test, y_pred_mitigated)
recall_avant = recall_score(y_test, y_pred)
recall_apres = recall_score(y_test, y_pred_mitigated)
print(f"\nAccuracy globale  : {acc_avant:.4f} -> {acc_apres:.4f}")
print(f"Recall global     : {recall_avant:.4f} -> {recall_apres:.4f}")
print("(La mitigation dégrade parfois légèrement la performance globale : "
      "c'est le compromis équité/performance à documenter dans le rapport, "
      "et à valider avec l'équipe métier avant mise en production.)")

# Sauvegarde de l'objet de mitigation, repris par l'API pour la décision finale
joblib.dump(mitigator, REPORTS_DIR / "threshold_optimizer.pkl")
print("\nObjet de mitigation sauvegardé : reports/threshold_optimizer.pkl")

# ----------------------------------------------------------------
# 4. Graphiques : avant / après mitigation
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metric_frame.by_group["selection_rate"].plot(kind="bar", ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Taux de sélection par âge — AVANT mitigation")
axes[0, 0].tick_params(axis="x", rotation=45)

metric_frame_mitigated.by_group["selection_rate"].plot(kind="bar", ax=axes[0, 1], color="seagreen")
axes[0, 1].set_title("Taux de sélection par âge — APRÈS mitigation")
axes[0, 1].tick_params(axis="x", rotation=45)

metric_frame.by_group["recall"].plot(kind="bar", ax=axes[1, 0], color="indianred")
axes[1, 0].set_title("Recall par âge — AVANT mitigation")
axes[1, 0].tick_params(axis="x", rotation=45)

metric_frame_mitigated.by_group["recall"].plot(kind="bar", ax=axes[1, 1], color="darkorange")
axes[1, 1].set_title("Recall par âge — APRÈS mitigation")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig(REPORTS_DIR / "fairness_by_age.png", dpi=120, bbox_inches="tight")
plt.close()
print("Graphique sauvegardé : reports/fairness_by_age.png")

# Table comparative avant/après
comparison_table = metric_frame.by_group.copy()
comparison_table.columns = [f"{c}_avant" for c in comparison_table.columns]
for c in metric_frame_mitigated.by_group.columns:
    comparison_table[f"{c}_apres"] = metric_frame_mitigated.by_group[c]
comparison_table.to_csv(REPORTS_DIR / "fairness_metrics_table.csv")
print("Table sauvegardée : reports/fairness_metrics_table.csv")

# ----------------------------------------------------------------
# 5. Test de robustesse — perturbation des données (modèle brut)
# ----------------------------------------------------------------
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
summary = {
    "demographic_parity_difference_avant": dp_diff,
    "demographic_parity_difference_apres": dp_diff_mitigated,
    "equalized_odds_difference_avant": eo_diff,
    "equalized_odds_difference_apres": eo_diff_mitigated,
    "accuracy_avant": acc_avant,
    "accuracy_apres": acc_apres,
    "recall_avant": recall_avant,
    "recall_apres": recall_apres,
    "taux_changement_perturbation": float(taux_changement),
}
with open(REPORTS_DIR / "fairness_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("RÉSUMÉ POUR LE RAPPORT")
print("=" * 60)
for k, v in summary.items():
    print(f"{k:45s}: {v:.4f}" if isinstance(v, float) else f"{k:45s}: {v}")
print("\nFichiers générés : fairness_by_age.png, fairness_metrics_table.csv, fairness_summary.json")
print("Objet de mitigation : reports/threshold_optimizer.pkl (repris par l'API)")
print("Prochaine étape : API FastAPI (src/api.py)")
print("=" * 60)
