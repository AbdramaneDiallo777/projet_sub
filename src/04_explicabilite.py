import sys
import types


if "numba" not in sys.modules:
    try:
        import numba  # noqa: F401
        import numba.typed  # noqa: F401
    except ImportError:
        def _identity_decorator(*args, **kwargs):
            # Permet d'utiliser aussi bien @numba.njit que @numba.njit(...)
            if args and callable(args[0]) and not kwargs:
                return args[0]
            return lambda f: f

        fake_numba = types.ModuleType("numba")
        fake_numba.njit = _identity_decorator
        fake_numba.jit = _identity_decorator
        fake_numba.vectorize = _identity_decorator
        fake_numba.prange = range

        fake_numba_typed = types.ModuleType("numba.typed")
        fake_numba_typed.List = list  # numba.typed.List se comporte comme une liste normale
        fake_numba_typed.Dict = dict
        fake_numba.typed = fake_numba_typed

        sys.modules["numba"] = fake_numba
        sys.modules["numba.typed"] = fake_numba_typed
        print("(info) numba indisponible sur ce PC -> mode de secours activé, "
              "sans impact sur les résultats)")

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

xgb = joblib.load(REPORTS_DIR / "model_xgb.pkl")
test = pd.read_csv(DATA_DIR / "test_clean.csv")

TARGET = "SeriousDlqin2yrs"
X_test = test.drop(columns=[TARGET, "AgeGroup"])
y_test = test[TARGET]

print("=" * 60)
print("Calcul des SHAP values sur XGBoost...")
print("=" * 60)

#
np.random.seed(42)
sample_idx = np.random.choice(X_test.index, size=1500, replace=False)
X_sample = X_test.loc[sample_idx]

# TreeExplainer est optimisé pour les modèles à arbres (XGBoost, RF, etc.)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_sample)

print(f"SHAP values calculées sur un échantillon de {len(X_sample)} clients")


plt.figure()
shap.plots.bar(shap_values, show=False, max_display=13)
plt.title("Importance moyenne des variables (SHAP)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "shap_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nGraphique sauvegardé : reports/shap_importance.png")


plt.figure()
shap.plots.beeswarm(shap_values, show=False, max_display=13)
plt.title("Impact et direction des variables (SHAP)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("Graphique sauvegardé : reports/shap_summary.png")


probas = xgb.predict_proba(X_sample)[:, 1]
idx_risque = np.argmax(probas)

print(f"\nClient sélectionné pour l'exemple : probabilité de défaut = {probas[idx_risque]:.2%}")

plt.figure()
shap.plots.waterfall(shap_values[idx_risque], show=False, max_display=10)
plt.title(f"Explication détaillée — client à risque (proba={probas[idx_risque]:.1%})")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "shap_waterfall_risque.png", dpi=120, bbox_inches="tight")
plt.close()
print("Graphique sauvegardé : reports/shap_waterfall_risque.png")

idx_sur = np.argmin(probas)
print(f"Client sélectionné (sûr) : probabilité de défaut = {probas[idx_sur]:.2%}")

plt.figure()
shap.plots.waterfall(shap_values[idx_sur], show=False, max_display=10)
plt.title(f"Explication détaillée — client sûr (proba={probas[idx_sur]:.1%})")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "shap_waterfall_sur.png", dpi=120, bbox_inches="tight")
plt.close()
print("Graphique sauvegardé : reports/shap_waterfall_sur.png")


importance_moyenne = pd.DataFrame({
    "variable": X_sample.columns,
    "importance_shap": np.abs(shap_values.values).mean(axis=0),
}).sort_values("importance_shap", ascending=False)

print("\n" + "=" * 60)
print("TOP 5 VARIABLES LES PLUS IMPORTANTES")
print("=" * 60)
print(importance_moyenne.head(5).to_string(index=False))

importance_moyenne.to_csv(REPORTS_DIR / "shap_importance_table.csv", index=False)

print("\n" + "=" * 60)
print("Tous les graphiques SHAP sont dans reports/")
print("Table d'importance sauvegardée : reports/shap_importance_table.csv")
print("Prochaine étape : équité / robustesse (src/05_equite.py)")
print("=" * 60)
