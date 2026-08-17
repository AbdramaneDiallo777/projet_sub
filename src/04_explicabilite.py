"""
Étape 4 — Explicabilité (SHAP)
Projet : Scoring de risque explicable et robuste

Explique les décisions du modèle XGBoost (le meilleur des 2 modèles) avec SHAP :
    1. Importance globale des variables (quelles variables comptent le plus, en moyenne)
    2. Summary plot (impact + direction de chaque variable)
    3. Waterfall plot (explication détaillée pour un client précis)

Comment lancer :
    python src/04_explicabilite.py
"""

import sys
import types

# ------------------------------------------------------------------
# Contournement Windows : sur certains PC, une stratégie de contrôle
# d'application (Smart App Control) bloque le fichier .dll utilisé par
# `numba` pour accélérer certains calculs internes de SHAP.
# On remplace `numba` par un module factice AVANT l'import de shap :
# shap fonctionne toujours normalement, juste un peu plus lentement
# sur les gros volumes (sans impact ici, vu qu'on travaille sur un
# échantillon de 1500 lignes).
# ------------------------------------------------------------------
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

# ----------------------------------------------------------------
# 1. Chargement du modèle et des données
# ----------------------------------------------------------------
xgb = joblib.load("reports/model_xgb.pkl")
test = pd.read_csv("data/test_clean.csv")

TARGET = "SeriousDlqin2yrs"
X_test = test.drop(columns=[TARGET, "AgeGroup"])
y_test = test[TARGET]

print("=" * 60)
print("Calcul des SHAP values sur XGBoost...")
print("=" * 60)

# Pour aller plus vite, on calcule les SHAP values sur un échantillon
# (1500 clients) plutôt que sur les 30 000 lignes du test set complet.
# C'est une pratique standard : les résultats globaux (importance des
# variables) sont déjà stables avec un échantillon de cette taille.
np.random.seed(42)
sample_idx = np.random.choice(X_test.index, size=1500, replace=False)
X_sample = X_test.loc[sample_idx]

# TreeExplainer est optimisé pour les modèles à arbres (XGBoost, RF, etc.)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_sample)

print(f"SHAP values calculées sur un échantillon de {len(X_sample)} clients")

# ----------------------------------------------------------------
# 2. Importance globale des variables (bar plot)
# ----------------------------------------------------------------
plt.figure()
shap.plots.bar(shap_values, show=False, max_display=13)
plt.title("Importance moyenne des variables (SHAP)")
plt.tight_layout()
plt.savefig("reports/shap_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nGraphique sauvegardé : reports/shap_importance.png")

# ----------------------------------------------------------------
# 3. Summary plot (impact + direction)
# ----------------------------------------------------------------
# Ce graphique montre, pour chaque variable, si une valeur haute ou basse
# pousse la prédiction vers "défaut" ou "pas de défaut"
plt.figure()
shap.plots.beeswarm(shap_values, show=False, max_display=13)
plt.title("Impact et direction des variables (SHAP)")
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("Graphique sauvegardé : reports/shap_summary.png")

# ----------------------------------------------------------------
# 4. Waterfall plot pour un client à risque (exemple individuel)
# ----------------------------------------------------------------
# On choisit un client que le modèle juge à risque (proba élevée) pour
# avoir un exemple parlant dans le rapport
probas = xgb.predict_proba(X_sample)[:, 1]
idx_risque = np.argmax(probas)

print(f"\nClient sélectionné pour l'exemple : probabilité de défaut = {probas[idx_risque]:.2%}")

plt.figure()
shap.plots.waterfall(shap_values[idx_risque], show=False, max_display=10)
plt.title(f"Explication détaillée — client à risque (proba={probas[idx_risque]:.1%})")
plt.tight_layout()
plt.savefig("reports/shap_waterfall_risque.png", dpi=120, bbox_inches="tight")
plt.close()
print("Graphique sauvegardé : reports/shap_waterfall_risque.png")

# ----------------------------------------------------------------
# 5. Waterfall plot pour un client sûr (contre-exemple)
# ----------------------------------------------------------------
idx_sur = np.argmin(probas)
print(f"Client sélectionné (sûr) : probabilité de défaut = {probas[idx_sur]:.2%}")

plt.figure()
shap.plots.waterfall(shap_values[idx_sur], show=False, max_display=10)
plt.title(f"Explication détaillée — client sûr (proba={probas[idx_sur]:.1%})")
plt.tight_layout()
plt.savefig("reports/shap_waterfall_sur.png", dpi=120, bbox_inches="tight")
plt.close()
print("Graphique sauvegardé : reports/shap_waterfall_sur.png")

# ----------------------------------------------------------------
# 6. Résumé texte de l'importance des variables (pour le rapport écrit)
# ----------------------------------------------------------------
importance_moyenne = pd.DataFrame({
    "variable": X_sample.columns,
    "importance_shap": np.abs(shap_values.values).mean(axis=0),
}).sort_values("importance_shap", ascending=False)

print("\n" + "=" * 60)
print("TOP 5 VARIABLES LES PLUS IMPORTANTES")
print("=" * 60)
print(importance_moyenne.head(5).to_string(index=False))

importance_moyenne.to_csv("reports/shap_importance_table.csv", index=False)

print("\n" + "=" * 60)
print("Tous les graphiques SHAP sont dans reports/")
print("Table d'importance sauvegardée : reports/shap_importance_table.csv")
print("Prochaine étape : équité / robustesse (src/05_equite.py)")
print("=" * 60)
