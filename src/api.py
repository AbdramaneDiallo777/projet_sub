"""
Étape 6 — API FastAPI
Projet : Scoring de risque explicable et robuste

Expose le modèle XGBoost via une API REST :
    - POST /predict : prend un profil client, retourne un score de risque
      + les 5 variables qui ont le plus influencé la décision (SHAP)

Comment lancer (depuis la racine scoring_credit) :
    uvicorn src.api:app --reload

Une fois lancé, ouvre http://127.0.0.1:8000/docs pour tester l'API
directement dans le navigateur (interface Swagger générée automatiquement).
"""

import sys
import types

# Contournement numba (voir étapes 4 et 5) pour SHAP
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

import joblib
import shap
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ----------------------------------------------------------------
# 1. Chargement du modèle (une seule fois, au démarrage de l'API)
# ----------------------------------------------------------------
xgb = joblib.load("reports/model_xgb.pkl")
explainer = shap.TreeExplainer(xgb)

# Ordre exact des colonnes attendues par le modèle (doit correspondre
# à X_train dans 03_modelisation.py)
FEATURE_ORDER = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "TotalPastDueIncidents",
    "HasPastDueHistory",
    "IncomePerDependent",
]

app = FastAPI(
    title="API de scoring de risque crédit",
    description="Prédit la probabilité de défaut de paiement d'un client, "
                 "avec explication des facteurs les plus influents (SHAP).",
    version="1.0",
)


# ----------------------------------------------------------------
# 2. Schéma des données attendues en entrée
# ----------------------------------------------------------------
class ClientProfile(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., example=0.3, description="Taux d'utilisation du crédit renouvelable"
    )
    age: int = Field(..., example=45, description="Âge du client")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., example=0, alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float = Field(..., example=0.4, description="Ratio d'endettement")
    MonthlyIncome: float = Field(..., example=5000, description="Revenu mensuel")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., example=8)
    NumberOfTimes90DaysLate: int = Field(..., example=0)
    NumberRealEstateLoansOrLines: int = Field(..., example=1)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., example=0, alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: int = Field(..., example=1)

    class Config:
        populate_by_name = True


# ----------------------------------------------------------------
# 3. Fonction utilitaire : reconstruit les features dérivées
#    (les mêmes que celles créées à l'étape 2 - feature engineering)
# ----------------------------------------------------------------
def construire_features(profil: ClientProfile) -> pd.DataFrame:
    data = profil.model_dump(by_alias=True)

    total_past_due = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + data["NumberOfTime60-89DaysPastDueNotWorse"]
        + data["NumberOfTimes90DaysLate"]
    )
    data["TotalPastDueIncidents"] = total_past_due
    data["HasPastDueHistory"] = int(total_past_due > 0)
    data["IncomePerDependent"] = data["MonthlyIncome"] / (data["NumberOfDependents"] + 1)

    df = pd.DataFrame([data])[FEATURE_ORDER]
    return df


# ----------------------------------------------------------------
# 4. Endpoint principal
# ----------------------------------------------------------------
@app.post("/predict")
def predict(profil: ClientProfile):
    X = construire_features(profil)

    proba_defaut = float(xgb.predict_proba(X)[0, 1])
    decision = "à risque" if proba_defaut >= 0.5 else "sûr"

    # Explication SHAP pour CE client précis
    shap_values = explainer(X)
    contributions = pd.Series(
        shap_values.values[0], index=X.columns
    ).sort_values(key=abs, ascending=False)

    top_facteurs = [
        {
            "variable": var,
            "valeur_client": float(X[var].iloc[0]),
            "impact_shap": round(float(contrib), 4),
            "effet": "augmente le risque" if contrib > 0 else "diminue le risque",
        }
        for var, contrib in contributions.head(5).items()
    ]

    return {
        "probabilite_defaut": round(proba_defaut, 4),
        "decision": decision,
        "facteurs_principaux": top_facteurs,
    }


@app.get("/")
def health_check():
    return {"status": "API opérationnelle", "modele": "XGBoost - scoring crédit"}