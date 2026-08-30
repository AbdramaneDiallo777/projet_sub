"""
Étape 6 — API FastAPI
Projet : Scoring de risque explicable et robuste

Expose le modèle XGBoost via une API REST :
    - POST /predict : prend un profil client, retourne un score de risque
      + les 5 variables qui ont le plus influencé la décision (SHAP)

Corrections apportées par rapport à la version initiale :
    - Chemins robustes (indépendants du dossier de lancement).
    - Le nettoyage/feature engineering réutilise src/preprocessing.py
      (les mêmes fonctions et les mêmes paramètres calculés sur le train
      à l'étape 2) au lieu d'une logique dupliquée et non testée ici.
    - La décision finale n'utilise plus un seuil fixe de 0.5 : elle
      s'appuie sur l'objet de mitigation d'équité (ThresholdOptimizer,
      étape 5), qui applique un seuil ajusté par tranche d'âge pour
      satisfaire la contrainte d'équité. La probabilité brute et
      l'explication SHAP restent celles du modèle XGBoost original
      (transparence conservée).

Comment lancer (depuis n'importe quel dossier) :
    uvicorn src.api:app --reload   (depuis la racine scoring_credit)
    ou
    uvicorn api:app --reload       (depuis src/)

Une fois lancé, ouvre http://127.0.0.1:8000/docs pour tester l'API
directement dans le navigateur (interface Swagger générée automatiquement).
"""

import sys
import types
from pathlib import Path

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

import json

import warnings

import joblib
import shap
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

# Même warning inoffensif qu'à l'étape 5 (voir 05_equite.py), déclenché
# ici par le ThresholdOptimizer utilisé pour la décision finale.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="fairlearn.postprocessing.*"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import apply_cleaning, add_engineered_features, load_cleaning_params

# ----------------------------------------------------------------
# 0. Chemins robustes
# ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

# ----------------------------------------------------------------
# 1. Chargement du modèle, des paramètres de nettoyage et de la
#    mitigation d'équité (une seule fois, au démarrage de l'API)
# ----------------------------------------------------------------
xgb = joblib.load(REPORTS_DIR / "model_xgb.pkl")
explainer = shap.TreeExplainer(xgb)
cleaning_params = load_cleaning_params(REPORTS_DIR / "cleaning_params.json")

with open(REPORTS_DIR / "decision_threshold.json", "r", encoding="utf-8") as f:
    decision_threshold = json.load(f)["threshold"]

# L'objet de mitigation d'équité est optionnel : si l'étape 5 n'a pas
# encore été lancée, l'API retombe sur le seuil optimisé "brut" (sans
# ajustement par groupe) plutôt que de planter.
fairness_mitigator = None
mitigator_path = REPORTS_DIR / "threshold_optimizer.pkl"
if mitigator_path.exists():
    fairness_mitigator = joblib.load(mitigator_path)

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
                 "avec explication des facteurs les plus influents (SHAP) "
                 "et une décision corrigée pour l'équité entre tranches d'âge.",
    version="1.1",
)


# ----------------------------------------------------------------
# 2. Schéma des données attendues en entrée
# ----------------------------------------------------------------
class ClientProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., json_schema_extra={"example": 0.3},
        description="Taux d'utilisation du crédit renouvelable",
    )
    age: int = Field(..., json_schema_extra={"example": 45}, description="Âge du client")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., json_schema_extra={"example": 0}, alias="NumberOfTime30-59DaysPastDueNotWorse"
    )
    DebtRatio: float = Field(
        ..., json_schema_extra={"example": 0.4}, description="Ratio d'endettement"
    )
    MonthlyIncome: float = Field(
        ..., json_schema_extra={"example": 5000}, description="Revenu mensuel"
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(..., json_schema_extra={"example": 8})
    NumberOfTimes90DaysLate: int = Field(..., json_schema_extra={"example": 0})
    NumberRealEstateLoansOrLines: int = Field(..., json_schema_extra={"example": 1})
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., json_schema_extra={"example": 0}, alias="NumberOfTime60-89DaysPastDueNotWorse"
    )
    NumberOfDependents: int = Field(..., json_schema_extra={"example": 1})


# ----------------------------------------------------------------
# 3. Reconstruction des features — réutilise EXACTEMENT la même logique
#    (et les mêmes paramètres, calculés sur le train) que le pipeline
#    d'entraînement, via src/preprocessing.py.
# ----------------------------------------------------------------
def construire_features(profil: ClientProfile) -> pd.DataFrame:
    data = profil.model_dump(by_alias=True)
    df = pd.DataFrame([data])

    df = apply_cleaning(df, cleaning_params)
    df = add_engineered_features(df)

    age_group = df["AgeGroup"].iloc[0]
    X = df[FEATURE_ORDER]
    return X, age_group


# ----------------------------------------------------------------
# 4. Endpoint principal
# ----------------------------------------------------------------
@app.post("/predict")
def predict(profil: ClientProfile):
    X, age_group = construire_features(profil)

    proba_defaut = float(xgb.predict_proba(X)[0, 1])

    # Décision : si la mitigation d'équité est disponible, on l'utilise
    # (seuil ajusté par tranche d'âge). Sinon, on retombe sur le seuil
    # global optimisé (F1) calculé à l'étape 3.
    if fairness_mitigator is not None:
        decision_brute = int(
            fairness_mitigator.predict(
                X, sensitive_features=pd.Series([age_group]), random_state=42
            )[0]
        )
        methode_decision = "seuil ajusté par tranche d'âge (mitigation d'équité)"
    else:
        decision_brute = int(proba_defaut >= decision_threshold)
        methode_decision = f"seuil global optimisé ({decision_threshold:.3f})"

    decision = "à risque" if decision_brute == 1 else "sûr"

    # Explication SHAP pour CE client précis (toujours basée sur le
    # modèle brut : la mitigation ne change que le seuil final, pas les
    # probabilités ni leur explication)
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
        "methode_decision": methode_decision,
        "tranche_age": str(age_group),
        "facteurs_principaux": top_facteurs,
    }


@app.get("/")
def health_check():
    return {
        "status": "API opérationnelle",
        "modele": "XGBoost - scoring crédit",
        "mitigation_equite_active": fairness_mitigator is not None,
    }
