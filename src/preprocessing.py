from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

AGE_BINS = [0, 30, 40, 50, 60, 70, 120]
AGE_LABELS = ["<30", "30-40", "40-50", "50-60", "60-70", "70+"]


def fit_cleaning_params(df: pd.DataFrame) -> dict:
    """
    Calcule toutes les statistiques de nettoyage (médianes, seuils de
    percentile) à partir du DataFrame fourni. À appeler UNIQUEMENT sur
    le train set.
    """
    params: dict = {}

    # Médianes de remplacement pour les codes d'erreur 96/98
    params["late_cols_medians"] = {}
    for col in LATE_COLS:
        valeurs_valides = df.loc[~df[col].isin([96, 98]), col]
        params["late_cols_medians"][col] = float(valeurs_valides.median())

    # Seuils de capping (99.5e percentile), calculés APRÈS correction des
    # codes d'erreur pour ne pas fausser les percentiles
    df_tmp = df.copy()
    for col, med in params["late_cols_medians"].items():
        df_tmp.loc[df_tmp[col].isin([96, 98]), col] = med

    params["outlier_caps"] = {}
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        params["outlier_caps"][col] = float(df_tmp[col].quantile(0.995))

    # Médiane pour NumberOfDependents
    params["dependents_median"] = float(df["NumberOfDependents"].median())

    # Médianes de MonthlyIncome par tranche d'âge
    age_bracket = pd.cut(df["age"], bins=AGE_BINS)
    med_par_tranche = df.groupby(age_bracket, observed=True)["MonthlyIncome"].median()
    params["income_median_by_age_bracket"] = {
        str(k): (float(v) if pd.notna(v) else None) for k, v in med_par_tranche.items()
    }
    params["income_global_median"] = float(df["MonthlyIncome"].median())

    return params


def apply_cleaning(df: pd.DataFrame, params: dict) -> pd.DataFrame:

    df = df.copy()

    # Codes d'erreur 96/98 -> médiane (du train)
    for col, med in params["late_cols_medians"].items():
        if col in df.columns:
            df.loc[df[col].isin([96, 98]), col] = med

    # Capping des outliers (seuils du train)
    for col, seuil in params["outlier_caps"].items():
        if col in df.columns:
            df[col] = df[col].clip(upper=seuil)

    # NumberOfDependents manquant -> médiane du train
    if "NumberOfDependents" in df.columns:
        df["NumberOfDependents"] = df["NumberOfDependents"].fillna(
            params["dependents_median"]
        )

    # MonthlyIncome manquant -> médiane par tranche d'âge (du train)
    if "MonthlyIncome" in df.columns and "age" in df.columns:
        age_bracket = pd.cut(df["age"], bins=AGE_BINS).astype(str)
        medians_map = params["income_median_by_age_bracket"]
        fallback = medians_map.get(str(age_bracket), params["income_global_median"])
        imputed = age_bracket.map(medians_map)
        imputed = imputed.fillna(params["income_global_median"])
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(imputed)
        # sécurité ultime
        df["MonthlyIncome"] = df["MonthlyIncome"].fillna(params["income_global_median"])

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée les variables dérivées. Purement déterministe ligne par ligne
    (aucune statistique globale) -> peut être appliqué sans risque de
    fuite de données sur train, test, ou une requête API individuelle.
    """
    df = df.copy()

    df["TotalPastDueIncidents"] = df[LATE_COLS].sum(axis=1)
    df["HasPastDueHistory"] = (df["TotalPastDueIncidents"] > 0).astype(int)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    if "age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS
        )

    return df


def save_cleaning_params(params: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def load_cleaning_params(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
