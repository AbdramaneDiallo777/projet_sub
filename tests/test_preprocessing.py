"""
Tests unitaires du module de preprocessing partagé (src/preprocessing.py).

Comment lancer (depuis la racine scoring_credit) :
    pip install pytest
    pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import (
    fit_cleaning_params,
    apply_cleaning,
    add_engineered_features,
    LATE_COLS,
)


@pytest.fixture
def fake_train_df():
    """Petit dataset synthétique reproduisant les problèmes connus du
    dataset réel (codes 96/98, valeurs manquantes, outliers)."""
    n = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.integers(20, 80, size=n),
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1.5, size=n),
        "DebtRatio": rng.uniform(0, 2, size=n),
        "MonthlyIncome": rng.uniform(1000, 10000, size=n),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, size=n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, size=n),
        "NumberOfDependents": rng.integers(0, 4, size=n).astype(float),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, size=n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 5, size=n),
        "NumberOfTimes90DaysLate": rng.integers(0, 5, size=n),
    })
    # Injecte des codes d'erreur 96/98 sur quelques lignes
    df.loc[0:2, "NumberOfTime30-59DaysPastDueNotWorse"] = 98
    df.loc[0:2, "NumberOfTime60-89DaysPastDueNotWorse"] = 98
    df.loc[0:2, "NumberOfTimes90DaysLate"] = 96
    # Injecte des valeurs manquantes
    df.loc[5:15, "MonthlyIncome"] = np.nan
    df.loc[3, "NumberOfDependents"] = np.nan
    # Injecte un outlier extrême
    df.loc[7, "RevolvingUtilizationOfUnsecuredLines"] = 999.0
    return df


def test_fit_cleaning_params_returns_expected_keys(fake_train_df):
    params = fit_cleaning_params(fake_train_df)
    assert set(params.keys()) == {
        "late_cols_medians", "outlier_caps",
        "dependents_median", "income_median_by_age_bracket",
        "income_global_median",
    }
    assert set(params["late_cols_medians"].keys()) == set(LATE_COLS)


def test_apply_cleaning_removes_error_codes(fake_train_df):
    params = fit_cleaning_params(fake_train_df)
    cleaned = apply_cleaning(fake_train_df, params)
    for col in LATE_COLS:
        assert not cleaned[col].isin([96, 98]).any(), (
            f"{col} contient encore des codes d'erreur 96/98 après nettoyage"
        )


def test_apply_cleaning_fills_missing_values(fake_train_df):
    params = fit_cleaning_params(fake_train_df)
    cleaned = apply_cleaning(fake_train_df, params)
    assert cleaned["MonthlyIncome"].isnull().sum() == 0
    assert cleaned["NumberOfDependents"].isnull().sum() == 0


def test_apply_cleaning_caps_outliers(fake_train_df):
    params = fit_cleaning_params(fake_train_df)
    cleaned = apply_cleaning(fake_train_df, params)
    seuil = params["outlier_caps"]["RevolvingUtilizationOfUnsecuredLines"]
    assert cleaned["RevolvingUtilizationOfUnsecuredLines"].max() <= seuil + 1e-9


def test_apply_cleaning_does_not_recompute_stats_on_new_data(fake_train_df):
    """
    Test clé anti-data-leakage : appliquer les params du train à un
    DataFrame différent (simulant le test set / une requête API) ne doit
    JAMAIS recalculer de statistique sur ces nouvelles données -> le
    seuil de capping appliqué doit être identique à celui du train, même
    si les nouvelles données ont une distribution différente.
    """
    params = fit_cleaning_params(fake_train_df)
    seuil_train = params["outlier_caps"]["RevolvingUtilizationOfUnsecuredLines"]

    nouvelles_donnees = fake_train_df.copy()
    nouvelles_donnees["RevolvingUtilizationOfUnsecuredLines"] = 5000.0  # distribution très différente

    cleaned = apply_cleaning(nouvelles_donnees, params)
    # Toutes les valeurs doivent être cappées exactement au seuil du train
    assert (cleaned["RevolvingUtilizationOfUnsecuredLines"] == seuil_train).all()


def test_add_engineered_features_creates_expected_columns(fake_train_df):
    params = fit_cleaning_params(fake_train_df)
    cleaned = apply_cleaning(fake_train_df, params)
    result = add_engineered_features(cleaned)
    for col in ["TotalPastDueIncidents", "HasPastDueHistory", "IncomePerDependent", "AgeGroup"]:
        assert col in result.columns


def test_has_past_due_history_is_binary_indicator(fake_train_df):
    params = fit_cleaning_params(fake_train_df)
    cleaned = apply_cleaning(fake_train_df, params)
    result = add_engineered_features(cleaned)
    assert set(result["HasPastDueHistory"].unique()).issubset({0, 1})
    # cohérence : HasPastDueHistory=1 ssi TotalPastDueIncidents > 0
    assert (
        (result["HasPastDueHistory"] == 1) == (result["TotalPastDueIncidents"] > 0)
    ).all()


def test_income_per_dependent_never_divides_by_zero(fake_train_df):
    params = fit_cleaning_params(fake_train_df)
    cleaned = apply_cleaning(fake_train_df, params)
    result = add_engineered_features(cleaned)
    assert np.isfinite(result["IncomePerDependent"]).all()
