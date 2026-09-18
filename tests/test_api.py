import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
sys.path.insert(0, str(SRC_DIR))

pytestmark = pytest.mark.skipif(
    not (REPORTS_DIR / "model_xgb.pkl").exists(),
    reason="Le pipeline (etapes 2 a 5) doit avoir tourne au moins une fois "
           "pour generer les artefacts necessaires a l'API.",
)


@pytest.fixture(scope="module")
def client():
    import api  # import différé : nécessite les artefacts déjà générés
    return TestClient(api.app)


VALID_PROFILE = {
    "RevolvingUtilizationOfUnsecuredLines": 0.3,
    "age": 45,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.4,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "API opérationnelle"
    assert "mitigation_equite_active" in body


def test_predict_returns_expected_shape(client):
    response = client.post("/predict", json=VALID_PROFILE)
    assert response.status_code == 200
    body = response.json()

    assert "probabilite_defaut" in body
    assert 0.0 <= body["probabilite_defaut"] <= 1.0
    assert body["decision"] in {"à risque", "sûr"}
    assert "methode_decision" in body
    assert "tranche_age" in body
    assert len(body["facteurs_principaux"]) == 5
    for facteur in body["facteurs_principaux"]:
        assert facteur["effet"] in {"augmente le risque", "diminue le risque"}


def test_predict_high_risk_profile_flagged_as_risky(client):
    """Un profil manifestement dégradé (nombreux retards de paiement,
    revenu très faible) doit être classé à risque avec une proba haute."""
    profil_risque = dict(VALID_PROFILE)
    profil_risque.update({
        "NumberOfTime30-59DaysPastDueNotWorse": 5,
        "NumberOfTime60-89DaysPastDueNotWorse": 3,
        "NumberOfTimes90DaysLate": 4,
        "MonthlyIncome": 800,
        "DebtRatio": 1.5,
    })
    response = client.post("/predict", json=profil_risque)
    assert response.status_code == 200
    body = response.json()
    assert body["probabilite_defaut"] > 0.5


def test_predict_missing_field_returns_422(client):
    profil_incomplet = dict(VALID_PROFILE)
    del profil_incomplet["age"]
    response = client.post("/predict", json=profil_incomplet)
    assert response.status_code == 422
