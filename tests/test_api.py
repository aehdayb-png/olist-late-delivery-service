import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_payload():
    return {
        "order_purchase_timestamp": "2017-10-02T10:56:33",
        "order_estimated_delivery_date": "2017-10-18T00:00:00",
        "customer_state": "SP",
        "n_items": 1,
        "total_price": 29.99,
        "total_freight": 8.72,
        "n_distinct_products": 1,
        "n_distinct_sellers": 1,
        "total_payment": 38.71,
        "n_payment_entries": 3,
        "max_installments": 1,
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_version"] == "logreg_v1"
    assert response.json()["model_type"] == "LogisticRegression"
    assert response.json()["features_count"] == 39


def test_single_prediction_endpoint(client, sample_payload):
    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["late_probability"] <= 1
    assert 0 <= result["late_risk_percent"] <= 100
    assert result["predicted_is_late"] in [0, 1]
    assert result["model_version"] == "logreg_v1"


def test_batch_prediction_endpoint(client, sample_payload):
    response = client.post(
        "/predict/batch",
        json=[sample_payload, sample_payload],
    )

    assert response.status_code == 200

    result = response.json()
    assert result["total_orders"] == 2
    assert len(result["predictions"]) == 2
    assert result["model_version"] == "logreg_v1"


def test_invalid_state_returns_422(client, sample_payload):
    sample_payload["customer_state"] = "XX"

    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 422


def test_empty_batch_returns_422(client):
    response = client.post("/predict/batch", json=[])

    assert response.status_code == 422


def test_unknown_request_field_returns_422(client, sample_payload):
    sample_payload["unexpected_field"] = "invalid"

    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 422
