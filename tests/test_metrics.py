from fastapi.testclient import TestClient

from app.main import app


def test_metrics_endpoint_counts_requests():
    with TestClient(app) as client:
        client.get("/health")
        body = client.get("/metrics").text
    assert "api_requests_total" in body
    assert "api_request_latency_seconds" in body
