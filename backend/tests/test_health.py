"""
test_health.py
Smoke test — confirms the FastAPI app boots and responds. This is
deliberately the first test in the repo so GitHub Actions CI has
something real to run from commit #1, before the RAG logic exists.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_returns_message():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
