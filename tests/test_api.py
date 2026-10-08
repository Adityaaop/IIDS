import sys
from pathlib import Path
import pytest

# Ensure backend directory is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "backend"))

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def auth_client(client):
    with client.session_transaction() as sess:
        sess["authenticated"] = True
        sess["username"] = "admin"
    return client


def test_status_endpoint(client):
    """Verify backend health and artifact availability."""
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "online"
    assert "models_loaded" in data


def test_unauthenticated_access_rejected(client):
    """Verify protected endpoints reject unauthenticated requests with 401."""
    response = client.get("/api/settings")
    assert response.status_code == 401


def test_settings_retrieval(auth_client):
    """Verify default engine settings for authenticated operator."""
    response = auth_client.get("/api/settings")
    assert response.status_code == 200
    data = response.get_json()
    assert "settings" in data
    assert data["settings"]["engine"] in {"hybrid", "rf_only", "cnn_only"}


def test_predict_requires_payload(auth_client):
    """Verify prediction endpoint rejects empty body."""
    response = auth_client.post("/api/predict", json={})
    assert response.status_code in {400, 422}


def test_login_successful(client):
    """Verify admin login with default credentials."""
    response = client.post("/api/login", json={"username": "admin", "password": "admin"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
