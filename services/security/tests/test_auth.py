"""auth: Bearer/API-key dependency + token helper. No DB needed."""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from services.security import auth


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(auth.ENV_KEY, "s3cr3t-key")
    app = FastAPI()

    @app.get("/protected")
    def protected(principal: str = Depends(auth.require_auth)):
        return {"principal": principal}

    return TestClient(app)


def test_issue_token_returns_configured_key(monkeypatch):
    monkeypatch.setenv(auth.ENV_KEY, "abc123")
    assert auth.issue_token() == "abc123"


def test_issue_token_raises_when_unset(monkeypatch):
    monkeypatch.delenv(auth.ENV_KEY, raising=False)
    with pytest.raises(RuntimeError):
        auth.issue_token()


def test_no_credentials_401(client):
    assert client.get("/protected").status_code == 401


def test_wrong_key_401(client):
    r = client.get("/protected", headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_bearer_token_ok(client):
    r = client.get("/protected", headers={"Authorization": "Bearer s3cr3t-key"})
    assert r.status_code == 200 and r.json()["principal"] == "api-key"


def test_x_api_key_ok(client):
    r = client.get("/protected", headers={"X-API-Key": "s3cr3t-key"})
    assert r.status_code == 200


def test_closed_when_no_key_configured(monkeypatch):
    monkeypatch.delenv(auth.ENV_KEY, raising=False)
    assert auth.verify("anything") is False  # unset env → API refuses all


def test_error_never_echoes_token(client):
    r = client.get("/protected", headers={"X-API-Key": "leaky-secret-value"})
    assert "leaky-secret-value" not in r.text
