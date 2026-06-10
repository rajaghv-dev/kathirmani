"""API tests. /health works without a DB; DB-backed endpoints skip if Postgres is down."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.api.app import app

client = TestClient(app)


def _db_ok() -> bool:
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def test_health_always_200():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.skipif(not _db_ok(), reason="no Postgres")
def test_cameras_list():
    r = client.get("/cameras")
    assert r.status_code == 200 and isinstance(r.json(), list)


@pytest.mark.skipif(not _db_ok(), reason="no Postgres")
def test_model_registry_and_activate(monkeypatch):
    assert client.get("/api/model-registry").status_code == 200   # dev-open read
    # activate is admin-gated → run keyed (require_role enforces auth + role)
    monkeypatch.setenv("PLATFORM_API_KEY", "k")
    admin = {"X-API-Key": "k", "X-Role": "admin"}
    assert client.post("/api/model-registry/activate/nvidia_gb10_retail_balanced", headers=admin).status_code == 200
    assert client.get("/api/model-registry/active", headers={"X-API-Key": "k"}).json()["profile_name"] == "nvidia_gb10_retail_balanced"
    assert client.post("/api/model-registry/activate/does_not_exist", headers=admin).status_code == 404
    # right key, wrong role → 403 (RBAC)
    assert client.post("/api/model-registry/activate/nvidia_gb10_retail_balanced",
                       headers={"X-API-Key": "k", "X-Role": "viewer"}).status_code == 403
    # keyed but no token → 401 (authn)
    assert client.post("/api/model-registry/activate/nvidia_gb10_retail_balanced",
                       headers={"X-Role": "admin"}).status_code == 401


def test_auth_enforced_when_keyed(monkeypatch):
    monkeypatch.setenv("PLATFORM_API_KEY", "secret-test-key")
    assert client.get("/cameras").status_code == 401          # no creds → 401 (before DB)
    # valid key passes auth — assert against /health (auth-gated but DB-independent:
    # it swallows DB errors) so this security check runs even with no Postgres.
    assert client.get("/health", headers={"X-API-Key": "secret-test-key"}).status_code != 401
