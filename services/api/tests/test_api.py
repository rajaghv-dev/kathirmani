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
def test_model_registry_and_activate():
    assert client.get("/api/model-registry").status_code == 200
    r = client.post("/api/model-registry/activate/nvidia_gb10_retail_balanced")
    assert r.status_code == 200
    assert client.get("/api/model-registry/active").json()["profile_name"] == "nvidia_gb10_retail_balanced"
    assert client.post("/api/model-registry/activate/does_not_exist").status_code == 404
