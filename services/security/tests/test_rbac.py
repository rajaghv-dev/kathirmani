"""rbac: least-privilege matrix + require_role dependency. No DB needed."""
from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from services.security import auth, rbac


def test_matrix_least_privilege():
    # viewer: read only
    assert rbac.allowed("viewer", "read")
    assert not rbac.allowed("viewer", "approve")
    assert not rbac.allowed("viewer", "activate-profile")
    # reviewer: read + approve/reject, not admin actions
    assert rbac.allowed("reviewer", "read")
    assert rbac.allowed("reviewer", "approve")
    assert rbac.allowed("reviewer", "reject")
    assert not rbac.allowed("reviewer", "retention")
    # admin: everything
    for a in ("read", "approve", "reject", "activate-profile", "retention", "backup"):
        assert rbac.allowed("admin", a)


def test_matrix_unknown_inputs():
    assert not rbac.allowed("ghost", "read")
    assert not rbac.allowed("admin", "nuke-everything")


def test_require_role_unknown_role_raises():
    with pytest.raises(ValueError):
        rbac.require_role("superuser")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(auth.ENV_KEY, "k")
    app = FastAPI()

    @app.post("/approve", dependencies=[Depends(rbac.require_role("reviewer"))])
    def approve():
        return {"ok": True}

    @app.post("/retention", dependencies=[Depends(rbac.require_role("admin"))])
    def retention():
        return {"ok": True}

    return TestClient(app)


def _h(role):
    return {"X-API-Key": "k", "X-Role": role}


def test_reviewer_endpoint_role_gate(client):
    assert client.post("/approve", headers=_h("viewer")).status_code == 403
    assert client.post("/approve", headers=_h("reviewer")).status_code == 200
    assert client.post("/approve", headers=_h("admin")).status_code == 200


def test_admin_endpoint_role_gate(client):
    assert client.post("/retention", headers=_h("reviewer")).status_code == 403
    assert client.post("/retention", headers=_h("admin")).status_code == 200


def test_role_gate_requires_auth_first(client):
    # no API key → 401 before role is even considered
    assert client.post("/approve", headers={"X-Role": "admin"}).status_code == 401
