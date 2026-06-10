"""Console gateway — serves the SPA and proxies to upstreams, degrading cleanly
when an upstream is down. No real API/review-UI/Grafana needed (httpx is mocked)."""
from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

import app as console


client = TestClient(console.app)


def test_index_serves_spa():
    r = client.get("/")
    assert r.status_code == 200
    assert "Kathirmani" in r.text and "Console" in r.text


def test_config_exposes_upstreams():
    r = client.get("/config")
    j = r.json()
    assert set(j) >= {"grafana_url", "api_url", "review_url"}


def test_proxy_down_upstream_returns_clean_502(monkeypatch):
    """A dead upstream must surface as a structured 502, never an unhandled 500,
    so the SPA can render an 'offline' state."""
    class _Boom:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def request(self, *a, **k): raise httpx.ConnectError("refused")
    monkeypatch.setattr(console.httpx, "AsyncClient", lambda *a, **k: _Boom())
    r = client.get("/backend/cameras")
    assert r.status_code == 502
    assert r.json()["error"] == "upstream_unreachable"


def test_proxy_relays_upstream_json(monkeypatch):
    """A healthy upstream's JSON + status code are relayed through unchanged."""
    payload = [{"id": "cam_entry", "name": "Entry"}]

    class _OK:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def request(self, method, url, **k):
            assert url.endswith("/cameras")
            return httpx.Response(200, json=payload,
                                  request=httpx.Request(method, url))
    monkeypatch.setattr(console.httpx, "AsyncClient", lambda *a, **k: _OK())
    r = client.get("/backend/cameras")
    assert r.status_code == 200
    assert r.json() == payload


def test_review_proxy_forwards_post(monkeypatch):
    seen = {}

    class _OK:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def request(self, method, url, content=None, **k):
            seen["method"], seen["url"], seen["body"] = method, url, content
            return httpx.Response(200, json={"status": "confirmed"},
                                  request=httpx.Request(method, url))
    monkeypatch.setattr(console.httpx, "AsyncClient", lambda *a, **k: _OK())
    r = client.post("/review/incidents/abc/review",
                    json={"decision": "approve", "reviewer": "qa"})
    assert r.status_code == 200 and r.json()["status"] == "confirmed"
    assert seen["method"] == "POST" and seen["url"].endswith("/incidents/abc/review")
