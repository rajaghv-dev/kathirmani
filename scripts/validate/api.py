#!/usr/bin/env python3
"""Validate the API component (FastAPI installed; app importable; /health if running)."""
from __future__ import annotations

from _common import C, ROOT, cli, try_import


def checks():
    out = [try_import("fastapi"), try_import("uvicorn")]
    app_py = ROOT / "services" / "api" / "app.py"
    out.append(C("services/api/app.py present", app_py.exists()))
    # in-process health via TestClient (no running server needed)
    try:
        from fastapi.testclient import TestClient
        from services.api.app import app
        r = TestClient(app).get("/health")
        out.append(C("API /health (in-process)", r.status_code == 200, f"HTTP {r.status_code}"))
    except Exception as e:
        out.append(C("API /health (in-process)", False, str(e)[:60]))
    # live server (optional)
    try:
        import requests
        requests.get("http://localhost:8000/health", timeout=2)
        out.append(C("API live on :8000", True))
    except Exception:
        out.append(C("API live on :8000", False, "not running — make api", warn=True))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("api — FastAPI platform surface", checks()))
