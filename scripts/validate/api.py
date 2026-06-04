#!/usr/bin/env python3
"""Validate the API component (FastAPI installed; app importable; /health if running)."""
from __future__ import annotations

from _common import C, ROOT, cli, try_import


def checks():
    out = [try_import("fastapi"), try_import("uvicorn")]
    app = ROOT / "services" / "api" / "app.py"
    out.append(C("services/api/app.py present", app.exists(),
                 "Phase 2 in progress" if not app.exists() else "", warn=not app.exists()))
    try:
        import requests
        r = requests.get("http://localhost:8000/health", timeout=3)
        out.append(C("API /health", r.status_code == 200, f"HTTP {r.status_code}", warn=True))
    except Exception:
        out.append(C("API /health", False, "not running — make api", warn=True))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("api — FastAPI platform surface", checks()))
