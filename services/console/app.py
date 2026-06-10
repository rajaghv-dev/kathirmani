"""Kathirmani Console — the single front door (BFF / gateway) for the platform.

The platform is a set of cooperating services (the read API on :8000, the human
review UI on :8010, Grafana on :3000, the search pipeline). Operators previously had
to know each port and surface. This service unifies them: it serves ONE cohesive
single-page console and **proxies** to the upstream services server-side, so the
browser talks to a single origin (no CORS, auth header passthrough, one URL to share).

Routes:
  GET  /                      -> the SPA shell (static/index.html)
  GET  /static/*              -> SPA assets
  GET  /config                -> upstream URLs the SPA needs (e.g. Grafana embed)
  GET  /healthz               -> console + best-effort upstream reachability
  *    /backend/{path}        -> proxied to the platform API   (API_URL)
  *    /review/{path}         -> proxied to the review UI       (REVIEW_URL)

Upstreams are env-configurable so the same console works in dev and prod:
  CONSOLE_API_URL     (default http://localhost:8000)
  CONSOLE_REVIEW_URL  (default http://localhost:8010)
  CONSOLE_GRAFANA_URL (default http://localhost:3000)

Run: `make run-console`  (uvicorn services.console.app:app --port 8080).
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

_STATIC = Path(__file__).resolve().parent / "static"

API_URL = os.environ.get("CONSOLE_API_URL", "http://localhost:8000").rstrip("/")
REVIEW_URL = os.environ.get("CONSOLE_REVIEW_URL", "http://localhost:8010").rstrip("/")
GRAFANA_URL = os.environ.get("CONSOLE_GRAFANA_URL", "http://localhost:3000").rstrip("/")

# Headers we forward upstream (auth passthrough) and never copy back verbatim.
_FORWARD_REQ_HEADERS = ("authorization", "x-api-key", "content-type", "accept")
_DROP_RESP_HEADERS = ("content-encoding", "content-length", "transfer-encoding", "connection")

app = FastAPI(title="Kathirmani Console", version="0.1.0")

app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/config")
def config():
    """What the SPA needs to know about the deployment (e.g. where to embed Grafana)."""
    return {"grafana_url": GRAFANA_URL, "api_url": API_URL, "review_url": REVIEW_URL}


async def _proxy(request: Request, base: str, path: str) -> JSONResponse:
    """Forward a request to an upstream service and relay its JSON response. A down
    upstream becomes a clean 502 JSON (never an unhandled 500) so the SPA can render
    an 'offline' state instead of breaking."""
    url = f"{base}/{path.lstrip('/')}"
    fwd = {k: v for k, v in request.headers.items() if k.lower() in _FORWARD_REQ_HEADERS}
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.request(request.method, url, params=request.query_params,
                                     content=body or None, headers=fwd)
    except httpx.HTTPError as e:
        return JSONResponse(
            {"error": "upstream_unreachable", "upstream": base, "detail": str(e)},
            status_code=502)
    headers = {k: v for k, v in r.headers.items() if k.lower() not in _DROP_RESP_HEADERS}
    media = r.headers.get("content-type", "application/json")
    return JSONResponse(content=_safe_json(r), status_code=r.status_code,
                        headers={"x-upstream": base}) if "json" in media else \
        JSONResponse({"raw": r.text}, status_code=r.status_code, headers=headers)


def _safe_json(r: httpx.Response):
    try:
        return r.json()
    except Exception:
        return {"raw": r.text}


@app.api_route("/backend/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def backend(path: str, request: Request):
    return await _proxy(request, API_URL, path)


@app.api_route("/review/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def review(path: str, request: Request):
    return await _proxy(request, REVIEW_URL, path)


@app.get("/healthz")
async def healthz():
    """Console is up; report best-effort reachability of each upstream."""
    out = {"console": "ok", "upstreams": {}}
    async with httpx.AsyncClient(timeout=3.0) as client:
        for name, base in (("api", API_URL), ("review", REVIEW_URL), ("grafana", GRAFANA_URL)):
            try:
                r = await client.get(f"{base}/health" if name != "grafana"
                                     else f"{base}/api/health")
                out["upstreams"][name] = {"ok": r.status_code < 500, "status": r.status_code}
            except Exception as e:
                out["upstreams"][name] = {"ok": False, "error": type(e).__name__}
    return out
