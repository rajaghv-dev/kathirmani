#!/usr/bin/env python3
"""Validate the observability stack (service reachability — warns if down)."""
from __future__ import annotations

from _common import C, cli

ENDPOINTS = [
    ("Prometheus", "http://localhost:9090/-/healthy"),
    ("Grafana", "http://localhost:3000/api/health"),
    ("Loki", "http://localhost:3100/ready"),
    ("Netdata", "http://localhost:19999/api/v1/info"),
    ("serve_metrics", "http://localhost:8900/metrics"),
]


def checks():
    import requests
    out = []
    for name, url in ENDPOINTS:
        try:
            r = requests.get(url, timeout=3)
            out.append(C(name, r.status_code < 500, f"HTTP {r.status_code}", warn=r.status_code >= 500))
        except Exception:
            out.append(C(name, False, "down — bash start_stack.sh", warn=True))
    # renderer (via Grafana render of a 1px image)
    try:
        r = requests.get("http://localhost:3000/render/d-solo/marlin-model-perf/x?panelId=1&width=10&height=10",
                         auth=("admin", "admin"), timeout=15)
        out.append(C("Grafana image renderer", r.headers.get("content-type", "").startswith("image"),
                     "renderer sidecar not wired" if not r.headers.get("content-type", "").startswith("image") else "",
                     warn=True))
    except Exception:
        out.append(C("Grafana image renderer", False, "renderer down", warn=True))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("observability — Prometheus/Grafana/Loki/Netdata/renderer", checks()))
