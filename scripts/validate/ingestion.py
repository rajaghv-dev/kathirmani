#!/usr/bin/env python3
"""Validate the ingestion component (PyAV + package + configs)."""
from __future__ import annotations

from pathlib import Path

from _common import C, ROOT, cli, try_import


def checks():
    out = [try_import("av")]
    try:
        from ingestion.windows import window_bounds
        w = window_bounds(10, 5, 2)
        out.append(C("ingestion package imports", True))
        out.append(C("window math (10s/5s/2s)", w[-1][1] == 10.0, str(w)))
    except Exception as e:
        out.append(C("ingestion package imports", False, str(e)[:60]))
    cams = ROOT / "configs" / "cameras.yaml"
    out.append(C("configs/cameras.yaml present", cams.exists()))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("ingestion — PyAV segmenter", checks()))
