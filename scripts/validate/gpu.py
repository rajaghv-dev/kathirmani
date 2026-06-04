#!/usr/bin/env python3
"""Validate the GPU/runtime component (driver, Docker, NVIDIA Docker runtime)."""
from __future__ import annotations

import shutil
import subprocess

from _common import C, cli


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        return None


def checks():
    out = []
    smi = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    out.append(C("nvidia-smi / GPU", bool(smi and smi.returncode == 0),
                 (smi.stdout.strip() if smi and smi.returncode == 0 else "no GPU/driver"), warn=not smi))
    out.append(C("docker CLI", shutil.which("docker") is not None, warn=True))
    info = _run(["docker", "info", "--format", "{{json .Runtimes}}"])
    has_nv = bool(info and "nvidia" in (info.stdout or ""))
    out.append(C("Docker NVIDIA runtime registered", has_nv,
                 "run: make setup-nvidia-docker" if not has_nv else "", warn=True))
    out.append(C("nvidia-ctk present", shutil.which("nvidia-ctk") is not None, warn=True))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("gpu — driver + Docker NVIDIA runtime", checks()))
