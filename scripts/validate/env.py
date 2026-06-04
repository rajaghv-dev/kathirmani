#!/usr/bin/env python3
"""Validate the base Python env + shared libs."""
from __future__ import annotations

import sys

from _common import C, cli, try_import


def checks():
    out = [C(f"python {sys.version.split()[0]}", sys.version_info >= (3, 10),
             "need >=3.10")]
    out.append(C("running in a venv", sys.prefix != sys.base_prefix,
                 "activate .venv", warn=True))
    for m in ("yaml", "numpy", "prometheus_client", "requests", "PIL"):
        out.append(try_import(m))
    return out


if __name__ == "__main__":
    raise SystemExit(cli("env — base Python + shared libs", checks()))
