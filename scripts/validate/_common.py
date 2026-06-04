"""Shared helpers for per-component validators.

Each validator module exposes `checks() -> list[Check]`. Run one directly
(`python scripts/validate/<c>.py`) or all via `scripts/validate/doctor.py`.
A failed check exits non-zero; warnings (degraded but not fatal) don't.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GREEN, RED, YEL, DIM, RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    warn: bool = False          # warn=True → reported but not a hard failure


def C(name: str, ok: bool, detail: str = "", warn: bool = False) -> Check:
    return Check(name, ok, detail, warn)


def _glyph(c: Check) -> str:
    if c.ok:
        return f"{GREEN}✓{RST}"
    return f"{YEL}⚠{RST}" if c.warn else f"{RED}✗{RST}"


def cli(title: str, checks: list[Check]) -> int:
    print(f"\n{title}")
    for c in checks:
        d = f"  {DIM}{c.detail}{RST}" if c.detail else ""
        print(f"  {_glyph(c)} {c.name}{d}")
    hard = [c for c in checks if not c.ok and not c.warn]
    return 1 if hard else 0


def try_import(mod: str) -> Check:
    try:
        __import__(mod)
        return C(f"import {mod}", True)
    except Exception as e:
        return C(f"import {mod}", False, str(e)[:60])
