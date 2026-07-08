"""Repo-root pytest setup: put the repo root and src/ on sys.path so the
platform packages (`ai_workers.*`, `services.*`, `model_plugins.*`, `ingestion`,
`db`, `benchmarks`) and the legacy `kathirmani` package (src layout) all import
without an editable install, from any invocation directory."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
