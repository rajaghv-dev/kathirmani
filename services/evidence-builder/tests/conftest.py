"""Test setup: put the repo root and this package dir on sys.path so
`import builder` and `from db import ...` both work without an editable install
(the dir name `evidence-builder` isn't an importable module name)."""
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]            # services/evidence-builder
_ROOT = Path(__file__).resolve().parents[3]           # repo root
for p in (str(_ROOT), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)
