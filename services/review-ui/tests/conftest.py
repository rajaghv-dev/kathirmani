"""Test setup: put the repo root and this package dir on sys.path so
`import app` and `from db import ...` both work without an editable install
(the dir name `review-ui` isn't an importable module name)."""
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]            # services/review-ui
_ROOT = Path(__file__).resolve().parents[3]           # repo root
for p in (str(_ROOT), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)
