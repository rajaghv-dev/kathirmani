"""Put the repo root and this package dir on sys.path so `import app` works without
an editable install (mirrors the other service test packages)."""
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]            # services/console
_ROOT = Path(__file__).resolve().parents[3]           # repo root
for p in (str(_ROOT), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)
