"""Test setup: put the repo root, model-plugins, and this package dir on
sys.path so `import plugin`, `import host`, `import worker`, `from base...`,
and `from ingestion...` all work without an editable install."""
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]            # ai-workers/vlm-worker
_ROOT = Path(__file__).resolve().parents[3]           # repo root
for p in (str(_ROOT), str(_ROOT / "model-plugins"), str(_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)
