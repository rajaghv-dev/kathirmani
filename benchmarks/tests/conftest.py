"""Put the repo root on sys.path so `import benchmarks...` works without an
editable install, from either the repo root or this directory."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]   # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
