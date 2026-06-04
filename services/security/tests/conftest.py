"""Put the repo root on sys.path so `import services.security...` and the `db`
helper resolve without an editable install (mirrors services/api/tests)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
