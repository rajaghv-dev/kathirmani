import sys
from pathlib import Path

# repo root on sys.path so `import importlib` of the hyphenated package works,
# and so config paths resolve. The service dir has a hyphen, so tests import the
# twin/loader modules by file location via the package added below.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "services" / "digital-twin"))
sys.path.insert(0, str(ROOT))
