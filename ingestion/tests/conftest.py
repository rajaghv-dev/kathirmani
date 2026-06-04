import sys
from pathlib import Path

# repo root on sys.path so `import ingestion` works without an editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
