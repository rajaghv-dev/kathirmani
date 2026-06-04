"""Digital-twin reference framework (Phase 10).

A reusable, YAML-driven store digital twin: load the store→camera→FOV→zone graph
from config, validate camera↔zone relationships, and map events onto zones. A new
store onboards by YAML alone — no code change (master plan §12, spec/10 Phase 10).
"""
# The directory name is hyphenated, so this is not importable as a real package;
# the modules are imported flat (service dir on sys.path) by tests/CLI. The
# try/except keeps `from services... import` working if a path shim ever maps it.
try:  # package context
    from .twin import StoreTwin
    from .loader import load_twin
except ImportError:  # flat context: `import twin` / `import loader`
    pass

__all__ = ["StoreTwin", "load_twin"]
