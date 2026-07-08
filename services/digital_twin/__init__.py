"""Digital-twin reference framework (Phase 10).

A reusable, YAML-driven store digital twin: load the store→camera→FOV→zone graph
from config, validate camera↔zone relationships, and map events onto zones. A new
store onboards by YAML alone — no code change (master plan §12, spec/10 Phase 10).
"""
from .twin import StoreTwin
from .loader import load_twin

__all__ = ["StoreTwin", "load_twin"]
