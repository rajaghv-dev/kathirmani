"""OSS video ingestion (Phase 1): RTSP/file → 10-sec clips → 5-sec windows →
storage + metadata + queue. PyAV-based (GStreamer for live/HW later — see sources.py).
Backends are pluggable; defaults are hermetic file-first (spec/10 Phase 1)."""
from .ingest import IngestOptions, run_ingest

__all__ = ["IngestOptions", "run_ingest"]
