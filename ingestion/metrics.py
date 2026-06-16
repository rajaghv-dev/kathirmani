"""Ingestion Prometheus metrics (master plan §13.1 ingestion + camera-health).

Exposed on METRICS_PORT during a run; Prometheus scrape wiring is Phase 3. Names are
the `ingest_*` family (sibling to the `kathirmani_*` model metrics).
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_CAM = ["store_id", "camera_id"]

segments_written_total = Counter("ingest_segments_written_total", "Clips written", _CAM)
windows_created_total = Counter("ingest_windows_created_total", "AI windows created", _CAM)
frames_decoded_total = Counter("ingest_frames_decoded_total", "Frames decoded", _CAM)
segment_write_latency = Histogram("ingest_segment_write_latency_seconds",
                                   "Per-clip segment+store latency", _CAM,
                                   buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
rtsp_connected = Gauge("ingest_rtsp_connected", "Source open (1) / failed (0)", _CAM)
errors_total = Counter("ingest_errors_total", "Ingestion errors", _CAM + ["stage"])

camera_health_score = Gauge("ingest_camera_health_score", "Composite health 0..1", _CAM)
camera_black_frame_score = Gauge("ingest_camera_black_frame_score", "Fraction dark", _CAM)
camera_blur_score = Gauge("ingest_camera_blur_score", "Blur 0..1 (1=blurry)", _CAM)
camera_freeze_score = Gauge("ingest_camera_freeze_score", "Freeze 0..1 (1=frozen)", _CAM)
camera_fps_actual = Gauge("ingest_camera_fps_actual", "Observed fps", _CAM)


def serve(port: int) -> None:
    start_http_server(port)
