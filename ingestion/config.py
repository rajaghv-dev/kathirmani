"""Ingestion paths + settings. Anchored on the repo root, not cwd."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"            # runtime output (gitignored)
CLIPS_DIR = DATA_DIR / "clips"
THUMBS_DIR = DATA_DIR / "thumbnails"
QUEUE_DIR = DATA_DIR / "queue"
META_DIR = DATA_DIR / "metadata"

METRICS_PORT = 8901                          # ingestion /metrics (Prometheus scrape, Phase 3)


@dataclass
class SegmentSettings:
    storage_clip_duration_sec: int = 10
    ai_window_duration_sec: int = 5
    ai_window_stride_sec: int = 2
    pre_event_context_sec: int = 5
    post_event_context_sec: int = 10
    evidence_clip_duration_sec: int = 20


@dataclass
class CameraCfg:
    id: str
    name: str
    role: str
    source_file: str | None = None
    rtsp_url_env: str | None = None
    fps_target: int = 10
    segment_sec: int = 10
    enabled: bool = True


@dataclass
class CamerasConfig:
    store_id: str
    timezone: str
    segmenting: SegmentSettings
    cameras: list[CameraCfg] = field(default_factory=list)


def load_cameras(path: Path | None = None) -> CamerasConfig:
    path = path or (CONFIGS_DIR / "cameras.yaml")
    doc = yaml.safe_load(Path(path).read_text())
    seg = SegmentSettings(**(doc.get("segmenting") or {}))
    cams = [CameraCfg(**c) for c in doc.get("cameras", [])]
    return CamerasConfig(
        store_id=doc["store_id"], timezone=doc.get("timezone", "UTC"),
        segmenting=seg, cameras=cams,
    )
