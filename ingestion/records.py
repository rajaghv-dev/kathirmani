"""Ingestion records + the §8 wire messages they serialize to."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SegmentRecord:
    camera_id: str
    store_id: str
    index: int
    start_time: str            # ISO8601
    end_time: str
    duration_sec: float
    storage_path: str          # where the clip lives (local path or s3://...)
    codec: str = ""
    fps: float = 0.0
    width: int = 0
    height: int = 0
    checksum: str = ""         # sha256 — evidence integrity (spec/10 trust)
    thumbnail_path: str | None = None
    health: dict[str, float] = field(default_factory=dict)
    status: str = "ready"
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_message(self) -> dict[str, Any]:
        """video_segment.ready (master plan §8.1)."""
        return {
            "type": "video_segment.ready",
            "segment_id": self.segment_id,
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "storage_path": self.storage_path,
        }

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WindowRecord:
    segment_id: str
    camera_id: str
    window_start_sec: float
    window_end_sec: float
    clip_path: str
    status: str = "pending"
    window_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_message(self) -> dict[str, Any]:
        """ai_window.ready (master plan §8.2)."""
        return {
            "type": "ai_window.ready",
            "window_id": self.window_id,
            "segment_id": self.segment_id,
            "camera_id": self.camera_id,
            "window_start_sec": self.window_start_sec,
            "window_end_sec": self.window_end_sec,
            "clip_path": self.clip_path,
        }

    def to_row(self) -> dict[str, Any]:
        return asdict(self)
