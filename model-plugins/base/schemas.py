"""Task contracts + wire schemas (master plan A2.2 + §8).

Contracts are keyed by **task**, not by model — so any model that fills the task
slot is swappable. Plugins consume/produce these shapes; the queue + DB speak the
§8 JSON. Kept as dataclasses for typing; serialize to plain dicts on the wire.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- Task contracts (A2.2): input -> output per task -------------------------

TASK_CONTRACTS: dict[str, dict[str, str]] = {
    "detection":          {"input": "frames|clip_path", "output": "detections[]"},
    "tracking":           {"input": "detections+frames", "output": "tracks[]"},
    "embedding":          {"input": "clip|text|image|event", "output": "vector+metadata"},
    "vlm_clip_reasoning": {"input": "clip_path+prompt+hypothesis", "output": "structured_json"},
    "summarization":      {"input": "clips|events|time_range", "output": "timestamped_summary"},
    "search_critic":      {"input": "query+candidates", "output": "reranked_results"},
    # research/optional
    "digital_twin_simulation": {"input": "world_state+scenario", "output": "prediction"},
}


# ---- Wire schemas (§8) — what crosses the queue/DB boundary ------------------

@dataclass
class VideoSegmentReady:                       # §8.1
    segment_id: str
    store_id: str
    camera_id: str
    start_time: str
    end_time: str
    storage_path: str
    type: str = "video_segment.ready"


@dataclass
class AIWindowReady:                            # §8.2
    window_id: str
    segment_id: str
    camera_id: str
    window_start_sec: float
    window_end_sec: float
    clip_path: str
    type: str = "ai_window.ready"


@dataclass
class CommonEvent:                              # §8.3 — the pluggable event schema
    event_id: str
    store_id: str
    camera_id: str
    event_type: str
    severity: str = "info"
    confidence: float = 0.0
    source_engine: str = ""
    segment_id: str | None = None
    ai_window_id: str | None = None
    objects: list[str] = field(default_factory=list)
    track_ids: list[str] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    needs_vlm_verification: bool = False
    evidence_path: str | None = None


@dataclass
class VLMVerification:                          # §8.4 — VLMClipReasoningPlugin output
    verdict: str                                # e.g. suspicious | benign | unclear
    confidence: float
    observed_actions: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    recommended_next_step: str = ""
    structured_event_type: str = ""
    explanation: str = ""


@dataclass
class Detection:                                # DetectionPlugin output element
    label: str
    confidence: float
    bbox: list[float]                           # [x, y, w, h] normalized
    frame_time_sec: float
    model_name: str = ""


KNOWN_TASKS = set(TASK_CONTRACTS)
