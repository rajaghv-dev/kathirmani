"""Loki push client for structured inference logging.

Sends logs directly to Loki's HTTP push API with labels that make
Grafana queries contextual: filter by camera, event type, or query.
"""
import json
import time
import threading
import requests
from typing import Any

LOKI_URL = "http://localhost:3100/loki/api/v1/push"

_queue: list[dict] = []
_lock = threading.Lock()
_flush_thread: threading.Thread | None = None
_stop = threading.Event()


def _ns() -> str:
    """Current time as nanosecond Unix timestamp string (Loki format)."""
    return str(int(time.time() * 1e9))


def _push(streams: list[dict]) -> None:
    try:
        requests.post(LOKI_URL, json={"streams": streams}, timeout=3)
    except Exception:
        pass  # never block inference on log failures


def log(
    message: str,
    camera: str = "",
    event_type: str = "info",
    extra: dict[str, Any] | None = None,
) -> None:
    """Push one structured log line to Loki."""
    labels = {
        "app": "marlin",
        "store": "kathirmani",
        "event_type": event_type,
    }
    if camera:
        labels["camera"] = camera.replace('"', "'")

    body = {"msg": message}
    if extra:
        body.update(extra)

    label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    stream = {
        "stream": labels,
        "values": [[_ns(), json.dumps(body, ensure_ascii=False)]],
    }
    threading.Thread(target=_push, args=([stream],), daemon=True).start()


# ── Convenience wrappers ───────────────────────────────────────────────────

def log_inference_start(cameras: list[str]) -> None:
    log(
        f"Inference started on {len(cameras)} cameras",
        event_type="run_start",
        extra={"cameras": cameras, "camera_count": len(cameras)},
    )


def log_caption(camera: str, scene: str, events: list[dict]) -> None:
    log(
        f"Caption complete — {len(events)} event(s) detected",
        camera=camera,
        event_type="caption_complete",
        extra={
            "event_count": len(events),
            "scene": scene[:300] if scene else "",
            "events": [
                {"start": e.get("start", 0), "end": e.get("end", 0),
                 "description": e.get("description", "")}
                for e in events
            ],
        },
    )


def log_find_hit(camera: str, query: str, span: list, raw: str) -> None:
    log(
        f"Found: {query}",
        camera=camera,
        event_type="find_hit",
        extra={
            "query": query,
            "span_start_sec": span[0],
            "span_end_sec": span[1],
            "duration_sec": round(span[1] - span[0], 2),
            "raw": raw,
        },
    )


def log_find_miss(camera: str, query: str) -> None:
    log(
        f"Not found: {query}",
        camera=camera,
        event_type="find_miss",
        extra={"query": query},
    )


def log_error(camera: str, stage: str, error: str) -> None:
    log(
        f"Error in {stage}: {error[:200]}",
        camera=camera,
        event_type="error",
        extra={"stage": stage, "error": error[:500]},
    )


def log_fusion(unique_events: int, raw_events: int, best_cameras: dict[str, str]) -> None:
    log(
        f"Fusion complete — {unique_events} unique events (dedup from {raw_events})",
        event_type="fusion_complete",
        extra={
            "unique_event_count": unique_events,
            "raw_event_count": raw_events,
            "best_cameras": best_cameras,
        },
    )


def log_run_complete(videos: int, total_events: int, wall_time_sec: float) -> None:
    log(
        f"Run complete — {videos} cameras, {total_events} events, {wall_time_sec:.0f}s",
        event_type="run_complete",
        extra={
            "videos_processed": videos,
            "total_events": total_events,
            "wall_time_sec": round(wall_time_sec, 1),
        },
    )
