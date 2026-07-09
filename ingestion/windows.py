"""5-sec sliding AI windows over a segment (master plan §3.2).

Pure math, no IO — the unit-testable core. A 10-sec clip at 5-sec windows / 2-sec
stride yields windows at [0,5] [2,7] [4,9] [5,10] (the last clamped to the clip end).
"""
from __future__ import annotations

from .records import WindowRecord


def window_bounds(duration_sec: float, win_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    if win_sec <= 0 or stride_sec <= 0:
        raise ValueError("win_sec and stride_sec must be > 0")
    if duration_sec <= 0:
        return []
    out: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_sec:
        end = min(start + win_sec, duration_sec)
        out.append((round(start, 3), round(end, 3)))
        if end >= duration_sec:
            break
        start += stride_sec
    # ensure a final window flush to the clip end (covers the tail)
    last_start = round(duration_sec - win_sec, 3)
    if last_start > 0 and (not out or out[-1][0] < last_start) and out[-1][1] < duration_sec:
        out.append((max(0.0, last_start), round(duration_sec, 3)))
    return out


def windows_for_segment(seg, win_sec: float, stride_sec: float,
                        skip_before_sec: float = 0.0) -> list[WindowRecord]:
    """Windows for one clip. `skip_before_sec` drops windows lying ENTIRELY
    inside the clip's overlap prefix (already covered by the previous clip's
    tail windows); boundary-crossing windows are kept — containing them is
    the point of the overlap."""
    return [
        WindowRecord(
            segment_id=seg.segment_id, camera_id=seg.camera_id,
            window_start_sec=s, window_end_sec=e, clip_path=seg.storage_path,
        )
        for (s, e) in window_bounds(seg.duration_sec, win_sec, stride_sec)
        if e > skip_before_sec
    ]
