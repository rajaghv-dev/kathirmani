"""Tracker — a dependency-free multi-object IoU/centroid tracker (master plan
Phase 4 follow-up: real persistent track_ids).

The OSS detector (plugin.py) emits per-window `Detection`s with no identity. This
module assigns *persistent* `track_id`s to those detections across successive
frames/windows **within a camera**, so the rule engine can reason per-subject
(dwell, billing-bypass) instead of falling back to a coarse `cam:<id>` bucket.

Design (matches the repo's "lazy + non-fatal, fully testable, no GPU/weights"
convention):

  * Pure python, no numpy/scipy/lap — a greedy IoU+centroid matcher (Hungarian is
    deliberately avoided to stay dependency-light). Deterministic given the same
    input order, so it's trivially unit-testable.
  * Per-camera state: a detection is only ever matched to a track on the same
    camera. Track ids are unique across cameras (a global counter) so they're
    safe as `track_external_id`s.
  * Greedy association by descending IoU, tie-broken by centroid distance; an
    unmatched detection spawns a new track. A track not seen for `max_age_sec`
    (wall/stream time) is retired, so a re-appearance after a gap gets a NEW id.

API: `Tracker().update(detections, camera_id, frame_time) -> detections` where
each returned Detection carries a populated `.track_id`. `update` mutates the
passed dataclasses in place (and returns the same list) for caller convenience.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


from model_plugins.base.schemas import Detection  # noqa: E402


def _to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    """[x, y, w, h] (normalized, top-left origin) -> (x1, y1, x2, y2)."""
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x, y, x + w, y + h)


def iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-union of two [x, y, w, h] boxes. Pure python."""
    ax1, ay1, ax2, ay2 = _to_xyxy(a)
    bx1, by1, bx2, by2 = _to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _centroid(bbox: list[float]) -> tuple[float, float]:
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x + w / 2.0, y + h / 2.0)


def _centroid_dist(a: list[float], b: list[float]) -> float:
    ca, cb = _centroid(a), _centroid(b)
    dx, dy = ca[0] - cb[0], ca[1] - cb[1]
    return (dx * dx + dy * dy) ** 0.5


@dataclass
class _Track:
    """One live track: identity + last-seen state for matching/retirement."""
    track_id: int
    camera_id: str
    label: str
    bbox: list[float]
    first_seen: float
    last_seen: float
    hits: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class Tracker:
    """Deterministic, dependency-free multi-object tracker.

    Construct once per worker and call `update(...)` per detection batch (one
    AI window / frame). Holds per-camera track state in memory; no GPU/DB.

    Parameters
    ----------
    iou_threshold:    minimum IoU to associate a detection with an existing track.
    max_centroid_dist: fallback gate — when IoU is below threshold but centroids
                      are within this normalized distance AND labels match, still
                      associate (handles fast/small motion where boxes barely
                      overlap). Set <=0 to disable the centroid fallback.
    max_age_sec:      a track unseen for longer than this (by frame_time) is
                      retired; a later detection then starts a NEW id.
    match_label:      only associate detections to a track of the same label.
    """

    def __init__(self, iou_threshold: float = 0.3, max_centroid_dist: float = 0.08,
                 max_age_sec: float = 5.0, match_label: bool = True) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_centroid_dist = float(max_centroid_dist)
        self.max_age_sec = float(max_age_sec)
        self.match_label = bool(match_label)
        self._tracks: dict[str, list[_Track]] = {}   # camera_id -> live tracks
        self._next_id = 1                             # global, monotonic

    # ---- public API --------------------------------------------------------
    def update(self, detections: list[Detection], camera_id: str,
               frame_time: float) -> list[Detection]:
        """Assign persistent track_ids to `detections` (mutated in place) and
        return them. Detections are matched against this camera's live tracks
        (greedy by descending IoU, centroid-distance tie-break); unmatched
        detections spawn new tracks; stale tracks are retired."""
        cam = camera_id or ""
        self._retire_stale(cam, frame_time)
        live = self._tracks.setdefault(cam, [])

        # candidate (detection_idx, track_idx, iou, centroid_dist) pairs that pass the gate
        cands: list[tuple[float, float, int, int]] = []
        for di, det in enumerate(detections):
            for ti, trk in enumerate(live):
                if self.match_label and det.label != trk.label:
                    continue
                ov = iou(det.bbox, trk.bbox)
                dist = _centroid_dist(det.bbox, trk.bbox)
                gated = ov >= self.iou_threshold or (
                    self.max_centroid_dist > 0.0 and dist <= self.max_centroid_dist)
                if gated:
                    # sort key: highest IoU first, then smallest centroid distance
                    cands.append((-ov, dist, di, ti))
        cands.sort()

        used_det: set[int] = set()
        used_trk: set[int] = set()
        assigned: dict[int, int] = {}                 # detection_idx -> track_idx
        for _neg_ov, _dist, di, ti in cands:
            if di in used_det or ti in used_trk:
                continue
            used_det.add(di)
            used_trk.add(ti)
            assigned[di] = ti

        for di, det in enumerate(detections):
            if di in assigned:
                trk = live[assigned[di]]
                trk.bbox = list(det.bbox)
                trk.last_seen = frame_time
                trk.hits += 1
            else:
                trk = _Track(
                    track_id=self._next_id,
                    camera_id=cam,
                    label=det.label,
                    bbox=list(det.bbox),
                    first_seen=frame_time,
                    last_seen=frame_time,
                )
                self._next_id += 1
                live.append(trk)
            det.track_id = str(trk.track_id)
        return detections

    def live_tracks(self, camera_id: str | None = None) -> list[_Track]:
        """Snapshot of current live tracks (all cameras, or one camera)."""
        if camera_id is not None:
            return list(self._tracks.get(camera_id or "", []))
        return [t for ts in self._tracks.values() for t in ts]

    # ---- internals ---------------------------------------------------------
    def _retire_stale(self, camera_id: str, frame_time: float) -> None:
        live = self._tracks.get(camera_id)
        if not live:
            return
        self._tracks[camera_id] = [
            t for t in live if (frame_time - t.last_seen) <= self.max_age_sec
        ]
