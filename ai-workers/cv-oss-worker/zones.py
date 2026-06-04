"""Zone geometry: load polygons + point-in-polygon (master plan §6.3).

Zones are normalized [0..1] image polygons, one set per camera
(`configs/zones/kathirmani_zones.yaml`). A detection's bbox-center is tested
against every polygon of its camera with a pure-python ray-casting test, so
the worker can attribute detections to shelves/billing/entry/exit without any
heavy GIS dependency. Loading is best-effort: a missing/unreadable config
yields an empty map (the worker still runs, just emits no zone tags).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Default config path, relative to repo root (this file is 2 levels deep under
# ai-workers/cv-oss-worker/). Override-able via load_zones(path=...).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZONES_PATH = _REPO_ROOT / "configs" / "zones" / "kathirmani_zones.yaml"


@dataclass
class Zone:
    id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: list[list[float]]            # [[x, y], ...] normalized [0..1]


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    """Ray-casting (even-odd) test. `point` is (x, y); `polygon` is a list of
    [x, y] vertices (implicitly closed). Pure python — no numpy required."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        # does the horizontal ray at y cross the edge (i, j)?
        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


class ZoneMap:
    """Camera -> zones lookup with detection→zone attribution."""

    def __init__(self, zones: Iterable[Zone], store_id: str = "") -> None:
        self.store_id = store_id
        self.zones: list[Zone] = list(zones)
        self._by_camera: dict[str, list[Zone]] = {}
        for z in self.zones:
            self._by_camera.setdefault(z.camera_id, []).append(z)

    def zones_for_camera(self, camera_id: str) -> list[Zone]:
        return self._by_camera.get(camera_id, [])

    def map_point_to_zones(self, point: tuple[float, float], camera_id: str) -> list[str]:
        """Return ids of every zone (on this camera) containing the point."""
        return [z.id for z in self.zones_for_camera(camera_id)
                if point_in_polygon(point, z.polygon)]

    def zone_types(self, zone_ids: list[str]) -> list[str]:
        """Resolve zone ids -> their zone_type strings (e.g. 'shelf')."""
        by_id = {z.id: z.zone_type for z in self.zones}
        return [by_id[zid] for zid in zone_ids if zid in by_id]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    """bbox is [x, y, w, h] (normalized, top-left origin) -> center (cx, cy)."""
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return (x + w / 2.0, y + h / 2.0)


def load_zones(path: str | Path | None = None) -> ZoneMap:
    """Load the zone config into a ZoneMap. Best-effort: any failure
    (missing file, no PyYAML) returns an empty map so the worker still runs."""
    p = Path(path) if path else DEFAULT_ZONES_PATH
    try:
        import yaml
        with open(p) as fh:
            doc = yaml.safe_load(fh) or {}
    except Exception:
        return ZoneMap([], store_id="")
    zones = [
        Zone(
            id=z["id"],
            camera_id=z.get("camera_id", ""),
            name=z.get("name", z["id"]),
            zone_type=z.get("zone_type", ""),
            polygon=[[float(c[0]), float(c[1])] for c in z.get("polygon", [])],
        )
        for z in doc.get("zones", [])
    ]
    return ZoneMap(zones, store_id=doc.get("store_id", ""))


def map_detection_to_zones(
    bbox_center: tuple[float, float],
    camera_id: str,
    zone_map: ZoneMap | None = None,
) -> list[str]:
    """Public helper (deliverable §2): map a bbox-center to zone ids.

    `bbox_center` may be a center tuple OR a full [x, y, w, h] bbox — both are
    accepted for convenience. `zone_map` defaults to the on-disk config.
    """
    zm = zone_map or load_zones()
    pt = bbox_center
    if len(pt) == 4:                       # tolerate a full bbox being passed
        pt = _bbox_center(list(pt))
    return zm.map_point_to_zones((pt[0], pt[1]), camera_id)
