"""StoreTwin — the in-memory store→camera→zone graph (master plan §12).

Pure-python + numpy. Built from three YAML docs (store root, camera registry, zone
defs) by `loader.load_twin`. Holds the entities and their relationships:
  store ──has──> camera ──observes──> zone (zone_type ∈ shelf|billing_counter|
  entry|exit|queue_area|staff_only|high_value_area|blind_spot|...).
Polygons are normalized [0..1] image coords; point-in-polygon is ray-casting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    id: str
    name: str
    role: str | None = None
    enabled: bool = True


@dataclass
class Zone:
    id: str
    camera_id: str
    name: str
    zone_type: str
    polygon: list[list[float]]  # normalized [[x,y], ...]


@dataclass
class StoreTwin:
    store_id: str
    name: str
    location: str | None = None
    timezone: str = "Asia/Kolkata"
    cameras: dict[str, Camera] = field(default_factory=dict)
    zones: dict[str, Zone] = field(default_factory=dict)

    # ---- relationships ------------------------------------------------------
    def zones_of_camera(self, camera_id: str) -> list[Zone]:
        return [z for z in self.zones.values() if z.camera_id == camera_id]

    # ---- validation ---------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of hard problems (empty list == valid)."""
        problems: list[str] = []
        for z in self.zones.values():
            # every zone's camera must exist and belong to this store
            if z.camera_id not in self.cameras:
                problems.append(
                    f"zone '{z.id}' references unknown camera '{z.camera_id}'"
                )
            # polygon must be well-formed: ≥3 vertices, each an [x,y] in [0,1]
            poly = z.polygon
            if not isinstance(poly, list) or len(poly) < 3:
                problems.append(f"zone '{z.id}' polygon needs ≥3 vertices")
                continue
            for pt in poly:
                if (
                    not isinstance(pt, (list, tuple))
                    or len(pt) != 2
                    or not all(isinstance(v, (int, float)) for v in pt)
                ):
                    problems.append(f"zone '{z.id}' has a malformed vertex {pt!r}")
                    break
                if not (0.0 <= pt[0] <= 1.0 and 0.0 <= pt[1] <= 1.0):
                    problems.append(
                        f"zone '{z.id}' vertex {pt!r} outside normalized [0,1]"
                    )
                    break
        return problems

    # ---- geometry: normalized point → zones ---------------------------------
    @staticmethod
    def _point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
        """Ray-casting (even-odd) test. `poly` is [[x,y], ...] normalized."""
        pts = np.asarray(poly, dtype=float)
        n = len(pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            # does a horizontal ray from (x,y) cross edge (j→i)?
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside

    def zones_for_point(self, camera_id: str, x: float, y: float) -> list[str]:
        """Zone ids (on `camera_id`) whose polygon contains normalized (x,y)."""
        return [
            z.id
            for z in self.zones_of_camera(camera_id)
            if self._point_in_polygon(x, y, z.polygon)
        ]

    # ---- event mapping ------------------------------------------------------
    def map_event(self, event: dict) -> dict:
        """Enrich an event dict with zone_ids/zone_types it falls in.

        Looks for `camera_id` and a normalized point. The point is read from
        `x`/`y`, or `norm_x`/`norm_y`, or the center of a normalized `bbox`
        [x1,y1,x2,y2]. Returns a shallow copy with `zone_ids` + `zone_types`
        added (empty lists if no point or no match — never raises).
        """
        out = dict(event)
        cam = event.get("camera_id")
        pt = self._event_point(event)
        zone_ids: list[str] = []
        if cam in self.cameras and pt is not None:
            zone_ids = self.zones_for_point(cam, pt[0], pt[1])
        out["zone_ids"] = zone_ids
        out["zone_types"] = sorted({self.zones[z].zone_type for z in zone_ids})
        return out

    @staticmethod
    def _event_point(event: dict) -> tuple[float, float] | None:
        if "x" in event and "y" in event:
            return float(event["x"]), float(event["y"])
        if "norm_x" in event and "norm_y" in event:
            return float(event["norm_x"]), float(event["norm_y"])
        bbox = event.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return None

    # ---- summary ------------------------------------------------------------
    def summary(self) -> dict:
        """Counts: cameras, zones-by-type, and blind-spot cameras (no zone)."""
        by_type: dict[str, int] = {}
        for z in self.zones.values():
            by_type[z.zone_type] = by_type.get(z.zone_type, 0) + 1
        covered = {z.camera_id for z in self.zones.values()}
        blind = sorted(cid for cid in self.cameras if cid not in covered)
        return {
            "store_id": self.store_id,
            "cameras": len(self.cameras),
            "zones": len(self.zones),
            "zones_by_type": dict(sorted(by_type.items())),
            "blind_spot_cameras": blind,  # cameras observing no zone
        }
