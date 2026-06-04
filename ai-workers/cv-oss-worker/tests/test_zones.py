"""Zone geometry: point-in-polygon + detection→zone attribution."""
from __future__ import annotations

from zones import (ZoneMap, Zone, load_zones, map_detection_to_zones,
                   point_in_polygon)

_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_point_in_polygon_inside_outside():
    assert point_in_polygon((0.5, 0.5), _SQUARE) is True
    assert point_in_polygon((1.5, 0.5), _SQUARE) is False
    assert point_in_polygon((-0.1, 0.5), _SQUARE) is False


def test_point_in_polygon_degenerate():
    assert point_in_polygon((0.5, 0.5), [[0.0, 0.0], [1.0, 1.0]]) is False  # <3 verts


def test_triangle():
    tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert point_in_polygon((0.1, 0.1), tri) is True
    assert point_in_polygon((0.9, 0.9), tri) is False


def test_map_detection_to_zones_with_synthetic_map():
    zm = ZoneMap([Zone("z1", "camA", "Shelf", "shelf", _SQUARE)])
    # center of [x,y,w,h]=[0.4,0.4,0.2,0.2] is (0.5,0.5) -> inside z1
    assert map_detection_to_zones((0.4, 0.4, 0.2, 0.2), "camA", zm) == ["z1"]
    assert map_detection_to_zones((0.5, 0.5), "camA", zm) == ["z1"]
    assert map_detection_to_zones((0.5, 0.5), "camB", zm) == []   # other camera
    assert zm.zone_types(["z1"]) == ["shelf"]


def test_load_real_zone_config():
    zm = load_zones()                                 # repo config
    assert isinstance(zm, ZoneMap)
    # left_aisle has a shelf polygon covering most of the frame
    zids = zm.map_point_to_zones((0.5, 0.5), "left_aisle")
    assert "left_shelf_zone" in zids
    assert "shelf" in zm.zone_types(zids)
