"""Phase 10 twin tests — pure config, no DB/services.

Proves: the kathirmani twin loads + validates clean; a point inside a shelf polygon
maps to that shelf zone; map_event enriches a sample event; and the DEMO second
store loads + validates with the SAME code (onboard-by-YAML, zero code change).
"""
from pathlib import Path

from loader import load_twin

ROOT = Path(__file__).resolve().parents[3]
KATHIR = ROOT / "configs" / "stores" / "kathirmani.yaml"
SECOND = ROOT / "configs" / "digital_twin" / "second_store.yaml"


def test_kathirmani_loads_and_validates():
    twin = load_twin(KATHIR)
    assert twin.store_id == "kathirmani_01"
    assert len(twin.cameras) == 5
    assert len(twin.zones) == 6
    assert twin.validate() == []  # no hard problems


def test_point_in_shelf_zone():
    twin = load_twin(KATHIR)
    # left_shelf_zone polygon spans x[0,1] y[0.2,0.9]; center point is inside it.
    zones = twin.zones_for_point("left_aisle", 0.5, 0.5)
    assert "left_shelf_zone" in zones
    # a point above the polygon (y=0.05) is outside.
    assert twin.zones_for_point("left_aisle", 0.5, 0.05) == []


def test_map_event_enriches():
    twin = load_twin(KATHIR)
    ev = {"event_type": "item_pickup", "camera_id": "left_aisle", "x": 0.5, "y": 0.5}
    out = twin.map_event(ev)
    assert out["zone_ids"] == ["left_shelf_zone"]
    assert out["zone_types"] == ["shelf"]
    # bbox-center form also works (billing counter zone).
    ev2 = {"camera_id": "bill_counter", "bbox": [0.4, 0.6, 0.6, 0.8]}
    assert "bill_counter_zone" in twin.map_event(ev2)["zone_ids"]
    # event with no point / unknown camera → empty, never raises.
    assert twin.map_event({"camera_id": "nope"})["zone_ids"] == []


def test_summary_blind_spots():
    twin = load_twin(KATHIR)
    s = twin.summary()
    assert s["cameras"] == 5
    assert s["zones_by_type"]["shelf"] == 2
    # 'center' camera has no zone defined → blind spot.
    assert "center" in s["blind_spot_cameras"]


def test_second_store_onboards_by_yaml():
    """Same StoreTwin code, different YAML → loads + validates clean."""
    twin = load_twin(SECOND)
    assert twin.store_id == "demo_store_02"
    assert len(twin.cameras) == 2
    assert len(twin.zones) == 2
    assert twin.validate() == []
    out = twin.map_event({"camera_id": "demo_aisle", "x": 0.5, "y": 0.5})
    assert out["zone_types"] == ["shelf"]
