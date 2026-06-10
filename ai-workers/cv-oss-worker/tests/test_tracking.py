"""Tracker — deterministic multi-object IoU/centroid tracker (no GPU/weights/DB).

Asserts the persistent-identity contract the rule engine relies on:
  * the same object across successive frames keeps ONE track_id,
  * distinct co-present objects get DISTINCT ids,
  * a re-appearance after a gap (> max_age_sec) gets a NEW id.
"""
from __future__ import annotations

from base.schemas import Detection
from tracking import Tracker, iou


def _det(label: str, bbox: list[float], t: float) -> Detection:
    return Detection(label=label, confidence=0.9, bbox=bbox, frame_time_sec=t,
                     model_name="test")


def test_iou_basics():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert iou([0, 0, 1, 1], [2, 2, 1, 1]) == 0.0           # disjoint
    half = iou([0.0, 0.0, 0.2, 0.2], [0.1, 0.0, 0.2, 0.2])  # partial overlap
    assert 0.0 < half < 1.0


def test_same_object_across_frames_keeps_one_id():
    tr = Tracker()
    f0 = tr.update([_det("person", [0.40, 0.45, 0.18, 0.40], 0.0)], "left_aisle", 0.0)
    id0 = f0[0].track_id
    assert id0 is not None
    # next frame: same person drifted slightly -> high IoU -> same id
    f1 = tr.update([_det("person", [0.42, 0.46, 0.18, 0.40], 1.0)], "left_aisle", 1.0)
    assert f1[0].track_id == id0
    f2 = tr.update([_det("person", [0.44, 0.47, 0.18, 0.40], 2.0)], "left_aisle", 2.0)
    assert f2[0].track_id == id0


def test_distinct_objects_get_distinct_ids():
    tr = Tracker()
    dets = tr.update([
        _det("person", [0.40, 0.45, 0.18, 0.40], 0.0),
        _det("item",   [0.10, 0.10, 0.06, 0.06], 0.0),
    ], "left_aisle", 0.0)
    ids = {d.track_id for d in dets}
    assert len(ids) == 2, ids
    # two persons far apart in the same frame -> two ids
    dets2 = tr.update([
        _det("person", [0.05, 0.05, 0.10, 0.20], 1.0),
        _det("person", [0.80, 0.05, 0.10, 0.20], 1.0),
    ], "right_aisle", 1.0)
    assert dets2[0].track_id != dets2[1].track_id


def test_new_object_after_gap_gets_new_id():
    tr = Tracker(max_age_sec=5.0)
    a = tr.update([_det("person", [0.40, 0.45, 0.18, 0.40], 0.0)], "left_aisle", 0.0)
    id0 = a[0].track_id
    # same bbox but after a long gap (> max_age_sec) -> the old track retired -> new id
    b = tr.update([_det("person", [0.40, 0.45, 0.18, 0.40], 100.0)], "left_aisle", 100.0)
    assert b[0].track_id != id0


def test_ids_unique_across_cameras():
    tr = Tracker()
    a = tr.update([_det("person", [0.4, 0.4, 0.2, 0.4], 0.0)], "cam_a", 0.0)
    b = tr.update([_det("person", [0.4, 0.4, 0.2, 0.4], 0.0)], "cam_b", 0.0)
    # identical geometry on a different camera must NOT share the id
    assert a[0].track_id != b[0].track_id


def test_label_gates_association():
    tr = Tracker()
    a = tr.update([_det("person", [0.4, 0.4, 0.2, 0.4], 0.0)], "cam_a", 0.0)
    # a perfectly-overlapping but differently-labelled detection is a new track
    b = tr.update([_det("item", [0.4, 0.4, 0.2, 0.4], 1.0)], "cam_a", 1.0)
    assert b[0].track_id != a[0].track_id


def test_update_mutates_and_returns_same_list():
    tr = Tracker()
    dets = [_det("person", [0.4, 0.4, 0.2, 0.4], 0.0)]
    out = tr.update(dets, "cam_a", 0.0)
    assert out is dets
    assert dets[0].track_id is not None
