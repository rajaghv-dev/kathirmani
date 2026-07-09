"""CvOssDetector — must work with no GPU, no ultralytics weights load, no DB.

Covers the 10-point plugin essentials we can run hermetically: fake_infer
shape/schema, the lifecycle methods, and the detection→event mapping yielding
a schema-valid §8.3 CommonEvent flagged for VLM verification.
"""
from __future__ import annotations

from dataclasses import fields

from model_plugins.base.schemas import CommonEvent, Detection
from ai_workers.cv_oss_worker.plugin import CvOssDetector, default_config


def _window() -> dict:
    return {
        "type": "ai_window.ready",
        "window_id": "11111111-1111-1111-1111-111111111111",
        "segment_id": "22222222-2222-2222-2222-222222222222",
        "camera_id": "left_aisle",                    # has a `shelf` zone
        "window_start_sec": 0.0,
        "window_end_sec": 5.0,
        "clip_path": "/nonexistent/clip.mp4",         # forces fake path in infer()
        "store_id": "kathirmani_01",
    }


def test_fake_infer_returns_valid_detections():
    p = CvOssDetector()
    out = p.fake_infer(_window())
    assert set(out) >= {"detections", "events", "model_run"}
    assert out["detections"], "fake_infer must emit detections"
    det_keys = {f.name for f in fields(Detection)}
    for d in out["detections"]:
        # round-trips into the Detection dataclass (schema-valid)
        Detection(**{k: d[k] for k in det_keys})
        assert 0.0 <= d["confidence"] <= 1.0
        assert len(d["bbox"]) == 4
    assert out["model_run"]["faked"] is True


def test_infer_falls_back_to_fake_without_model():
    p = CvOssDetector()
    out = p.infer(_window())                           # no weights/clip -> fake path
    assert out["detections"]
    assert out["model_run"]["faked"] is True
    assert "latency_ms" in out["model_run"]


def test_detection_to_event_is_schema_valid_common_event():
    p = CvOssDetector()
    out = p.infer(_window())
    assert out["events"], "person-in-shelf + item must raise a hypothesis"
    ev = out["events"][0]
    ev_keys = {f.name for f in fields(CommonEvent)}
    obj = CommonEvent(**{k: ev[k] for k in ev_keys})   # schema-valid round-trip
    assert obj.event_type == "suspicious_item_interaction"
    assert obj.needs_vlm_verification is True
    assert obj.source_engine == "cv_oss_detector"
    assert obj.severity == "medium"
    assert obj.zones, "event must carry the matched shelf zone id(s)"


def test_no_event_when_no_shelf_zone():
    p = CvOssDetector()
    w = _window()
    w["camera_id"] = "bill_counter"                    # billing/queue zones, no shelf
    out = p.infer(w)
    assert out["events"] == []


def test_lifecycle_and_health_metrics():
    p = CvOssDetector()
    p.load()                                           # non-fatal even without weights
    h = p.health()
    assert h.ok is True                                # fake-infer mode is still healthy
    m = p.metrics()
    assert set(m) >= {"requests_total", "fake_infer_total", "model_loaded"}
    p.infer(_window())
    assert p.metrics()["requests_total"] >= 1.0
    p.unload()
    assert p.metrics()["model_loaded"] == 0.0


def test_detections_carry_track_id():
    """Every emitted detection gets a persistent track_id from the Tracker."""
    p = CvOssDetector()
    out = p.infer(_window())
    assert out["detections"]
    for d in out["detections"]:
        assert d.get("track_id"), "detection must carry a track_id"


def test_event_carries_track_ids_and_persist_across_windows():
    """The suspicious-item event carries real track_ids, and the same camera's
    subject keeps its id across successive windows (persistent identity)."""
    p = CvOssDetector()
    w0 = _window()
    out0 = p.infer(w0)
    assert out0["events"]
    tids0 = out0["events"][0]["track_ids"]
    assert tids0, "event must carry track_ids"
    # person id is stable across a second window on the same camera
    w1 = _window()
    w1["window_start_sec"] = 1.0
    out1 = p.infer(w1)
    person0 = next(d["track_id"] for d in out0["detections"] if d["label"] == "person")
    person1 = next(d["track_id"] for d in out1["detections"] if d["label"] == "person")
    assert person0 == person1, "same person should keep its track_id across windows"


def test_default_config_matches_detection_task():
    cfg = default_config()
    assert cfg.task == "detection"
    assert cfg.plugin == "cv_oss_detector"
    assert cfg.runtime == "ultralytics"
    assert cfg.model_id == "jameslahm/yoloe-11s-seg"


def test_one_event_per_subject_track_not_per_frame_appearance():
    """Real windows carry the SAME person once per decoded frame (~125×). The
    hypothesis must be per subject (track_id) per window, not per appearance —
    the pre-fix behaviour flooded ~600 duplicate events per busy window."""
    p = CvOssDetector()
    mn = p.config.model_id
    dets = []
    for frame in range(50):                            # same two subjects, 50 frames
        t = frame / 10.0
        dets.append(Detection(label="person", confidence=0.9,
                              bbox=[0.40, 0.45, 0.18, 0.40],
                              frame_time_sec=t, model_name=mn, track_id="p1"))
        dets.append(Detection(label="person", confidence=0.9,
                              bbox=[0.42, 0.45, 0.18, 0.40],
                              frame_time_sec=t, model_name=mn, track_id="p2"))
        dets.append(Detection(label="item", confidence=0.8,
                              bbox=[0.46, 0.55, 0.06, 0.06],
                              frame_time_sec=t, model_name=mn, track_id="i1"))
    req = {"window_id": "w-1", "segment_id": "s-1", "camera_id": "left_aisle",
           "store_id": "kathirmani_01", "clip_path": "/nonexistent/clip.mp4"}
    events = p.detections_to_events(dets, req)
    assert len(events) == 2, "one event per subject track, not per frame"
    assert len({e.event_id for e in events}) == 2, "distinct ids per subject"
    for e in events:
        assert e.objects == ["item", "person"], "objects deduped + sorted"


def test_motion_changed_fraction_gate_boundary():
    """Frame gate math: identical frames -> 0 change (skip); a burst that moves
    >15% of pixels by more than the delta threshold -> detect."""
    import numpy as np
    from ai_workers.cv_oss_worker.plugin import (MOTION_MIN_CHANGED,
                                                 motion_changed_fraction)
    base = np.full((90, 160), 100, dtype=np.uint8)
    assert motion_changed_fraction(base, base.copy()) == 0.0

    # 10% of pixels change hard -> below the 15% default -> would be skipped
    minor = base.copy()
    minor[:9, :] = 200                                 # 9/90 rows = 10%
    frac = motion_changed_fraction(base, minor)
    assert 0.09 < frac < 0.11 and frac < MOTION_MIN_CHANGED

    # 20% of pixels change hard -> above threshold -> would be detected
    major = base.copy()
    major[:18, :] = 200                                # 18/90 rows = 20%
    frac = motion_changed_fraction(base, major)
    assert frac > MOTION_MIN_CHANGED

    # sub-delta wiggle (noise) never counts, no matter how many pixels
    noisy = base + np.uint8(5)                         # delta 5 < pixel_delta 25
    assert motion_changed_fraction(base, noisy) == 0.0
