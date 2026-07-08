"""evidence_builder tests — pure assembly, sha256 determinism, manifest-only
degradation, and the injected/no-DB build path. NO GPU / real video / DB."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from services.evidence_builder import builder


# ---- synthetic rows --------------------------------------------------------
T0 = datetime(2026, 6, 3, 10, 0, 0, tzinfo=timezone.utc)


def _event(eid, secs, etype="suspicious_item_interaction", conf=0.6):
    return {"id": eid, "store_id": "kathirmani_01", "camera_id": "shelf_a",
            "segment_id": f"seg-{eid}", "event_type": etype, "severity": "warn",
            "confidence": conf, "source_engine": "rule_engine",
            "event_time": (T0.replace(second=secs)).isoformat()}


def _obs(oid, eid, secs, verdict="suspicious", conf=0.81):
    return {"id": oid, "event_id": eid, "model_name": "nemotron-vl",
            "prompt_version": "retail_loss_v1",
            "parsed_json": {"verdict": verdict, "confidence": conf},
            "parse_success": True,
            "created_at": T0.replace(second=secs).isoformat()}


# ---- timeline --------------------------------------------------------------
def test_timeline_orders_events_and_observations_by_time():
    events = [_event("e2", 20), _event("e1", 5)]
    obs = [_obs("o1", "e1", 8), _obs("o2", "e2", 25)]
    tl = builder.assemble_timeline(events, obs)
    assert [i["ts"] for i in tl] == sorted(i["ts"] for i in tl)
    kinds = [i["kind"] for i in tl]
    assert kinds.count("event") == 2 and kinds.count("vlm_observation") == 2
    # first entry is the earliest event
    assert tl[0]["kind"] == "event" and tl[0]["event_id"] == "e1"


def test_timeline_handles_missing_timestamps_and_string_parsed_json():
    events = [_event("e1", 5)]
    events[0]["event_time"] = None                    # no usable time -> sorts last
    obs = [{"id": "o1", "event_id": "e1", "model_name": "m",
            "parsed_json": json.dumps({"verdict": "normal", "confidence": 0.2}),
            "parse_success": True, "created_at": T0.isoformat()}]
    tl = builder.assemble_timeline(events, obs)
    assert tl[-1]["ts"] is None                        # the timeless event is last
    vlm = [i for i in tl if i["kind"] == "vlm_observation"][0]
    assert vlm["verdict"] == "normal" and vlm["confidence"] == 0.2


def test_timeline_empty_inputs():
    assert builder.assemble_timeline([], []) == []
    assert builder.assemble_timeline(None, None) == []


# ---- sha256 ----------------------------------------------------------------
def test_sha256_bytes_deterministic():
    a = builder.sha256_bytes(b"evidence-bytes")
    b = builder.sha256_bytes(b"evidence-bytes")
    assert a == b and len(a) == 64
    assert a != builder.sha256_bytes(b"evidence-bytez")


def test_sha256_file_matches_bytes(tmp_path):
    p = tmp_path / "clip.bin"
    data = b"\x00\x01\x02clip" * 1000
    p.write_bytes(data)
    assert builder.sha256_file(p) == builder.sha256_bytes(data)


# ---- stitch degradation (no PyAV decode needed: files absent) --------------
def test_stitch_missing_clips_is_manifest_only(tmp_path):
    segs = [{"id": "s1", "storage_path": str(tmp_path / "nope.mp4"),
             "start_time": T0.isoformat(), "end_time": T0.replace(second=10).isoformat()}]
    res = builder.stitch_clip(segs, T0.replace(second=5).isoformat(),
                              tmp_path / "out.mp4")
    assert res["ok"] is False
    assert res["sha256"] is None
    assert "manifest-only" in res["reason"]
    assert not (tmp_path / "out.mp4").exists()


def test_stitch_no_segments_at_all(tmp_path):
    res = builder.stitch_clip([], T0.isoformat(), tmp_path / "out.mp4")
    assert res["ok"] is False and res["sha256"] is None


# ---- build_package / build_evidence (injected, no DB) ----------------------
def test_build_package_manifest_only_with_synthetic_rows(tmp_path):
    incident = {"id": "inc-1", "title": "possible_loss", "incident_type": "possible_loss",
                "severity": "warn", "store_id": "kathirmani_01",
                "start_time": T0.isoformat()}
    events = [_event("e1", 5), _event("e2", 12)]
    obs = [_obs("o1", "e1", 8, verdict="suspicious", conf=0.9)]
    segs = [{"id": "seg-e1", "storage_path": str(tmp_path / "missing.mp4"),
             "start_time": T0.isoformat(), "checksum": "abc123"}]
    pkg = builder.build_package(incident, events, obs, segs, out_dir=tmp_path)
    assert pkg["manifest_only"] is True
    assert pkg["evidence_path"] is None
    assert pkg["chain_of_custody"]["evidence_sha256"] is None
    # source segment checksums are recorded for tamper detection
    assert pkg["chain_of_custody"]["source_segments"][0]["checksum"] == "abc123"
    assert pkg["counts"] == {"events": 2, "observations": 1, "segments": 1}
    # headline verdict = latest VLM observation
    assert pkg["headline_verdict"]["verdict"] == "suspicious"
    assert len(pkg["timeline"]) == 3
    # window centered on the earliest event time (anchor)
    assert pkg["window"]["anchor"] == events[0]["event_time"]


def test_build_evidence_injected_mode_no_db(tmp_path):
    pkg = builder.build_evidence(
        "inc-2", out_dir=tmp_path,
        incident={"id": "inc-2", "title": "t", "store_id": "s1"},
        events=[_event("e1", 5)], observations=[], segments=[])
    assert pkg["persisted"] is False               # no conn -> not persisted
    assert pkg["incident_id"] == "inc-2"
    assert pkg["manifest_only"] is True
    # the whole package is JSON-serializable (it goes into incidents.summary)
    assert json.loads(json.dumps(pkg))["incident_id"] == "inc-2"


def test_build_evidence_db_unavailable_returns_empty_package():
    # No conn injected, no real DB reachable in the hermetic test env -> empty
    # manifest package rather than an exception.
    pkg = builder.build_evidence("00000000-0000-0000-0000-000000000000")
    assert pkg["persisted"] is False
    assert pkg["manifest_only"] is True
    assert pkg["timeline"] == []
