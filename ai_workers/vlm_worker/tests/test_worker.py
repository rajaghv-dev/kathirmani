"""Worker loop — hermetic (FileQueue, no Postgres). DB path behind skipif."""
from __future__ import annotations

import json

import pytest

from ai_workers.vlm_worker import worker
from ai_workers.vlm_worker.host import load_plugin


def _has_db() -> bool:
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _flagged_event() -> dict:
    return {
        "type": "event.created",
        "event_id": "44444444-4444-4444-4444-444444444444",
        "store_id": "kathirmani_01",
        "camera_id": "left_aisle",
        "event_type": "suspicious_item_interaction",
        "severity": "medium",
        "needs_vlm_verification": True,
        "evidence_path": "/nonexistent/clip.mp4",
    }


def test_skips_unflagged_event():
    from ingestion.bus import FileQueue
    import tempfile
    from pathlib import Path

    q = FileQueue(Path(tempfile.mkdtemp()))
    p = load_plugin(None)
    ev = _flagged_event()
    ev["needs_vlm_verification"] = False
    s = worker.process_event(ev, p, q, conn=None)
    assert s["skipped"] is True
    q.close()


def test_process_event_no_db():
    """process_event runs the plugin on a flagged event with no DB / FileQueue."""
    from ingestion.bus import FileQueue
    import tempfile
    from pathlib import Path

    q = FileQueue(Path(tempfile.mkdtemp()))
    p = load_plugin(None)
    s = worker.process_event(_flagged_event(), p, q, conn=None)
    assert s["skipped"] is False
    assert s["verdict"] in {"suspicious", "benign", "unclear"}
    assert s["parse_success"] is True
    assert s["faked"] is True                              # no weights
    assert s["observation_written"] is False               # no DB
    q.close()


def test_run_once_file_backend(tmp_path, monkeypatch):
    """run(once=True) drains event.created, verifies flagged ones, skips others,
    and publishes a vlm.verified event for the flagged one."""
    monkeypatch.setattr(worker, "EVENTS_DIR", tmp_path)
    stream_file = tmp_path / "event_created.jsonl"
    flagged = _flagged_event()
    unflagged = dict(flagged, event_id="55555555-5555-5555-5555-555555555555",
                     needs_vlm_verification=False)
    stream_file.write_text(json.dumps(flagged) + "\n" + json.dumps(unflagged) + "\n")

    summaries = worker.run(once=True, limit=4, backend="file")
    assert len(summaries) == 2
    verified = [s for s in summaries if not s.get("skipped")]
    assert len(verified) == 1
    assert verified[0]["event_id"] == flagged["event_id"]

    # the verified event was published downstream
    out_file = tmp_path / "vlm_verified.jsonl"
    assert out_file.exists()
    pub = json.loads(out_file.read_text().splitlines()[0])
    assert pub["event_id"] == flagged["event_id"]
    assert pub["verdict"] in {"suspicious", "benign", "unclear"}


@pytest.mark.skipif(not _has_db(), reason="no Postgres")
def test_run_with_db_smoke():
    """If Postgres is up, run(once) against the pg backend must not crash."""
    summaries = worker.run(once=True, limit=2, backend="pg")
    assert isinstance(summaries, list)
