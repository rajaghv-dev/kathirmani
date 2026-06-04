"""Worker loop — hermetic (FileQueue, no Postgres). DB path behind skipif."""
from __future__ import annotations

import json

import pytest

import worker
from plugin import CvOssDetector


def _has_db() -> bool:
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def test_process_window_no_db():
    """process_window runs the plugin and survives with no DB / FileQueue."""
    from ingestion.bus import FileQueue
    import tempfile
    from pathlib import Path

    q = FileQueue(Path(tempfile.mkdtemp()))
    p = CvOssDetector()
    w = {"window_id": "w1", "camera_id": "left_aisle", "window_start_sec": 0.0,
         "clip_path": "/nope.mp4", "store_id": "kathirmani_01"}
    summary = worker.process_window(w, p, q, conn=None)
    assert summary["detections"] >= 1
    assert summary["detections_written"] == 0          # no DB
    assert summary["events"] >= 1                       # suspicious-item hypothesis
    q.close()


def test_run_once_file_backend(tmp_path, monkeypatch):
    """run(once=True) drains a file-queue stream and processes its windows."""
    # point the worker's FileQueue dir at a temp dir and seed one window
    monkeypatch.setattr(worker, "EVENTS_DIR", tmp_path)
    stream_file = tmp_path / "ai_window_ready.jsonl"
    wid = "33333333-3333-3333-3333-333333333333"      # valid UUID (DB may be live)
    stream_file.write_text(json.dumps({
        "type": "ai_window.ready", "window_id": wid, "camera_id": "left_aisle",
        "window_start_sec": 0.0, "clip_path": "/nope.mp4", "store_id": "kathirmani_01",
    }) + "\n")
    summaries = worker.run(once=True, limit=4, backend="file")
    assert len(summaries) == 1
    assert summaries[0]["window_id"] == wid
    assert summaries[0]["faked"] is True
    # the emitted event was published to the file queue
    ev_file = tmp_path / "event_created.jsonl"
    assert ev_file.exists()
    ev = json.loads(ev_file.read_text().splitlines()[0])
    assert ev["event_type"] == "suspicious_item_interaction"


@pytest.mark.skipif(not _has_db(), reason="no Postgres")
def test_run_with_db_smoke():
    """If Postgres is up, run(once) against the pg backend must not crash."""
    summaries = worker.run(once=True, limit=2, backend="pg")
    assert isinstance(summaries, list)
