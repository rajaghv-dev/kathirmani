"""Indexer + worker — text_repr builders (pure) and the no-DB guards.

DB-backed indexing is behind skipif; the text_repr builders and the no-DB
no-op paths are tested hermetically.
"""
from __future__ import annotations

import pytest

import worker
from indexer import (event_text_repr, index_all, index_events,
                     index_vlm_observations, upsert_embedding, vlm_text_repr)
from plugin import NvidiaEmbeddingPlugin


def _has_db() -> bool:
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


# ---- text_repr builders -----------------------------------------------------
def test_event_text_repr_includes_type_camera_zones_objects():
    row = {"event_type": "suspicious_item_interaction", "camera_id": "left_aisle",
           "severity": "medium",
           "metadata_json": {"zones": ["left_aisle_shelf"],
                             "objects": ["person", "item"]}}
    text = event_text_repr(row)
    assert "suspicious_item_interaction" in text
    assert "left_aisle" in text
    assert "person" in text and "item" in text
    assert "severity=medium" in text


def test_event_text_repr_tolerates_string_metadata_and_missing_fields():
    assert event_text_repr({"event_type": "x", "metadata_json": "not json"})
    assert event_text_repr({})                       # no fields -> "event"


def test_vlm_text_repr_builds_from_parsed_json():
    row = {"parsed_json": {"verdict": "suspicious",
                           "observed_actions": ["pickup", "conceal"],
                           "explanation": "person hid an item"}}
    text = vlm_text_repr(row)
    assert "verdict: suspicious" in text
    assert "pickup" in text
    assert "person hid an item" in text


# ---- no-DB guards -----------------------------------------------------------
def test_indexers_noop_without_db():
    p = NvidiaEmbeddingPlugin()
    assert index_events(None, p) == 0
    assert index_vlm_observations(None, p) == 0
    assert upsert_embedding(None, "event", "id", "m", [0.0] * 768, "t") is False


def test_index_all_keys_and_zero_counts_when_db_absent():
    """index_all always returns the two source keys. With no DB available the
    counts are 0; if a live Postgres is present it may index real rows (>=0)."""
    counts = index_all(conn=None)
    assert set(counts) == {"events", "vlm_observations"}
    if not _has_db():
        assert counts == {"events": 0, "vlm_observations": 0}


def test_worker_query_run_hermetic():
    """query_run runs the full pipeline; with no DB it returns [] (no candidates)
    but must not crash."""
    results = worker.query_run("a person picked an item and walked away", k=3)
    assert isinstance(results, list)


def test_worker_index_run_hermetic():
    counts = worker.index_run(limit=5)
    assert isinstance(counts, dict)


# ---- DB-backed (skipped without Postgres) -----------------------------------
@pytest.mark.skipif(not _has_db(), reason="no Postgres")
def test_index_all_against_db_smoke():
    counts = index_all()
    assert set(counts) == {"events", "vlm_observations"}
