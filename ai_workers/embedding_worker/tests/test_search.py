"""Search pipeline — query_parser, fuse, critic, and end-to-end search.

All hermetic (no DB): the DB-backed paths (metadata_filter / vector_search) are
exercised behind skipif; the pure-python pipeline is fully tested with
caller-supplied candidates.
"""
from __future__ import annotations

import pytest

from ai_workers.embedding_worker.plugin import NvidiaEmbeddingPlugin
from ai_workers.embedding_worker.search import (Candidate, SearchCriticStub, fuse, metadata_filter,
                    query_parser, search, vector_search)

EXAMPLE = "show a person who picked an item and walked away"


def _has_db() -> bool:
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


# ---- query_parser -----------------------------------------------------------
def test_query_parser_extracts_expected_filters():
    pq = query_parser(EXAMPLE)
    # "picked ... item" -> interaction; "walked away" + object -> bypass/loss
    assert "suspicious_item_interaction" in pq.event_types
    assert "possible_billing_bypass" in pq.event_types
    assert "possible_loss" in pq.event_types
    assert "item" in pq.objects


def test_query_parser_missing_billing_intent():
    pq = query_parser("someone left the store without paying")
    assert pq.event_types & {"possible_billing_bypass", "possible_loss",
                             "billing_bypass"}
    assert "exit" in pq.routes


def test_query_parser_camera_health():
    pq = query_parser("when was the camera blocked")
    assert "camera_health_issue" in pq.event_types


def test_query_parser_passes_through_store_camera():
    pq = query_parser("anything", store_id="kathirmani_01", camera_id="bill_counter")
    assert pq.store_id == "kathirmani_01"
    assert pq.camera_id == "bill_counter"


# ---- fuse -------------------------------------------------------------------
def test_fuse_ranks_higher_similarity_first():
    pq = query_parser(EXAMPLE)
    cands = [
        Candidate(entity_id="low", vector_score=0.20, event_type="other"),
        Candidate(entity_id="high", vector_score=0.90, event_type="other"),
    ]
    ranked = fuse(pq, cands)
    assert ranked[0].entity_id == "high"
    assert ranked[0].score > ranked[1].score
    assert "vector=" in ranked[0].why


def test_fuse_metadata_bonus_can_outrank_small_vector_gap():
    pq = query_parser(EXAMPLE)               # event_types include the loss set
    cands = [
        # slightly higher vector sim but irrelevant event_type
        Candidate(entity_id="a", vector_score=0.55, event_type="camera_health_issue"),
        # lower vector sim but matches the parsed intent -> metadata bonus
        Candidate(entity_id="b", vector_score=0.50,
                  event_type="suspicious_item_interaction"),
    ]
    ranked = fuse(pq, cands, w_vector=0.5, w_meta=0.5, w_time=0.0)
    assert ranked[0].entity_id == "b"        # metadata match wins


def test_fuse_recency_breaks_vector_ties():
    pq = query_parser(EXAMPLE)
    cands = [
        Candidate(entity_id="old", vector_score=0.5, event_type="x",
                  event_time="2026-06-01T10:00:00+00:00"),
        Candidate(entity_id="new", vector_score=0.5, event_type="x",
                  event_time="2026-06-03T10:00:00+00:00"),
    ]
    ranked = fuse(pq, cands)
    assert ranked[0].entity_id == "new"


# ---- critic -----------------------------------------------------------------
def test_critic_rerank_nudges_lexical_overlap_and_explains():
    pq = query_parser(EXAMPLE)
    cands = [
        Candidate(entity_id="match", vector_score=0.50,
                  text_repr="person picked item near shelf"),
        Candidate(entity_id="nomatch", vector_score=0.52, text_repr="bottle on rack"),
    ]
    ranked = fuse(pq, cands, w_meta=0.0, w_time=0.0)
    out = SearchCriticStub().rerank(EXAMPLE, ranked)
    assert out[0].entity_id == "match"       # keyword overlap nudges it above
    assert "critic" in out[0].why


def test_critic_respects_top_k():
    cands = [Candidate(entity_id=str(i), vector_score=i / 10.0) for i in range(8)]
    out = SearchCriticStub().rerank("x", cands, top_k=3)
    assert len(out) == 3


# ---- end-to-end (no DB: pass candidates) ------------------------------------
def test_search_end_to_end_with_supplied_candidates():
    plugin = NvidiaEmbeddingPlugin()
    cands = [
        Candidate(entity_id="hit", entity_type="event", vector_score=0.88,
                  event_type="suspicious_item_interaction",
                  text_repr="person picked an item and walked toward exit",
                  event_time="2026-06-03T09:00:00+00:00"),
        Candidate(entity_id="miss", entity_type="event", vector_score=0.10,
                  event_type="camera_health_issue", text_repr="camera frozen",
                  event_time="2026-06-02T09:00:00+00:00"),
    ]
    results = search(EXAMPLE, k=5, plugin=plugin, candidates=cands)
    assert results[0].entity_id == "hit"
    assert results[0].event_time                 # carries a timestamp
    assert results[0].why                        # carries an explanation
    assert len(results) <= 5


# ---- DB-backed paths (skipped without Postgres) -----------------------------
@pytest.mark.skipif(not _has_db(), reason="no Postgres")
def test_metadata_and_vector_search_smoke():
    pq = query_parser(EXAMPLE)
    assert isinstance(metadata_filter(pq), list)
    qvec = NvidiaEmbeddingPlugin().infer({"text": EXAMPLE})["vector"]
    assert isinstance(vector_search(qvec, k=5), list)


@pytest.mark.skipif(not _has_db(), reason="no Postgres")
def test_search_against_db_does_not_crash():
    assert isinstance(search(EXAMPLE, k=5), list)
