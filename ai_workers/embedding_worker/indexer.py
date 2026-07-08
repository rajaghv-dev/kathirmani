"""Indexer — read platform artifacts, build a text_repr, embed, upsert into
`embeddings` (master plan Phase 8 / spec/11 A4.3).

Sources (each row -> one `embeddings` row, entity_type tags the source):
  * `events`            -> entity_type='event'        (event_type + objects + zones)
  * `vlm_observations`  -> entity_type='vlm_observation' (parsed verdict/explanation)
  * `events` w/ clip    -> the visual path embeds the evidence clip's first frame.

For each row we build a compact natural-language `text_repr` (what the search
query will be matched against), embed it with `NvidiaEmbeddingPlugin` (fake-infer
without weights), and INSERT into `embeddings(entity_type, entity_id, model_name,
embedding, text_repr)`. Upsert = delete-then-insert per (entity_type, entity_id,
model_name) so re-indexing is idempotent (the table has no natural unique key).

Everything that touches Postgres is best-effort and guarded: with no DB the
`index_*` functions are no-ops returning 0, so the module imports and runs in CI.
"""
from __future__ import annotations

import json
from typing import Any


from .plugin import NvidiaEmbeddingPlugin


# ---- text_repr builders (pure, unit-testable) -------------------------------
def event_text_repr(row: dict[str, Any]) -> str:
    """Compact NL description of an event row for embedding/search match.
    e.g. 'suspicious_item_interaction at left_aisle [shelf] objects: person, item'."""
    meta = row.get("metadata_json") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    parts = [str(row.get("event_type", "event"))]
    if row.get("camera_id"):
        parts.append(f"at {row['camera_id']}")
    zones = meta.get("zones") or []
    if zones:
        parts.append(f"[{', '.join(map(str, zones))}]")
    objs = meta.get("objects") or []
    if objs:
        parts.append("objects: " + ", ".join(map(str, objs)))
    if row.get("severity") and row["severity"] != "info":
        parts.append(f"severity={row['severity']}")
    return " ".join(parts)


def vlm_text_repr(row: dict[str, Any]) -> str:
    """NL description of a vlm_observation: verdict + actions + explanation."""
    parsed = row.get("parsed_json") or {}
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            parsed = {}
    parts = []
    if parsed.get("verdict"):
        parts.append(f"verdict: {parsed['verdict']}")
    if parsed.get("structured_event_type"):
        parts.append(str(parsed["structured_event_type"]))
    acts = parsed.get("observed_actions") or []
    if acts:
        parts.append("actions: " + ", ".join(map(str, acts)))
    expl = parsed.get("explanation") or row.get("raw_output") or ""
    if expl:
        parts.append(str(expl)[:400])
    return " ".join(parts) or "vlm observation"


# ---- upsert -----------------------------------------------------------------
def upsert_embedding(conn, entity_type: str, entity_id: str, model_name: str,
                     vector: list[float], text_repr: str) -> bool:
    """Idempotent insert into `embeddings` (delete-then-insert on the logical
    key). Best-effort: returns False with no DB / on any error. Vector goes in as
    a pgvector literal '[v1,...]'."""
    if conn is None or not entity_id:
        return False
    lit = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM embeddings WHERE entity_type=%s AND entity_id=%s "
                "AND model_name=%s", (entity_type, entity_id, model_name))
            cur.execute(
                "INSERT INTO embeddings (entity_type, entity_id, model_name, "
                "embedding, text_repr) VALUES (%s, %s, %s, %s::vector, %s)",
                (entity_type, entity_id, model_name, lit, text_repr))
        conn.commit()
        return True
    except Exception:
        _rollback(conn)
        return False


# ---- per-source indexers ----------------------------------------------------
def index_events(conn, plugin: NvidiaEmbeddingPlugin, limit: int = 1000) -> int:
    """Embed recent `events` rows into `embeddings`. Returns count indexed."""
    if conn is None:
        return 0
    rows = _fetch(conn,
                  "SELECT id::text, event_type, camera_id, severity, metadata_json "
                  "FROM events ORDER BY event_time DESC LIMIT %s", (limit,),
                  ("id", "event_type", "camera_id", "severity", "metadata_json"))
    n = 0
    for r in rows:
        text = event_text_repr(r)
        vec = plugin.infer({"text": text})["vector"]
        if upsert_embedding(conn, "event", r["id"], plugin.config.model_id, vec, text):
            n += 1
    return n


def index_vlm_observations(conn, plugin: NvidiaEmbeddingPlugin,
                           limit: int = 1000) -> int:
    """Embed `vlm_observations` rows into `embeddings`. Returns count indexed."""
    if conn is None:
        return 0
    rows = _fetch(conn,
                  "SELECT id::text, parsed_json, raw_output FROM vlm_observations "
                  "ORDER BY created_at DESC LIMIT %s", (limit,),
                  ("id", "parsed_json", "raw_output"))
    n = 0
    for r in rows:
        text = vlm_text_repr(r)
        vec = plugin.infer({"text": text})["vector"]
        if upsert_embedding(conn, "vlm_observation", r["id"],
                            plugin.config.model_id, vec, text):
            n += 1
    return n


def index_all(conn=None, plugin: NvidiaEmbeddingPlugin | None = None,
              limit: int = 1000) -> dict[str, int]:
    """Index every supported source. Best-effort: {} counts are 0 with no DB."""
    if conn is None:
        conn = _db_conn()
    plugin = plugin or NvidiaEmbeddingPlugin()
    plugin.load()                                 # lazy/non-fatal
    return {
        "events": index_events(conn, plugin, limit),
        "vlm_observations": index_vlm_observations(conn, plugin, limit),
    }


# ---- helpers ----------------------------------------------------------------
def _fetch(conn, sql: str, params: tuple, cols: tuple[str, ...]) -> list[dict]:
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        _rollback(conn)
        return []


def _db_conn():
    try:
        from db import connect
        return connect()
    except Exception:
        return None


def _rollback(conn) -> None:
    try:
        conn.rollback()
    except Exception:
        pass
