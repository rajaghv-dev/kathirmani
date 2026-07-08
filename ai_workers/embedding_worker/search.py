"""Natural-language video search — the A4.3 pipeline (master plan Phase 8).

  query -> query_parser -> metadata_filter -> vector_search -> fuse -> critic_rerank

* `query_parser`   — extract structured filters (object / action / route /
  missing-billing intent) and map them to a set of candidate `event_type`s + a
  time hint. Pure-python; no model.
* `metadata_filter`— SQL over `events` constrained by the parsed event_type set
  (+ store/camera/time); returns candidate event rows. Guarded for no-DB.
* `vector_search`  — pgvector cosine (`<->` on a normalized space, or `<=>`)
  over `embeddings` for the query vector; returns (entity_id, distance, text).
  Guarded for no-DB.
* `fuse`           — combine vector similarity + metadata match + time proximity
  into one score and rank. Pure-python, unit-testable.
* `critic_rerank`  — optional `SearchCriticPlugin` (Nemotron critic) that
  reranks + explains; a deterministic fake stub by default (no model/GPU).

`search(query, k)` returns ranked results: each carries timestamps + a `why`.

Everything that touches Postgres is best-effort: with no DB the SQL paths return
[] and `search` still runs end-to-end over any caller-supplied candidates (used
by the tests), so the module is hermetic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


from .plugin import NvidiaEmbeddingPlugin


# ---- intent vocabulary (configs/event_rules.yaml) ---------------------------
# Map natural-language cues -> the event_type set they should retrieve. Keys are
# regex fragments matched case-insensitively against the query.
_OBJECT_WORDS = ["item", "product", "bottle", "box", "bag", "handbag",
                 "backpack", "goods", "merchandise"]
_INTENT_RULES: list[tuple[str, set[str]]] = [
    # missing-billing / walk-away-without-paying -> the loss hypotheses
    (r"without (billing|paying|scan|checkout)|skip(ped)? (the )?(bill|counter)|"
     r"not (pay|bill)|missing billing|bypass|walk(ed|s|ing)? away|left without",
     {"possible_billing_bypass", "possible_loss", "billing_bypass"}),
    # picked / took / grabbed an item near a shelf
    (r"pick(ed|s|ing)?( up)?|took|grab(bed|s|bing)?|interact|touch(ed|ing)?|"
     r"remov(e|ed|ing)|put (it )?in (a )?(bag|pocket)",
     {"suspicious_item_interaction"}),
    # camera health / occlusion
    (r"camera (block|cover|down|black|frozen|tamper)|obscured|blind",
     {"camera_health_issue"}),
]
# route cues -> zone names (used as a metadata hint, not a hard filter)
_ROUTE_RULES: list[tuple[str, str]] = [
    (r"\bexit\b|left the store|out the door|walked out", "exit"),
    (r"\bentry\b|entered|came in", "entry"),
    (r"\bshelf|aisle|rack", "shelf"),
    (r"\bbilling|counter|checkout|cashier|till", "billing_counter"),
]


@dataclass
class ParsedQuery:
    """Structured form of a natural-language query."""
    text: str
    event_types: set[str] = field(default_factory=set)   # candidate event_type set
    objects: list[str] = field(default_factory=list)     # nouns of interest
    routes: list[str] = field(default_factory=list)      # zone-name hints
    store_id: str | None = None
    camera_id: str | None = None


@dataclass
class Candidate:
    """One scored search hit. `vector_score` ∈ [0,1] (cosine sim), `event_type`
    drives the metadata bonus, `event_time` the recency/proximity term."""
    entity_id: str
    entity_type: str = "event"
    text_repr: str = ""
    vector_score: float = 0.0                     # cosine similarity 0..1
    event_type: str | None = None
    event_time: str | None = None                 # ISO ts
    camera_id: str | None = None
    score: float = 0.0                            # filled by fuse()
    why: str = ""                                 # filled by fuse()/critic


# ---- 1. query_parser --------------------------------------------------------
def query_parser(query: str, store_id: str | None = None,
                 camera_id: str | None = None) -> ParsedQuery:
    """Extract filters from a free-text query. Deterministic, no model."""
    q = query.lower()
    pq = ParsedQuery(text=query, store_id=store_id, camera_id=camera_id)
    for pat, ev_types in _INTENT_RULES:
        if re.search(pat, q):
            pq.event_types |= ev_types
    for pat, zone in _ROUTE_RULES:
        if re.search(pat, q) and zone not in pq.routes:
            pq.routes.append(zone)
    for w in _OBJECT_WORDS:
        if re.search(rf"\b{w}s?\b", q) and w not in pq.objects:
            pq.objects.append(w)
    # "picked an item and walked away" implies both interaction + bypass loss:
    if pq.objects and re.search(r"walk(ed|s|ing)? away|left|away", q):
        pq.event_types |= {"possible_billing_bypass", "possible_loss"}
    return pq


# ---- 2. metadata_filter (SQL over events; no-DB -> []) ----------------------
def metadata_filter(pq: ParsedQuery, conn=None, limit: int = 200) -> list[Candidate]:
    """Return candidate event rows matching the parsed event_type set (+ optional
    store/camera). Best-effort: returns [] with no DB."""
    if conn is None:
        conn = _db_conn()
    if conn is None:
        return []
    where = []
    params: list[Any] = []
    if pq.event_types:
        where.append("event_type = ANY(%s)")
        params.append(list(pq.event_types))
    if pq.store_id:
        where.append("store_id = %s"); params.append(pq.store_id)
    if pq.camera_id:
        where.append("camera_id = %s"); params.append(pq.camera_id)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    sql = (f"SELECT id::text, event_type, camera_id, event_time::text "
           f"FROM events {clause} ORDER BY event_time DESC LIMIT %s")
    params.append(limit)
    cands: list[Candidate] = []
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            for eid, et, cam, ts in cur.fetchall():
                cands.append(Candidate(entity_id=eid, entity_type="event",
                                       event_type=et, camera_id=cam, event_time=ts))
    except Exception:
        _rollback(conn)
    return cands


# ---- 3. vector_search (pgvector cosine; no-DB -> []) ------------------------
def vector_search(query_vec: list[float], conn=None, k: int = 50,
                  entity_type: str | None = None) -> list[Candidate]:
    """Cosine nearest-neighbours over the `embeddings` table for `query_vec`.

    Uses pgvector's cosine distance operator `<=>` (1 - cosine_similarity), so
    `vector_score = 1 - distance`. Best-effort: returns [] with no DB / no
    pgvector. The vector is passed as a pgvector literal '[v1,v2,...]'."""
    if conn is None:
        conn = _db_conn()
    if conn is None:
        return []
    lit = "[" + ",".join(f"{v:.6f}" for v in query_vec) + "]"
    where = "WHERE entity_type = %s" if entity_type else ""
    params: list[Any] = ([entity_type] if entity_type else []) + [lit, k]
    sql = (f"SELECT entity_id::text, entity_type, text_repr, "
           f"(embedding <=> %s::vector) AS dist FROM embeddings {where} "
           f"ORDER BY embedding <=> %s::vector LIMIT %s")
    # note: the cosine literal appears twice (SELECT + ORDER BY)
    params = ([entity_type] if entity_type else []) + [lit, lit, k]
    cands: list[Candidate] = []
    try:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            for eid, et, text, dist in cur.fetchall():
                cands.append(Candidate(
                    entity_id=eid, entity_type=et, text_repr=text or "",
                    vector_score=max(0.0, 1.0 - float(dist))))
    except Exception:
        _rollback(conn)
    return cands


# ---- 4. fuse ----------------------------------------------------------------
def fuse(pq: ParsedQuery, candidates: list[Candidate],
         w_vector: float = 0.7, w_meta: float = 0.2, w_time: float = 0.1,
         now_iso: str | None = None) -> list[Candidate]:
    """Combine vector similarity + metadata match + time proximity into one score
    and return candidates ranked desc. A candidate whose event_type is in the
    parsed set gets the metadata bonus; more-recent events get a small recency
    bump. Pure-python (unit-tested without a DB)."""
    ranked: list[Candidate] = []
    times = [_epoch(c.event_time) for c in candidates if c.event_time]
    t_min, t_max = (min(times), max(times)) if times else (0.0, 0.0)
    span = (t_max - t_min) or 1.0
    for c in candidates:
        meta = 1.0 if (c.event_type and c.event_type in pq.event_types) else 0.0
        # recency in [0,1]: newest among the candidate set scores 1.0
        if c.event_time:
            tprox = (_epoch(c.event_time) - t_min) / span
        else:
            tprox = 0.0
        c.score = round(w_vector * c.vector_score + w_meta * meta + w_time * tprox, 6)
        c.why = (f"vector={c.vector_score:.3f}*{w_vector} + "
                 f"meta={meta:.0f}*{w_meta}"
                 + (f" ({c.event_type}∈query)" if meta else "")
                 + f" + recency={tprox:.2f}*{w_time}")
        ranked.append(c)
    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


# ---- 5. critic_rerank (SearchCriticPlugin stub; fake by default) ------------
class SearchCriticStub:
    """A `SearchCriticPlugin` (Nemotron critic) stub — reranks + explains. The
    real plugin would call nvidia/NVIDIA-Nemotron-3-Nano (ollama) to judge each
    candidate against the query; this fake is deterministic (no model/GPU): it
    nudges candidates whose `text_repr` shares query keywords up, and writes a
    plain-language `why`. Swap `rerank` for the real LLM call in Phase 8.5."""

    task = "search_critic"

    def rerank(self, query: str, candidates: list[Candidate],
               top_k: int | None = None) -> list[Candidate]:
        terms = {t for t in re.findall(r"[a-z]{3,}", query.lower())}
        out: list[Candidate] = []
        for c in candidates:
            overlap = len(terms & set(re.findall(r"[a-z]{3,}", c.text_repr.lower())))
            bonus = 0.05 * overlap                # small, bounded keyword nudge
            c.score = round(c.score + bonus, 6)
            if overlap:
                c.why += f" | critic:+{bonus:.2f} (matched {overlap} term(s))"
            else:
                c.why += " | critic: no lexical overlap"
            out.append(c)
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k] if top_k else out


# ---- top-level pipeline -----------------------------------------------------
def search(query: str, k: int = 10, conn=None, store_id: str | None = None,
           camera_id: str | None = None, plugin: NvidiaEmbeddingPlugin | None = None,
           critic: SearchCriticStub | None = None,
           candidates: list[Candidate] | None = None) -> list[Candidate]:
    """End-to-end A4.3 search. With a DB it queries events + embeddings; without
    one (or when `candidates` is supplied) it fuses the given candidate set so the
    pipeline is fully exercisable offline. Returns top-k ranked Candidates with
    timestamps + `why`."""
    pq = query_parser(query, store_id=store_id, camera_id=camera_id)
    plugin = plugin or NvidiaEmbeddingPlugin()
    qvec = plugin.infer({"text": query})["vector"]

    if candidates is None:
        if conn is None:
            conn = _db_conn()
        meta = metadata_filter(pq, conn=conn)
        vec = vector_search(qvec, conn=conn, k=max(k * 5, 50))
        candidates = _merge(meta, vec)

    ranked = fuse(pq, candidates)
    critic = critic or SearchCriticStub()
    ranked = critic.rerank(query, ranked, top_k=k)
    return ranked


# ---- helpers ----------------------------------------------------------------
def _merge(meta: list[Candidate], vec: list[Candidate]) -> list[Candidate]:
    """Union metadata + vector candidates by entity_id, carrying the best of each
    signal (metadata gives event_type/time; vectors give vector_score/text)."""
    by_id: dict[str, Candidate] = {}
    for c in meta + vec:
        ex = by_id.get(c.entity_id)
        if ex is None:
            by_id[c.entity_id] = c
            continue
        ex.vector_score = max(ex.vector_score, c.vector_score)
        ex.event_type = ex.event_type or c.event_type
        ex.event_time = ex.event_time or c.event_time
        ex.camera_id = ex.camera_id or c.camera_id
        ex.text_repr = ex.text_repr or c.text_repr
    return list(by_id.values())


def _epoch(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        from datetime import datetime
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


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
