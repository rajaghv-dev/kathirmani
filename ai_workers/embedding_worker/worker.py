"""embedding_worker CLI (master plan Phase 8).

Two subcommands, both guarded for no-DB (they degrade to a hermetic no-op /
fake-vector path instead of crashing):

  index            — embed existing events + vlm_observations into `embeddings`
                     (idempotent upsert). Prints per-source counts.
  query "<text>"   — run the A4.3 search pipeline and print ranked hits with
                     timestamps + why.

Entrypoints: `index_run(limit=N)`, `query_run(text, k=N)`. `python -m` friendly.
"""
from __future__ import annotations



from .indexer import index_all
from .plugin import NvidiaEmbeddingPlugin
from .search import search


def _db_conn():
    try:
        from db import connect
        return connect()
    except Exception:
        return None


def index_run(limit: int = 1000) -> dict[str, int]:
    """Index all sources; returns per-source counts (0s with no DB)."""
    conn = _db_conn()
    plugin = NvidiaEmbeddingPlugin()
    plugin.load()
    try:
        counts = index_all(conn=conn, plugin=plugin, limit=limit)
    finally:
        plugin.unload()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return counts


def query_run(text: str, k: int = 10) -> list[dict]:
    """Run search; returns a list of small result dicts (also printed)."""
    conn = _db_conn()
    plugin = NvidiaEmbeddingPlugin()
    plugin.load()
    try:
        hits = search(text, k=k, conn=conn, plugin=plugin)
    finally:
        plugin.unload()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return [{"entity_type": h.entity_type, "entity_id": h.entity_id,
             "event_type": h.event_type, "event_time": h.event_time,
             "score": h.score, "why": h.why, "text_repr": h.text_repr}
            for h in hits]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="embedding_worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="embed existing rows into `embeddings`")
    pi.add_argument("--limit", type=int, default=1000)
    pq = sub.add_parser("query", help="natural-language video search")
    pq.add_argument("text")
    pq.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    if args.cmd == "index":
        counts = index_run(limit=args.limit)
        print(f"[embedding_worker] indexed: {counts}")
    elif args.cmd == "query":
        results = query_run(args.text, k=args.k)
        print(f"[embedding_worker] {len(results)} hit(s) for {args.text!r}:")
        for i, r in enumerate(results, 1):
            print(f"  {i}. score={r['score']:.4f} {r['entity_type']}/{r['entity_id']} "
                  f"[{r['event_type']}] @ {r['event_time']}")
            print(f"     text: {r['text_repr']}")
            print(f"     why:  {r['why']}")
