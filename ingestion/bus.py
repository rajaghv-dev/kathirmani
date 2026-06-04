"""Job queue (master plan §5.1/§8).

DECISION (2026-06-03): the platform queue is **Postgres `SELECT ... FOR UPDATE SKIP
LOCKED`** (`PgQueue`), not Redis — one fewer service, transactional enqueue+state, and
an auditable queue-as-table (good for the trust posture). At 30-camera scale the message
rate is modest, so Postgres is ample; Redis Streams (`RedisStreamQueue`) is kept only as
an optional high-fanout backend. `FileQueue` is the hermetic Phase-1 default (no service).
PgQueue lands in Phase 2 with the schema. Publishes `video_segment.ready` /
`ai_window.ready` (§8.1/§8.2).
"""
from __future__ import annotations

import abc
import json
import os
from pathlib import Path


class Queue(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def publish(self, message: dict) -> None: ...

    def close(self) -> None: ...


class FileQueue(Queue):
    name = "file"

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh: dict[str, object] = {}

    def publish(self, message: dict) -> None:
        stream = message.get("type", "events").replace(".", "_")
        fh = self._fh.get(stream)
        if fh is None:
            fh = open(self.dir / f"{stream}.jsonl", "a"); self._fh[stream] = fh
        fh.write(json.dumps(message) + "\n"); fh.flush()

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()


class RedisStreamQueue(Queue):
    name = "redis"

    def __init__(self, url: str):
        import redis
        self.r = redis.Redis.from_url(url)
        self.r.ping()

    def publish(self, message: dict) -> None:
        stream = message.get("type", "events")            # e.g. video_segment.ready
        # XADD with flat string fields (Redis stream values are strings)
        self.r.xadd(stream, {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                             for k, v in message.items()})


class PgQueue(Queue):
    """The platform queue: a Postgres `job_queue` table consumed with
    `SELECT … FOR UPDATE SKIP LOCKED` (replaces Redis — spec/10 Data stack).
    publish = INSERT; consume claims pending rows for one worker; ack/fail close them.
    """
    name = "pg"

    def __init__(self, dsn: str = ""):
        import psycopg
        self.conn = psycopg.connect(dsn or None, autocommit=True)

    def publish(self, message: dict) -> None:
        stream = message.get("type", "events")
        with self.conn.cursor() as cur:
            cur.execute("INSERT INTO job_queue (stream, payload) VALUES (%s, %s::jsonb)",
                        (stream, json.dumps(message)))

    def consume(self, stream: str, limit: int = 1) -> list[tuple[int, dict]]:
        """Claim up to `limit` pending jobs for this worker (SKIP LOCKED)."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, payload FROM job_queue WHERE stream=%s AND status='pending' "
                "ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT %s", (stream, limit))
            rows = cur.fetchall()
            if rows:
                cur.execute("UPDATE job_queue SET status='processing', locked_at=now() "
                            "WHERE id = ANY(%s)", ([r[0] for r in rows],))
        return [(r[0], r[1]) for r in rows]

    def ack(self, job_id: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("UPDATE job_queue SET status='done' WHERE id=%s", (job_id,))

    def fail(self, job_id: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute("UPDATE job_queue SET status='failed', attempts=attempts+1 "
                        "WHERE id=%s", (job_id,))

    def close(self) -> None:
        self.conn.close()


def make_queue(backend: str | None, out_dir: Path) -> Queue:
    backend = backend or os.environ.get("INGEST_QUEUE", "file")
    if backend == "pg":
        try:
            from db import dsn
            return PgQueue(os.environ.get("DATABASE_URL") or dsn())
        except Exception as e:                            # graceful fallback
            print(f"[queue] Postgres unavailable ({e}); falling back to file")
    if backend == "redis":                                # optional high-fanout backend
        try:
            return RedisStreamQueue(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        except Exception as e:                            # graceful fallback
            print(f"[queue] Redis unavailable ({e}); falling back to file")
    return FileQueue(out_dir)
