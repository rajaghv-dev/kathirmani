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
    """Phase-2 slot: the platform queue, a Postgres `job_queue` table consumed with
    `SELECT ... FOR UPDATE SKIP LOCKED`. Needs psycopg + the Phase-2 schema; raises
    until then so we don't pretend (Phase 1 uses FileQueue)."""
    name = "pg"

    def __init__(self, dsn: str):
        raise NotImplementedError(
            "PgQueue (Postgres SKIP LOCKED) lands in Phase 2 with db/schema.sql. "
            "Phase 1 uses FileQueue; this replaces Redis as the platform queue."
        )

    def publish(self, message: dict) -> None: ...


def make_queue(backend: str | None, out_dir: Path) -> Queue:
    backend = backend or os.environ.get("INGEST_QUEUE", "file")
    if backend == "pg":
        return PgQueue(os.environ.get("DATABASE_URL", ""))
    if backend == "redis":                                # optional high-fanout backend
        try:
            return RedisStreamQueue(os.environ.get("REDIS_URL", "redis://localhost:6379"))
        except Exception as e:                            # graceful fallback
            print(f"[queue] Redis unavailable ({e}); falling back to file")
    return FileQueue(out_dir)
