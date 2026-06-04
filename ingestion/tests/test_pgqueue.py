"""PgQueue (Postgres SKIP LOCKED) roundtrip. Skips if Postgres is down."""
from __future__ import annotations

import pytest


def _db_ok() -> bool:
    try:
        from db import connect
        with connect() as c, c.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_ok(), reason="no Postgres")


def test_publish_consume_ack():
    from db import connect, dsn
    from ingestion.bus import PgQueue

    q = PgQueue(dsn())
    try:
        q.publish({"type": "test.queue", "marker": 4242})
        jobs = q.consume("test.queue", limit=10)
        assert any(p.get("marker") == 4242 for _, p in jobs)
        job_id = jobs[0][0]
        q.ack(job_id)
        # a re-consume must not return acked jobs (claimed/done are excluded)
        again = q.consume("test.queue", limit=10)
        assert job_id not in [j[0] for j in again]
    finally:
        with connect() as c, c.cursor() as cur:        # cleanup test rows
            cur.execute("DELETE FROM job_queue WHERE stream='test.queue'")
            c.commit()
        q.close()
