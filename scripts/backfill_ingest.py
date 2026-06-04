#!/usr/bin/env python3
"""Backfill ingestion JSONL → Postgres (the file-first → DB bridge, spec/db).

Reads data/metadata/{segments,windows}.jsonl (written by `python -m ingestion`) and
inserts video_segments + ai_windows. Idempotent: segments dedup on (camera_id,
start_time), windows insert only if their segment exists. `make backfill`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from db import connect  # noqa: E402

META = ROOT / "data" / "metadata"


def _rows(name):
    p = META / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main() -> int:
    segs, wins = _rows("segments.jsonl"), _rows("windows.jsonl")
    if not segs and not wins:
        print(f"nothing to backfill ({META} empty — run `python -m ingestion` first)")
        return 0
    ns = nw = 0
    with connect() as conn, conn.cursor() as cur:
        for s in segs:
            cur.execute(
                "INSERT INTO video_segments (id,store_id,camera_id,start_time,end_time,"
                "duration_sec,storage_path,thumbnail_path,codec,fps,width,height,checksum,"
                "health_json,status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (camera_id,start_time) DO NOTHING",
                (s["segment_id"], s["store_id"], s["camera_id"], s["start_time"], s["end_time"],
                 s["duration_sec"], s["storage_path"], s.get("thumbnail_path"), s.get("codec"),
                 s.get("fps"), s.get("width"), s.get("height"), s.get("checksum"),
                 json.dumps(s.get("health", {})), s.get("status", "ready")),
            )
            ns += cur.rowcount
        for w in wins:
            cur.execute(
                "INSERT INTO ai_windows (id,segment_id,camera_id,window_start_sec,"
                "window_end_sec,clip_path,status) "
                "SELECT %s,%s,%s,%s,%s,%s,%s WHERE EXISTS "
                "(SELECT 1 FROM video_segments WHERE id=%s) "
                "ON CONFLICT (id) DO NOTHING",
                (w["window_id"], w["segment_id"], w["camera_id"], w["window_start_sec"],
                 w["window_end_sec"], w.get("clip_path"), w.get("status", "pending"),
                 w["segment_id"]),
            )
            nw += cur.rowcount
        conn.commit()
    print(f"backfill: {ns} segments, {nw} windows inserted "
          f"({len(segs)} / {len(wins)} read; rest already present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
