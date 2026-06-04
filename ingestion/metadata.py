"""Metadata sink for segment/window rows.

JSONL is the hermetic default + the file-first contract the repo already lives by
(`results/*.json` is truth; the DB indexes it — spec/db). The Postgres sink is a
documented Phase-2 slot: the schema (`video_segments`/`ai_windows`, master plan §6)
lands in Phase 2, then this writes rows. Until then JSONL is authoritative.
"""
from __future__ import annotations

import abc
import json
import os
from pathlib import Path


class MetadataSink(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def write_segment(self, row: dict) -> None: ...

    @abc.abstractmethod
    def write_window(self, row: dict) -> None: ...

    def close(self) -> None: ...


class JsonlSink(MetadataSink):
    name = "jsonl"

    def __init__(self, out_dir: Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._seg = open(self.dir / "segments.jsonl", "a")
        self._win = open(self.dir / "windows.jsonl", "a")

    def write_segment(self, row: dict) -> None:
        self._seg.write(json.dumps(row) + "\n"); self._seg.flush()

    def write_window(self, row: dict) -> None:
        self._win.write(json.dumps(row) + "\n"); self._win.flush()

    def close(self) -> None:
        self._seg.close(); self._win.close()


class PostgresSink(MetadataSink):
    """Phase-2 slot. Inserts into video_segments / ai_windows once the schema +
    migrations exist (master plan §6). Raises until then so we don't pretend."""
    name = "postgres"

    def __init__(self, dsn: str):
        raise NotImplementedError(
            "PostgresSink lands in Phase 2 with db/schema.sql + migrations. "
            "Phase 1 uses JsonlSink (file-first); the Phase-2 backfill adapter "
            "loads these JSONL rows into Postgres."
        )

    def write_segment(self, row: dict) -> None: ...
    def write_window(self, row: dict) -> None: ...


def make_sink(backend: str | None, out_dir: Path) -> MetadataSink:
    backend = backend or os.environ.get("INGEST_METADATA", "jsonl")
    if backend == "postgres":
        return PostgresSink(os.environ.get("DATABASE_URL", ""))
    return JsonlSink(out_dir)
