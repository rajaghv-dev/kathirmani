"""CLI: python -m ingestion [--duration N] [--camera ID]... [--storage local|minio]
[--metadata jsonl|postgres] [--queue file|redis|pg] [--serve-metrics]

Default backends are file-first (local FS clips, JSONL metadata, file queue) so a run
needs no external service. Point at real backends with the flags / env (spec/10)."""
from __future__ import annotations

import argparse
import json
import sys

from .ingest import IngestOptions, run_ingest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ingestion")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds of each source to ingest (smoke tests)")
    ap.add_argument("--camera", action="append", default=None,
                    help="camera id or name (repeatable); default = all enabled")
    ap.add_argument("--storage", choices=["local", "minio"], default=None)
    ap.add_argument("--metadata", choices=["jsonl", "postgres"], default=None)
    ap.add_argument("--queue", choices=["file", "redis", "pg"], default=None)
    ap.add_argument("--serve-metrics", action="store_true")
    ap.add_argument("--cameras-yaml", default=None)
    args = ap.parse_args(argv)

    opts = IngestOptions(
        duration_limit=args.duration, storage=args.storage,
        metadata=args.metadata, queue=args.queue, serve_metrics=args.serve_metrics,
    )
    summary = run_ingest(opts, cameras_yaml=args.cameras_yaml, only=args.camera)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
