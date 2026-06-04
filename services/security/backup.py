"""Backup / DR helper (spec/10 + master plan §12: backup/DR, Phase 12).

Two artifacts make the platform restorable from a cold start:
  1. a Postgres logical dump (pg_dump) of the structured store;
  2. a clips manifest — sha256 of every clip blob on the filesystem — so evidence
     integrity is verifiable after restore (matches video_segments.checksum).

Smoke-testable: `pg_dump_command()` and `clips_manifest()` build the command /
manifest WITHOUT shelling out, so tests need no Postgres and no real pg_dump.
DSN is read from the environment; it is never printed (it carries the DB password).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def pg_dump_command(out_path: str | Path, dsn: str | None = None) -> list[str]:
    """Build the pg_dump argv (not executed here).

    Uses --dbname=<dsn> so the password stays inside the URL rather than on the
    visible command line as a separate flag. Caller redacts before logging.
    """
    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        try:
            from db import dsn as _dsn
            dsn = _dsn()
        except Exception:
            dsn = ""
    return [
        "pg_dump",
        "--format=custom",      # compressed, restorable with pg_restore
        "--no-owner",
        f"--file={out_path}",
        f"--dbname={dsn}",
    ]


def sha256_file(path: str | Path, _chunk: int = 1 << 20) -> str:
    """Streamed sha256 of a file (evidence-integrity checksum)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def clips_manifest(clips_dir: str | Path, patterns: tuple[str, ...] = ("*.mkv", "*.mp4")) -> list[dict]:
    """Walk `clips_dir`, return [{path, size, sha256}] for each clip blob.

    Pure filesystem work — no DB. Empty/missing dir → empty manifest (no crash).
    """
    root = Path(clips_dir)
    if not root.exists():
        return []
    out: list[dict] = []
    for pat in patterns:
        for f in sorted(root.rglob(pat)):
            if f.is_file():
                out.append({
                    "path": str(f),
                    "size": f.stat().st_size,
                    "sha256": sha256_file(f),
                })
    return out


def run_backup(out_dir: str | Path, clips_dir: str | Path, dsn: str | None = None,
               execute: bool = False) -> dict:
    """Plan (and optionally execute) a backup. Returns a manifest dict.

    `execute=False` (default) builds the artifacts/paths without running pg_dump —
    smoke-testable. `execute=True` writes the clips manifest JSON and runs pg_dump.
    The returned dict NEVER contains the DSN (it embeds a DB password).
    """
    import json

    out = Path(out_dir)
    stamp = _stamp()
    dump_path = out / f"pg_{stamp}.dump"
    manifest_path = out / f"clips_manifest_{stamp}.json"
    cmd = pg_dump_command(dump_path, dsn)
    manifest = clips_manifest(clips_dir)

    result = {
        "stamp": stamp,
        "pg_dump_path": str(dump_path),
        "manifest_path": str(manifest_path),
        "clip_count": len(manifest),
        # redact the DSN out of the echoed command (last arg carries credentials):
        "pg_dump_cmd_redacted": cmd[:-1] + ["--dbname=***REDACTED***"],
        "executed": execute,
    }

    if execute:  # pragma: no cover - requires pg_dump + a real DB
        import subprocess
        out.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        subprocess.run(cmd, check=True)
        from .audit import audit
        audit("backup.run", actor="backup-cli", target=str(out),
              clip_count=len(manifest), pg_dump_path=str(dump_path))
    return result
