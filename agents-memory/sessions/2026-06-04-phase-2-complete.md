# 2026-06-04 — Phase 2 complete: DB + PgQueue + backfill + API

## Goal

Finish Phase 2: the Postgres job queue (`PgQueue`), the JSON→rows backfill, and the
FastAPI surface — on top of the schema/migrate/seed already landed. Branch
`phase-2-complete`.

## What was done

- **`PgQueue`** (`ingestion/bus.py`) — real Postgres queue: `publish` (INSERT),
  `consume` (`SELECT … FOR UPDATE SKIP LOCKED` → claim → mark processing),
  `ack`/`fail`/`close`. `make_queue("pg")` uses `db.dsn()` and falls back to file if
  Postgres is down. Replaces Redis as the platform queue (spec/10 Data stack).
- **Backfill** (`scripts/backfill_ingest.py`, `make backfill`) — reads
  `data/metadata/{segments,windows}.jsonl` → `video_segments` + `ai_windows`,
  idempotent (segments dedup on (camera_id,start_time); windows insert only if their
  segment exists). The file-first → DB bridge.
- **API** (`services/api/app.py`, `make api`) — FastAPI: `/health`, `/cameras`,
  `/segments`, `/events`, `/incidents`, and the model-registry endpoints (A2.5):
  `GET /api/model-registry[/active]`, `POST …/activate/{profile}`, `GET /api/model-runs`.
- **Tests (12 pass):** PgQueue publish/consume/ack roundtrip + API (`/health` always;
  cameras + registry activate skip if no DB). httpx added for the TestClient.
- **Validation:** `validate/api.py` now asserts in-process `/health` via TestClient
  (green without a running server). `doctor` all-OK (api/gpu warn only = live server /
  NVIDIA runtime).
- **Verified end-to-end:** migrate → seed → ingest (file) → `make backfill` → 3
  segments + 10 windows in Postgres; `make api` serves; 12 tests green; lint OK.

## Not done / notes

- `db/__init__.py` makes `db` a real package (needed by the validator + API imports).
- Postgres rows come via the **backfill** path now; a worker writing rows directly
  (and consuming `PgQueue`) is Phase 4/6.
- Live RTSP (GStreamer `splitmuxsink`) still Phase 1.5; events table empty until the
  rule engine (Phase 5).

## Status

**Phase 2 complete + pushed.** Critical path so far: 0 ✅ → 1 ✅ → 2 ✅. Next: Phase 3
(observability: `model_*`/`ingest_*` scrape wiring, dashboards 01–18, OTel) ∥ Phase 4
(cv-oss-worker consuming `ai_window.ready` from PgQueue, emitting the common event schema).
