# 2026-06-03 — Phase 1: OSS ingestion (PyAV) + data-stack decision

## Goal

Phase 1 of the roadmap (`spec/10`): RTSP/file → durable 10-sec clips → 5-sec sliding
windows → object store + metadata + queue, with camera-health metrics. Branch
`phase-1-ingestion`.

## Context / decisions

- **Media: PyAV now, GStreamer later** (user Q). PyAV (libav) for file segmentation —
  vetted dep, no system ffmpeg, in-process + testable. GStreamer (`gst 1.24` on box,
  no Python `gi` yet) is the live path: `rtspsrc ! … ! splitmuxsink` = stream-copy
  10-sec clips; plus frame-reduction *mechanism* (`videorate`, keyframe-only,
  DeepStream `nvinfer interval=N`, `motioncells`). Frame *policy* (what's worth
  inferring) stays our deterministic gate. `ingestion/sources.py` Source ABC:
  File/Rtsp (PyAV) now, GStreamerSource = the live/Phase-4 slot.
- **Data stack: Postgres + filesystem; drop Redis/MinIO from default** (user Q).
  Postgres(+pgvector) = all structured state incl. the **queue** (`SKIP LOCKED`),
  replacing Redis. Clip blobs on local NVMe (not BYTEA in PG). Redis/MinIO kept as
  *optional* backends (docker-compose profile "optional"). → `spec/10` Data stack.

## What was done

- **`ingestion/` package:** `config` (loads configs/cameras.yaml), `sources`
  (File/Rtsp/GStreamer), `segmenter` (PyAV one-pass: 10-sec clips + first-frame +
  health samples), `windows` (5-sec @2s stride, pure/tested), `thumbnailer`,
  `health` (numpy black/blur/freeze + composite, no OpenCV), `storage`
  (LocalFS default / MinIO optional), `metadata` (JSONL default / Postgres = Phase-2
  slot), `bus` (FileQueue default / PgQueue = Phase-2 slot / Redis optional),
  `metrics` (`ingest_*` Prometheus), `ingest` (orchestrator, dedup guard, sha256
  checksums), `__main__` (CLI).
- **Tests (8, pass):** window math + the real PyAV segmentation path on a synthetic
  clip + health scoring + file-backend roundtrip — hermetic (no services/big .mkv).
- **`make ingest-sample`** wired (`DURATION=`, `CAMERA=`). `.gitignore` += `data/`.
  requirements += pyyaml (+ commented psycopg/redis/minio).
- **Smoke run** (`bill_counter`, 25s): 3 clips (2688×1520 h264) + 10 windows; clips +
  thumbnails on disk; metadata JSONL with sha256 checksum + health + timestamps; queue
  files `video_segment_ready.jsonl` (3) + `ai_window_ready.jsonl` (10).

## Not done / notes

- **Health `freeze_score` too aggressive** on low-motion CCTV (billing counter scored
  0.85 frozen → health 0.15, would false-trigger camera_health_issue). Needs
  calibration vs real footage — Phase 5 golden-set tuning item, not a correctness bug.
- **Postgres rows + PgQueue = Phase 2** (need schema + psycopg). Phase 1 is file-first;
  the Phase-2 backfill adapter loads JSONL → Postgres.
- **Live RTSP (GStreamer splitmuxsink)** = Phase 1.5; Phase 1 covered files + a PyAV
  RtspSource. Prometheus scrape of `ingest_*` = Phase 3.

## Status

**Phase 1 core complete on branch `phase-1-ingestion`** (file corpus → clips/windows/
metadata/queue, tested). Not pushed. Next: Phase 2 (Postgres schema + migrations +
API + JSON→rows backfill + PgQueue), or live RTSP via GStreamer.
