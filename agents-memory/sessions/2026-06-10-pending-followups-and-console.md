# 2026-06-10 — close real-hardware follow-ups + unified Console (front door)

Cleared the "honest caveats / real-hardware follow-ups" backlog (the code-doable
ones) and built the platform's single front door. All on `main`.

## Follow-ups completed
- **C-RADIO dim** (`ai-workers/embedding-worker/plugin.py`): replaced the lossy
  truncate/pad `_to_dim` with a **Gaussian random-projection head** (`_project`,
  module cache `_PROJECTION_CACHE`) — JL, cosine-preserving, deterministic per
  (src→dst) shape so the fake-path determinism contract holds. DB stays `vector(768)`
  (keeps the pgvector index); `configs/models.yaml` embedding.vector_dim `tbd → 768`
  with the rationale. 3 new tests. **Chose projection-head over a column migration.**
- **TrackingPlugin** (`ai-workers/cv-oss-worker/tracking.py`): pure-Python greedy
  IoU/centroid `Tracker`, per-camera persistent integer ids, label-gated, stale
  retirement. Wired into `CvOssDetector._package` (real+fake), so detections + the
  `suspicious_item_interaction` event carry real ids; worker `_write_tracks()`
  upserts the `tracks` table. `Detection` got an optional trailing `track_id` field.
  rule-engine `_track_key` already preferred `track_id`/`track_ids[0]` → cam:<id>
  only as fallback (added regression tests). cv 22 pass, rule 24 pass.
- **GStreamerSource** (`ingestion/sources.py`): implemented the live-RTSP Phase-1.5
  recorder via `gst-launch-1.0` subprocess — `rtspsrc ! rtph264depay ! h264parse !
  splitmuxsink max-size-time=10s` (stream-copy, `-e` for clean final clip).
  `available()`=shutil.which; `GStreamerUnavailable` when absent; `catalog_live_clips()`
  reuses `SegmentRecord`+`_sha256` so live clips catalog like PyAV segments.
  `source_for_camera` opt-in via `INGEST_SOURCE_BACKEND=gstreamer`. 15 tests.
- **audit_log GRANTs** → migration `0003_audit_grants`: NOLOGIN group role
  `marlin_app`, INSERT+SELECT only, REVOKE UPDATE/DELETE/TRUNCATE (also from PUBLIC).
  audit.py docstring updated to point at it.
- **Temporal overlap** → migration `0004_temporal_overlap`: `btree_gist` ext +
  absolute `tstzrange span` on video_segments (generated), incidents (generated,
  null-safe), ai_windows (plain col + `trg_ai_windows_span` trigger deriving abs time
  from parent segment + backfill). Composite GiST `(camera_id|store_id, span)` for
  index-served cross-clip/cross-camera overlap joins. events left point-in-time.
  Mirrored into `db/schema.sql` (doc only — runner uses migrations/).

## Unified Console — THE front door (`services/console/`, :8080) → PLATFORM.md
- User pain: "a bunch of dashboards and views" with no cohesive entry. Built a
  **BFF/gateway**: FastAPI serves one self-contained SPA (`static/index.html`, NV-green,
  sidebar shell, hash-routed) and **proxies server-side** to upstreams (single origin,
  no CORS, auth passthrough): `/backend/*`→API:8000, `/review/*`→review-ui:8010,
  embeds Grafana:3000. Sections: Overview/Incidents(approve-reject via review svc)/
  Events/Search/Segments/Cameras/Models/Observability. Degrades to inline "offline"
  states when an upstream is down (so it demos with nothing running). 5 tests.
- Added `GET /search?q=&k=` to **services/api** (wraps embedding-worker `query_run`,
  never 500s) so search is HTTP-queryable + proxied into the console.
- Makefile: `make console` / `run-console`; console added to TESTDIRS + .PHONY.
- `PLATFORM.md` (repo root) = "Start Here": one-URL front door, port table, how to
  query (console, `/docs` Swagger, `make search`, curl). README is still the old
  src/marlin doc — PLATFORM.md supersedes it for the platform.

## One-command launch + docs refactor (later same session)
- **`make platform`** (one command, one URL): `Dockerfile.app` (torch-free app image:
  fastapi/uvicorn/httpx/psycopg[binary]/pyyaml/numpy, repo bind-mounted) + new
  `docker-compose.platform.yml` (migrate→api→review-ui→console, depends_on healthy) +
  Makefile `platform`/`platform-down`/`platform-logs` (`COMPOSE_PLATFORM`). Console
  on :8080 is the front door; `CONSOLE_API_URL`/`REVIEW_URL`/`GRAFANA_URL` envs.
- **Verified live (base+app subset, skipped obs to dodge box's :3000):** postgres
  healthy → migrate **applied 0001–0004 exit 0** (incl. new 0003 GRANTs + 0004
  tstzrange/btree_gist — proves the SQL) + seeded 5 cams/6 zones/registry → api healthy
  → console served real cameras via console→api→PG. Screenshotted the SPA earlier with
  a throwaway `/tmp/mockapi.py` (NOT in repo).
- **Docs refactor (structure hygiene + docs; user chose this, NO risky renames):**
  README rewritten as platform front door (+ preserved legacy Marlin section); spec/09
  rewritten to the real tree (services/ai-workers/ingestion/db/console/compose + hyphen-
  dir import convention + planned rename); **new spec/14** (console + one-command
  deploy); spec/10 marked C-RADIO/temporal/Phase-1.5 resolved + added a consolidated
  **"What's to be done"** (A infra-blocked / B deferred-refactor / C open-follow-ups);
  spec/README + this memory's Spec index += `14`. No Python/config/build touched →
  264-test suite unaffected.

## Still genuinely blocked on the user / real infra (cannot be coded)
- `make setup-nvidia-docker` needs **sudo** (not passwordless) → GPU containers.
- **No NGC key** → DeepStream/NIM/TAO still blocked.
- Real-GPU benchmark/bake-off numbers + real Nemotron VSS summary need weights+GPU
  (the VLM worker chat path is already real/verified; these are the remaining fakes).
- Migrations 0003/0004 are written + mirrored but **not applied here** (no Postgres
  on the dev box) — apply with `make migrate` on the box with PG up.
