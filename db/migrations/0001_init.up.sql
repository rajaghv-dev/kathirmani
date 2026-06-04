-- Phase 2 — platform schema (master plan §6 + Addendum A6) + the Postgres job queue.
-- Postgres holds ALL structured state (metadata, events, vectors, model runs, queue).
-- Clip *blobs* live on the filesystem; rows here store paths + checksums (spec/10 Data stack).
-- gen_random_uuid() is built-in on PG13+; pgvector ships in the pgvector/pgvector image.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---- Store / camera / zone (digital-twin roots) ----------------------------
CREATE TABLE stores (
  id          TEXT PRIMARY KEY,                 -- e.g. kathirmani_01 (stable, from config)
  name        TEXT NOT NULL,
  location    TEXT,
  timezone    TEXT NOT NULL DEFAULT 'Asia/Kolkata',
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE cameras (
  id          TEXT PRIMARY KEY,                 -- e.g. bill_counter
  store_id    TEXT NOT NULL REFERENCES stores(id),
  name        TEXT NOT NULL,
  role        TEXT,
  rtsp_url_env TEXT,
  enabled     BOOLEAN DEFAULT true,
  fps_target  INT,
  created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE zones (
  id          TEXT PRIMARY KEY,
  store_id    TEXT NOT NULL REFERENCES stores(id),
  camera_id   TEXT REFERENCES cameras(id),
  name        TEXT NOT NULL,
  zone_type   TEXT NOT NULL,                    -- shelf|billing_counter|entry|exit|...
  polygon_json JSONB NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- ---- Ingestion: segments + AI windows --------------------------------------
CREATE TABLE video_segments (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  store_id      TEXT NOT NULL REFERENCES stores(id),
  camera_id     TEXT NOT NULL REFERENCES cameras(id),
  start_time    TIMESTAMPTZ NOT NULL,
  end_time      TIMESTAMPTZ NOT NULL,
  duration_sec  NUMERIC NOT NULL,
  storage_path  TEXT NOT NULL,                  -- filesystem path or s3:// URI
  thumbnail_path TEXT,
  codec         TEXT,
  fps           NUMERIC,
  width         INT,
  height        INT,
  checksum      TEXT,                           -- sha256 (evidence integrity)
  health_json   JSONB,                          -- black/blur/freeze/health_score
  status        TEXT DEFAULT 'ready',
  created_at    TIMESTAMPTZ DEFAULT now(),
  UNIQUE (camera_id, start_time)                -- duplicate-segment protection
);
CREATE INDEX idx_segments_cam_time ON video_segments (camera_id, start_time DESC);

CREATE TABLE ai_windows (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  segment_id    UUID NOT NULL REFERENCES video_segments(id) ON DELETE CASCADE,
  camera_id     TEXT NOT NULL REFERENCES cameras(id),
  window_start_sec NUMERIC NOT NULL,
  window_end_sec   NUMERIC NOT NULL,
  clip_path     TEXT,
  status        TEXT DEFAULT 'pending',
  created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_windows_status ON ai_windows (status);

-- ---- Perception outputs -----------------------------------------------------
CREATE TABLE detections (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ai_window_id  UUID REFERENCES ai_windows(id) ON DELETE CASCADE,
  model_name    TEXT NOT NULL,
  label         TEXT NOT NULL,
  confidence    NUMERIC,
  bbox_json     JSONB,
  frame_time_sec NUMERIC,
  created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_detections_window ON detections (ai_window_id);

CREATE TABLE tracks (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  camera_id     TEXT REFERENCES cameras(id),
  track_external_id TEXT,
  label         TEXT,
  first_seen    TIMESTAMPTZ,
  last_seen     TIMESTAMPTZ,
  metadata_json JSONB
);

-- ---- Events (partitioned by time — high volume) ----------------------------
CREATE TABLE events (
  id            UUID DEFAULT gen_random_uuid(),
  store_id      TEXT NOT NULL REFERENCES stores(id),
  camera_id     TEXT REFERENCES cameras(id),
  segment_id    UUID,
  ai_window_id  UUID,
  event_type    TEXT NOT NULL,
  severity      TEXT DEFAULT 'info',
  confidence    NUMERIC,
  source_engine TEXT,
  event_time    TIMESTAMPTZ NOT NULL DEFAULT now(),
  metadata_json JSONB,
  created_at    TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (id, event_time)
) PARTITION BY RANGE (event_time);
CREATE TABLE events_default PARTITION OF events DEFAULT;
CREATE INDEX idx_events_type_time ON events (event_type, event_time DESC);

CREATE TABLE vlm_observations (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  event_id      UUID,
  model_name    TEXT NOT NULL,
  prompt_version TEXT,
  input_clip_path TEXT,
  raw_output    TEXT,
  parsed_json   JSONB,
  parse_success BOOLEAN,
  ttft_ms       NUMERIC,
  output_tokens INT,
  tokens_per_sec NUMERIC,
  latency_ms    NUMERIC,
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- ---- Incidents --------------------------------------------------------------
CREATE TABLE incidents (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  store_id      TEXT NOT NULL REFERENCES stores(id),
  title         TEXT NOT NULL,
  incident_type TEXT,
  severity      TEXT,
  status        TEXT DEFAULT 'open',
  start_time    TIMESTAMPTZ,
  end_time      TIMESTAMPTZ,
  summary       TEXT,
  evidence_path TEXT,
  created_at    TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE incident_events (
  incident_id   UUID REFERENCES incidents(id) ON DELETE CASCADE,
  event_id      UUID NOT NULL,
  PRIMARY KEY (incident_id, event_id)
);

CREATE TABLE embeddings (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_type   TEXT NOT NULL,
  entity_id     UUID NOT NULL,
  model_name    TEXT NOT NULL,
  embedding     vector(768),                    -- dim must match the active embed model
  text_repr     TEXT,
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- ---- Model registry / run tracking (Addendum A6) ---------------------------
CREATE TABLE model_profiles (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  profile_name  TEXT UNIQUE NOT NULL,
  description   TEXT,
  config_json   JSONB NOT NULL,
  active        BOOLEAN DEFAULT false,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE model_registry (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id      TEXT NOT NULL,
  model_family  TEXT NOT NULL,
  vendor        TEXT NOT NULL DEFAULT 'nvidia',
  modality      TEXT[] NOT NULL,
  supported_tasks TEXT[] NOT NULL,
  supported_runtimes TEXT[] NOT NULL,
  precision     TEXT[],
  license_notes TEXT,
  source_url    TEXT,
  revision      TEXT,                            -- pinned commit sha (provenance)
  metadata_json JSONB,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE model_runs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model_profile_name TEXT NOT NULL,
  model_id      TEXT NOT NULL,
  task          TEXT NOT NULL,
  runtime       TEXT NOT NULL,
  gpu_name      TEXT,
  precision     TEXT,
  input_entity_type TEXT,
  input_entity_id UUID,
  output_entity_type TEXT,
  output_entity_id UUID,
  latency_ms    NUMERIC,
  ttft_ms       NUMERIC,
  output_tokens INT,
  tokens_per_sec NUMERIC,
  parse_success BOOLEAN,
  quality_score NUMERIC,
  error         TEXT,
  created_at    TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_model_runs_profile_task ON model_runs (model_profile_name, task, created_at DESC);

CREATE TABLE model_benchmark_runs (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  benchmark_name TEXT NOT NULL,
  model_profile_name TEXT NOT NULL,
  gpu_name      TEXT,
  dataset_name  TEXT,
  total_clips   INT,
  clips_per_min_gpu NUMERIC,
  p95_latency_ms NUMERIC,
  parse_success_rate NUMERIC,
  event_precision NUMERIC,
  event_recall  NUMERIC,
  estimated_cost_per_1000_clips NUMERIC,
  metadata_json JSONB,
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- ---- Job queue (Postgres SKIP LOCKED — replaces Redis; spec/10 Data stack) --
CREATE TABLE job_queue (
  id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  stream        TEXT NOT NULL,                  -- e.g. video_segment.ready
  payload       JSONB NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending',-- pending|processing|done|failed
  attempts      INT NOT NULL DEFAULT 0,
  locked_at     TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_jobq_pending ON job_queue (stream, created_at) WHERE status = 'pending';
