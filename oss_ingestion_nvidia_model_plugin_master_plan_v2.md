# Master Plan: OSS Video Ingestion + NVIDIA-Accelerated Video AI Platform

**Version:** 1.0  
**Target hardware:** NVIDIA DGX-class server and RTX A6000-class GPU box  
**Primary use case:** Retail store video analytics, loss/theft detection, natural-language video search, event verification, and digital_twin-driven store intelligence  
**Core decision:** Keep the ingestion layer OSS and durable. Use NVIDIA DeepStream/VSS/TensorRT/NIM/Triton wherever acceleration, scale, or reference benchmarking gives clear value.

---

## 0. Executive Intent

The platform should not be a one-off CCTV demo. It should become a reusable **Video Intelligence Platform** for retail stores and similar physical spaces.

The ingestion layer must be **open, explainable, replayable, and vendor-neutral**:

```text
RTSP / video files
  → OSS ingestion
  → durable 10-sec clips
  → 5-sec AI windows
  → metadata registry
  → queue
```

After ingestion, use NVIDIA acceleration wherever it makes sense:

```text
DeepStream        → real-time CV, detection, tracking, GPU decode
VSS 3.1           → video search, summarization, RT-CV, RT-Embedding, VIOS, LVS evaluation
TensorRT          → optimized detector/tracker models
TensorRT-LLM      → optimized LLM/VLM inference path
Triton            → production inference serving
NIM               → packaged NVIDIA inference microservices
DCGM Exporter     → GPU metrics into Prometheus/Grafana
```

For model experimentation and local developer iteration, support generic serving tools:

```text
Ollama
llama.cpp
vLLM
SGLang, optional
Python FastAPI workers
```

The system should be designed so that **almost no refactoring is required later**. That means:

1. Stable contracts between layers.
2. Common event schema.
3. Common model invocation interface.
4. Common observability.
5. Pluggable inference engines.
6. Clip-first audit trail.
7. Digital-twin-first store modelling.

---

## 1. Non-Negotiable Design Principles

### 1.1 OSS ingestion is the foundation

The ingestion layer owns:

```text
RTSP reliability
video segmentation
clip timestamps
metadata
camera health
dropped-frame accounting
replayability
evidence retention
```

It must not depend on any single AI framework.

### 1.2 DeepStream is an acceleration layer, not the platform boundary

DeepStream should be used for:

```text
GPU decode
multi-stream batching
detector inference
tracking
real-time metadata generation
message publishing
```

But DeepStream should not become the only place where business logic lives.

### 1.3 AI consumes durable clips, not fragile live streams

Preferred flow:

```text
RTSP → durable clips → AI windows → events
```

Avoid:

```text
RTSP → model directly → no replay
```

Direct live inference is allowed only for low-latency path, but every event must still link back to a stored clip.

### 1.4 Evidence is more important than raw detection

For loss/theft use cases, the system must preserve:

```text
before-event context
during-event clip
after-event context
camera id
zone id
person/object track ids
model outputs
confidence
human review decision
```

### 1.5 Common schema before model complexity

Before adding many models, define the schema:

```text
VideoSegment
AIWindow
Detection
Track
ZoneEvent
VLMObservation
Incident
EvidencePackage
DigitalTwinEntity
```

---

## 2. Recommended High-Level Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                           Retail Store                                │
│                                                                      │
│  IP Cameras / NVR / Stored Video Files                                │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    OSS Video Ingestion Layer                          │
│                                                                      │
│  FFmpeg / GStreamer                                                   │
│  RTSP reconnect                                                        │
│  10-sec segment writer                                                 │
│  5-sec sliding AI-window generator                                     │
│  thumbnails/contact sheets                                             │
│  camera health metrics                                                 │
│  metadata writer                                                       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
             ┌──────────────────┼──────────────────┐
             ▼                  ▼                  ▼
┌───────────────────┐  ┌─────────────────┐  ┌──────────────────────┐
│ Object Storage     │  │ Metadata DB      │  │ Event Queue           │
│ MinIO / NVMe       │  │ Postgres         │  │ Redis Streams/Kafka   │
│ 10-sec clips       │  │ segments/events  │  │ clip/window jobs      │
└───────────────────┘  └─────────────────┘  └──────────┬───────────┘
                                                        │
                                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         AI Processing Layer                           │
│                                                                      │
│  Fast CV Path:                                                        │
│    DeepStream + TensorRT + trackers                                   │
│                                                                      │
│  VLM Path:                                                            │
│    Ollama / llama.cpp / vLLM / NIM / TensorRT-LLM / Triton with NVIDIA open models             │
│                                                                      │
│  VSS Path:                                                            │
│    VSS 3.1 evaluation for RT-CV, RT-Embedding, VIOS, LVS, search       │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Event Intelligence Layer                           │
│                                                                      │
│  zone events                                                           │
│  person/object interactions                                            │
│  suspicious-loss hypotheses                                            │
│  VLM verification                                                      │
│  incident timeline                                                     │
│  human review                                                          │
│  evidence package                                                      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   Digital Twin Reference Framework                    │
│                                                                      │
│  store → cameras → FOV → zones → shelves → billing → exit path         │
│  events mapped to physical/business context                           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  Dashboard, Grafana, Review UI                        │
│                                                                      │
│  Grafana: infra/model/GPU/queue metrics                               │
│  Custom page: camera map, timeline, incidents, evidence replay         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Dataflow

### 3.1 Normal clip ingestion flow

```text
Camera RTSP stream
  → ingestion service pulls stream
  → segmenter writes 10-sec clip
  → thumbnail/contact sheet generated
  → metadata inserted into Postgres
  → queue message published
  → AI worker consumes clip/window
  → events inserted
  → dashboard shows timeline
```

### 3.2 AI windowing flow

Use 10-sec clips for storage, but 5-sec windows for action detection.

```text
Stored clip: 00s–10s

AI windows:
  window_001: 00s–05s
  window_002: 02s–07s
  window_003: 04s–09s
  window_004: 05s–10s
```

Recommended default:

```yaml
storage_clip_duration_sec: 10
ai_window_duration_sec: 5
ai_window_stride_sec: 2
pre_event_context_sec: 5
post_event_context_sec: 10
evidence_clip_duration_sec: 20
```

### 3.3 Suspicious loss/theft event flow

```text
1. Person enters shelf zone
2. Hand/object interaction detected
3. Object disappears or person carries item
4. Person bypasses billing or moves toward exit
5. Rule engine creates suspicious hypothesis
6. VLM verifies with context clip
7. Human reviewer confirms/rejects
8. Incident package generated
```

### 3.4 Multi-camera incident reconstruction

```text
camera_left_aisle   10:30:00–10:30:10
camera_center       10:30:00–10:30:10
camera_billing      10:30:00–10:30:10
camera_entry_exit   10:30:00–10:30:10

All clips share same time bucket.
The system reconstructs event sequence across camera views.
```

---

## 4. Repository Structure

Create the repository like this from day one:

```text
video-ai-platform/
├── README.md
├── Makefile
├── .env.example
├── docker-compose.yml
├── docker-compose.gpu.yml
├── docker-compose.observability.yml
├── docs/
│   ├── 00-intent.md
│   ├── 01-architecture.md
│   ├── 02-dataflow.md
│   ├── 03-digital_twin.md
│   ├── 04-models.md
│   ├── 05-testing.md
│   ├── 06-observability.md
│   ├── 07-benchmarking.md
│   └── 08-runbooks.md
├── configs/
│   ├── cameras.yaml
│   ├── stores/
│   │   └── kathirmani.yaml
│   ├── zones/
│   │   └── kathirmani_zones.yaml
│   ├── models.yaml
│   ├── event_rules.yaml
│   └── retention.yaml
├── ingestion/
│   ├── Dockerfile
│   ├── src/
│   │   ├── ingest_rtsp.py
│   │   ├── segment_writer.py
│   │   ├── health_monitor.py
│   │   ├── thumbnailer.py
│   │   └── metadata_writer.py
│   └── tests/
├── ai_workers/
│   ├── cv-deepstream-worker/
│   │   ├── Dockerfile
│   │   ├── configs/
│   │   ├── src/
│   │   └── tests/
│   ├── cv_oss_worker/
│   │   ├── Dockerfile
│   │   ├── src/
│   │   └── tests/
│   ├── vlm_worker/
│   │   ├── Dockerfile
│   │   ├── prompts/
│   │   ├── src/
│   │   └── tests/
│   ├── vss_eval_worker/
│   │   ├── README.md
│   │   └── experiments/
│   └── embedding_worker/
├── services/
│   ├── api/
│   │   ├── Dockerfile
│   │   ├── src/
│   │   └── tests/
│   ├── rule_engine/
│   │   ├── Dockerfile
│   │   ├── src/
│   │   └── tests/
│   ├── evidence_builder/
│   │   ├── Dockerfile
│   │   ├── src/
│   │   └── tests/
│   └── review_ui/
│       ├── Dockerfile
│       ├── src/
│       └── tests/
├── db/
│   ├── migrations/
│   ├── seed/
│   └── schema.sql
├── observability/
│   ├── prometheus/
│   ├── grafana/
│   │   ├── dashboards/
│   │   └── provisioning/
│   ├── loki/
│   ├── promtail/
│   ├── otel-collector/
│   └── dcgm-exporter/
├── scripts/
│   ├── setup_host.sh
│   ├── setup_nvidia_docker.sh
│   ├── generate_test_videos.sh
│   ├── load_sample_store.sh
│   ├── run_smoke_tests.sh
│   ├── run_benchmarks.sh
│   └── export_evidence_package.sh
├── testdata/
│   ├── videos/
│   ├── expected_events/
│   └── synthetic/
└── benchmarks/
    ├── configs/
    ├── results/
    └── reports/
```

---

## 5. Core Frameworks and Databases

### 5.1 Ingestion layer

| Purpose | Recommended choice |
|---|---|
| RTSP ingest | FFmpeg and/or GStreamer |
| Segmentation | FFmpeg segment muxer or GStreamer pipeline |
| Thumbnail/contact sheet | FFmpeg |
| Metadata write | Python FastAPI/service worker |
| Health metrics | Prometheus client |
| Queue publish | Redis Streams first, Kafka later if needed |

### 5.2 Storage and metadata

| Purpose | Recommended choice |
|---|---|
| Clip storage | MinIO or local NVMe path abstraction |
| Metadata DB | Postgres |
| Vector search | pgvector first |
| Cache/queue | Redis Streams |
| Long-term event archive | Postgres partitions |
| Logs | Loki |
| Metrics | Prometheus |
| Dashboards | Grafana |

### 5.3 AI and model-serving layer

| Purpose | Recommended choice |
|---|---|
| Real-time detector/tracker | DeepStream + TensorRT |
| Generic detector experiments | YOLO/RT-DETR Python worker |
| Local VLM/LLM experiments | Ollama |
| Lightweight local serving | llama.cpp |
| Higher-throughput serving | vLLM |
| NVIDIA production serving | TensorRT-LLM, Triton, NIM |
| VSS reference benchmark | NVIDIA VSS 3.1 |
| Embedding service | VSS RT-Embedding or custom worker |

### 5.4 Observability

| Area | Tool |
|---|---|
| Application metrics | Prometheus |
| GPU metrics | NVIDIA DCGM Exporter |
| Logs | Loki + Promtail |
| Distributed traces | OpenTelemetry Collector |
| Dashboards | Grafana |
| Custom operator UI | React/Next.js or simple FastAPI + HTMX |

---

## 6. Database Schema

### 6.1 stores

```sql
CREATE TABLE stores (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  location TEXT,
  timezone TEXT NOT NULL DEFAULT 'Asia/Kolkata',
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.2 cameras

```sql
CREATE TABLE cameras (
  id UUID PRIMARY KEY,
  store_id UUID REFERENCES stores(id),
  name TEXT NOT NULL,
  rtsp_url TEXT,
  role TEXT,
  enabled BOOLEAN DEFAULT true,
  resolution TEXT,
  fps INT,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.3 zones

```sql
CREATE TABLE zones (
  id UUID PRIMARY KEY,
  store_id UUID REFERENCES stores(id),
  camera_id UUID REFERENCES cameras(id),
  name TEXT NOT NULL,
  zone_type TEXT NOT NULL,
  polygon_json JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

Zone types:

```text
shelf
billing_counter
entry
exit
staff_only
high_value_area
queue_area
blind_spot
```

### 6.4 video_segments

```sql
CREATE TABLE video_segments (
  id UUID PRIMARY KEY,
  store_id UUID REFERENCES stores(id),
  camera_id UUID REFERENCES cameras(id),
  start_time TIMESTAMPTZ NOT NULL,
  end_time TIMESTAMPTZ NOT NULL,
  duration_sec INT NOT NULL,
  storage_path TEXT NOT NULL,
  thumbnail_path TEXT,
  codec TEXT,
  fps NUMERIC,
  width INT,
  height INT,
  checksum TEXT,
  status TEXT DEFAULT 'ready',
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.5 ai_windows

```sql
CREATE TABLE ai_windows (
  id UUID PRIMARY KEY,
  segment_id UUID REFERENCES video_segments(id),
  window_start_sec NUMERIC NOT NULL,
  window_end_sec NUMERIC NOT NULL,
  status TEXT DEFAULT 'pending',
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.6 detections

```sql
CREATE TABLE detections (
  id UUID PRIMARY KEY,
  ai_window_id UUID REFERENCES ai_windows(id),
  model_name TEXT NOT NULL,
  label TEXT NOT NULL,
  confidence NUMERIC,
  bbox_json JSONB,
  frame_time_sec NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.7 tracks

```sql
CREATE TABLE tracks (
  id UUID PRIMARY KEY,
  camera_id UUID REFERENCES cameras(id),
  track_external_id TEXT,
  label TEXT,
  first_seen TIMESTAMPTZ,
  last_seen TIMESTAMPTZ,
  metadata_json JSONB
);
```

### 6.8 events

```sql
CREATE TABLE events (
  id UUID PRIMARY KEY,
  store_id UUID REFERENCES stores(id),
  camera_id UUID REFERENCES cameras(id),
  segment_id UUID REFERENCES video_segments(id),
  ai_window_id UUID REFERENCES ai_windows(id),
  event_type TEXT NOT NULL,
  severity TEXT DEFAULT 'info',
  confidence NUMERIC,
  source_engine TEXT,
  event_time TIMESTAMPTZ,
  metadata_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.9 vlm_observations

```sql
CREATE TABLE vlm_observations (
  id UUID PRIMARY KEY,
  event_id UUID REFERENCES events(id),
  model_name TEXT NOT NULL,
  prompt_version TEXT,
  input_clip_path TEXT,
  raw_output TEXT,
  parsed_json JSONB,
  parse_success BOOLEAN,
  ttft_ms NUMERIC,
  output_tokens INT,
  tokens_per_sec NUMERIC,
  latency_ms NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.10 incidents

```sql
CREATE TABLE incidents (
  id UUID PRIMARY KEY,
  store_id UUID REFERENCES stores(id),
  title TEXT NOT NULL,
  incident_type TEXT,
  severity TEXT,
  status TEXT DEFAULT 'open',
  start_time TIMESTAMPTZ,
  end_time TIMESTAMPTZ,
  summary TEXT,
  evidence_path TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);
```

### 6.11 incident_events

```sql
CREATE TABLE incident_events (
  incident_id UUID REFERENCES incidents(id),
  event_id UUID REFERENCES events(id),
  PRIMARY KEY (incident_id, event_id)
);
```

### 6.12 embeddings

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE embeddings (
  id UUID PRIMARY KEY,
  entity_type TEXT NOT NULL,
  entity_id UUID NOT NULL,
  model_name TEXT NOT NULL,
  embedding vector(768),
  text_repr TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## 7. Configuration Files

### 7.1 cameras.yaml

```yaml
store_id: kathirmani_01
timezone: Asia/Kolkata

cameras:
  - id: bill_counter
    name: "Bill Counter"
    role: billing
    rtsp_url_env: RTSP_BILL_COUNTER
    fps_target: 10
    segment_sec: 10
    enabled: true

  - id: center
    name: "Center Camera"
    role: overview
    rtsp_url_env: RTSP_CENTER
    fps_target: 10
    segment_sec: 10
    enabled: true

  - id: left_aisle
    name: "Left Aisle"
    role: shelf
    rtsp_url_env: RTSP_LEFT_AISLE
    fps_target: 10
    segment_sec: 10
    enabled: true

  - id: entry_exit
    name: "Entry Exit"
    role: entry_exit
    rtsp_url_env: RTSP_ENTRY_EXIT
    fps_target: 10
    segment_sec: 10
    enabled: true

  - id: right_aisle
    name: "Right Aisle"
    role: shelf
    rtsp_url_env: RTSP_RIGHT_AISLE
    fps_target: 10
    segment_sec: 10
    enabled: true
```

### 7.2 models.yaml

```yaml
detectors:
  primary:
    engine: deepstream
    model_name: retail_person_object_detector
    precision: fp16
    input_size: [640, 640]
    labels:
      - person
      - bag
      - item
      - shelf
      - basket
      - trolley

trackers:
  primary:
    engine: deepstream
    tracker: nvdcf
    reid_enabled: true

vlm:
  default:
    provider: ollama
    model: nvidia-nemotron-vl-compatible
    endpoint: http://ollama:11434
    timeout_sec: 120

  high_throughput:
    provider: vllm
    model: selected-vlm
    endpoint: http://vllm:8000/v1
    timeout_sec: 120

  nvidia_prod:
    provider: nim_or_triton
    model: selected-nvidia-vlm
    endpoint: http://nim-vlm:8000/v1
    timeout_sec: 120

vss:
  enabled: false
  endpoint: http://vss:8000
  workflows:
    - rt_cv
    - rt_embedding
    - lvs
    - search
```

### 7.3 event_rules.yaml

```yaml
rules:
  - id: suspicious_item_interaction
    description: "Person interacts with item in shelf zone"
    conditions:
      - person_in_zone: shelf
      - object_near_person_hand: true
      - dwell_time_sec_gte: 3
    action:
      create_event: suspicious_item_interaction
      needs_vlm_verification: true

  - id: possible_billing_bypass
    description: "Person carrying item moves to exit without billing interaction"
    conditions:
      - person_has_item: true
      - visited_billing_zone: false
      - moved_to_zone: exit
    action:
      create_incident: possible_loss
      needs_vlm_verification: true

  - id: camera_health_issue
    description: "Camera blocked, black frame, frozen frame, or low visibility"
    conditions:
      - camera_health_score_lte: 0.5
    action:
      create_event: camera_health_issue
      severity: warning
```

---

## 8. API Contracts

### 8.1 Queue message: new video segment

```json
{
  "type": "video_segment.ready",
  "segment_id": "uuid",
  "store_id": "kathirmani_01",
  "camera_id": "left_aisle",
  "start_time": "2026-06-01T10:30:00+05:30",
  "end_time": "2026-06-01T10:30:10+05:30",
  "storage_path": "s3://clips/kathirmani/left_aisle/2026/06/01/10-30-00.mp4"
}
```

### 8.2 Queue message: new AI window

```json
{
  "type": "ai_window.ready",
  "window_id": "uuid",
  "segment_id": "uuid",
  "camera_id": "left_aisle",
  "window_start_sec": 2,
  "window_end_sec": 7,
  "clip_path": "s3://clips/..."
}
```

### 8.3 Common event output

```json
{
  "event_id": "uuid",
  "store_id": "kathirmani_01",
  "camera_id": "left_aisle",
  "segment_id": "uuid",
  "ai_window_id": "uuid",
  "event_type": "suspicious_item_interaction",
  "severity": "medium",
  "confidence": 0.74,
  "source_engine": "deepstream_rule_engine",
  "objects": ["person", "item", "bag"],
  "track_ids": ["person_17"],
  "zones": ["shelf_zone_A"],
  "needs_vlm_verification": true,
  "evidence_path": "s3://clips/..."
}
```

### 8.4 VLM verification output

```json
{
  "verdict": "suspicious",
  "confidence": 0.68,
  "observed_actions": [
    "person reached toward shelf",
    "person handled a small item",
    "person moved away from shelf"
  ],
  "missing_evidence": [
    "billing counter interaction not visible in this camera"
  ],
  "recommended_next_step": "check synchronized billing and exit camera clips",
  "structured_event_type": "possible_loss",
  "explanation": "The clip shows item interaction and concealment-like movement, but billing context is needed."
}
```

---

## 9. Model Strategy

### 9.1 Model classes

Use separate models for separate jobs.

```text
Detector model:
  Find persons, bags, items, shelves, baskets.

Tracker:
  Maintain IDs over time within a camera.

ReID model:
  Help associate same person across multiple camera views.

VLM model:
  Explain suspicious windows and verify event hypotheses.

Embedding model:
  Enable search over clips/events/incidents.

LLM model:
  Summarize incidents, generate reports, answer questions.
```

### 9.2 Do not use VLM for everything

Bad:

```text
Every 5-sec clip → VLM → event
```

Better:

```text
Every 5-sec clip → detector/tracker/rules
Only suspicious windows → VLM verification
```

This saves GPU time and improves TCO.

### 9.3 Serving choices

#### Development

```text
Ollama
llama.cpp
Python FastAPI wrapper
```

Use for quick experiments, prompts, demos, and local tests.

#### Benchmark

```text
vLLM
Triton
TensorRT-LLM
NIM
```

Use for tokens/sec, TTFT, concurrency, and GPU memory tests.

#### Production CV path

```text
DeepStream
TensorRT
Triton
```

Use for multi-camera real-time throughput.

---

## 10. DeepStream Integration Plan

### 10.1 Where DeepStream fits

DeepStream should consume either:

```text
A. live RTSP stream
B. stored video segment
C. file list from queue
```

Preferred for controlled pipeline:

```text
stored 10-sec clips → DeepStream batch worker → detections/tracks/events
```

Use live RTSP DeepStream only for low-latency pilot mode.

### 10.2 DeepStream outputs

DeepStream worker must publish only structured metadata to the platform:

```text
detections
tracks
zone crossings
object interactions
health events
```

It must not directly decide final theft verdict.

### 10.3 Required DeepStream tests

```text
1. Can process one 10-sec clip.
2. Can process 100 clips from queue.
3. Can process 5 synchronized camera clips.
4. Can process N live streams without crashing when one stream drops.
5. Emits events matching common schema.
6. GPU utilization is visible in Grafana.
7. p95 latency is recorded.
```

---

## 11. VSS 3.1 Evaluation Plan

VSS 3.1 should be treated as a **reference and benchmark path**, not necessarily the only implementation.

Evaluate:

```text
VIOS:
  video input/output/storage/recording/playback

RT-CV:
  real-time object detection/tracking

RT-Embedding:
  embeddings from live streams and files

LVS:
  long video summarization

Agent/Search:
  natural-language video query and incident report
```

### 11.1 VSS experiments

```text
Experiment 1:
  Upload 30-min retail footage.
  Ask for timestamped suspicious interactions.

Experiment 2:
  Run natural-language search:
  "Show people picking up an item and walking away."

Experiment 3:
  Compare VSS RT-CV events against custom DeepStream worker events.

Experiment 4:
  Compare VSS embeddings against custom pgvector embeddings.

Experiment 5:
  Compare VSS long-video summaries with human-created summaries.
```

### 11.2 VSS acceptance criteria

```text
VSS is useful if:
  - search result timestamps are accurate enough
  - summaries are explainable
  - event extraction can be integrated through API
  - GPU cost is acceptable
  - data can be linked back to our evidence clips
```

---

## 12. Digital Twin Reference Framework

The digital twin should be a structured representation of the store.

### 12.1 Entities

```text
Store
Floor
Camera
Camera Field of View
Zone
Shelf
Product Area
Billing Counter
Entry
Exit
Staff Area
High-Value Area
Person Track
Object Track
Event
Incident
Evidence Clip
```

### 12.2 Relationships

```text
camera observes zone
zone contains shelf
shelf contains product area
person enters zone
person exits zone
person interacts with object
object belongs to shelf
person approaches billing counter
person exits store
event belongs to clip
incident consists of events
```

### 12.3 Why this matters

Without a digital twin:

```text
Camera 3 saw person near shelf
```

With a digital twin:

```text
Person track 17 entered high-value shelf zone A,
handled an item for 4.2 sec,
did not visit billing zone,
and appeared in exit camera 12 sec later.
```

The second one is reusable across stores.

---

## 13. Observability

### 13.1 Metrics to capture

#### Ingestion metrics

```text
rtsp_connected
rtsp_reconnect_count
frames_received_total
frames_dropped_total
segments_written_total
segment_write_latency_ms
camera_fps_actual
camera_black_frame_score
camera_blur_score
camera_freeze_score
```

#### Queue metrics

```text
queue_depth
jobs_pending
jobs_processing
jobs_failed
oldest_job_age_sec
```

#### CV metrics

```text
cv_clips_processed_total
cv_fps_per_gpu
cv_latency_ms_p50
cv_latency_ms_p95
detections_per_clip
tracks_active
tracker_id_switches
```

#### VLM metrics

```text
vlm_requests_total
vlm_latency_ms_p50
vlm_latency_ms_p95
vlm_ttft_ms
vlm_output_tokens_total
vlm_tokens_per_sec
vlm_parse_success_rate
vlm_verdict_distribution
```

#### GPU metrics

```text
gpu_utilization
gpu_memory_used
gpu_memory_total
gpu_power_draw
gpu_temperature
gpu_encoder_utilization
gpu_decoder_utilization
```

Use NVIDIA DCGM Exporter for GPU metrics.

### 13.2 Grafana dashboards

Create these dashboards from day one:

```text
01 - Ingestion Health
02 - Camera Health
03 - Queue and Backlog
04 - CV Worker Performance
05 - VLM Worker Performance
06 - GPU Utilization and Memory
07 - Event Quality
08 - Incidents and Human Review
09 - Store Digital Twin View
10 - TCO Benchmark
```

### 13.3 Custom UI pages

Grafana is good for metrics. A custom page is required for operations.

Recommended pages:

```text
/store-map
  store layout, cameras, zones

/cameras
  live/saved preview, camera health

/timeline
  events by time and camera

/incidents
  suspicious events, status, reviewer

/evidence/:incident_id
  clips, model outputs, human decision

/search
  natural-language video search

/benchmarks
  GPU/model/run comparison
```

---

## 14. Setup Plan

### 14.1 Host prerequisites

Target OS:

```text
Ubuntu Server LTS
NVIDIA Driver
NVIDIA Container Toolkit
Docker
Docker Compose
CUDA-compatible runtime
```

### 14.2 Base services

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16

  redis:
    image: redis:7

  minio:
    image: minio/minio

  prometheus:
    image: prom/prometheus

  grafana:
    image: grafana/grafana

  loki:
    image: grafana/loki

  promtail:
    image: grafana/promtail

  otel-collector:
    image: otel/opentelemetry-collector

  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter
```

### 14.3 Application services

```yaml
services:
  ingestion:
    build: ./ingestion

  api:
    build: ./services/api

  rule_engine:
    build: ./services/rule_engine

  evidence_builder:
    build: ./services/evidence_builder

  cv-deepstream-worker:
    build: ./ai_workers/cv-deepstream-worker
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  vlm_worker:
    build: ./ai_workers/vlm_worker
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  review_ui:
    build: ./services/review_ui
```

---

## 15. Phased Implementation Plan

## Phase 0: Repository and architecture skeleton

### Goal

Create the full repo structure, configs, Makefile, Docker Compose files, and documentation.

### Deliverables

```text
repo scaffold
README
docs
config templates
docker-compose base
.env.example
```

### Tests

```text
make lint
make config-check
make docker-config
```

### Done when

```text
A new developer can clone repo, read README, and understand all layers.
```

---

## Phase 1: OSS ingestion layer

### Goal

Build robust RTSP/file ingestion and clip segmentation.

### Deliverables

```text
FFmpeg/GStreamer ingestion service
10-sec clip writer
5-sec AI-window metadata
thumbnail generator
Postgres video_segments rows
Redis queue messages
camera health metrics
```

### Tests

```text
Unit:
  segment path generation
  timestamp alignment
  metadata validation

Integration:
  ingest one sample video file
  produce 10-sec clips
  create AI windows
  insert DB rows
  publish queue messages

Failure:
  invalid RTSP URL
  dropped connection
  disk full simulation
  duplicate segment protection

Performance:
  ingest 5 camera files simultaneously
  measure segment latency
```

### Done when

```text
Sample 5-camera footage is segmented, registered, queued, and visible in Grafana.
```

---

## Phase 2: Metadata DB and API

### Goal

Create the queryable platform backend.

### Deliverables

```text
Postgres migrations
FastAPI backend
segment APIs
camera APIs
event APIs
incident APIs
health APIs
```

### Tests

```text
Unit:
  schema validation
  API request/response validation

Integration:
  create store
  create camera
  create zone
  create segment
  query timeline

DB:
  migration up/down
  indexes exist
  partition strategy tested
```

### Done when

```text
Review UI and workers can use stable API contracts.
```

---

## Phase 3: Grafana and observability foundation

### Goal

Make the platform easy to follow.

### Deliverables

```text
Prometheus scrape configs
Grafana dashboards
Loki logs
OpenTelemetry traces
DCGM GPU dashboard
```

### Tests

```text
Metrics:
  ingestion metrics visible
  queue metrics visible
  DB metrics visible
  GPU metrics visible

Logs:
  each service writes structured logs
  logs searchable by trace_id, camera_id, segment_id

Traces:
  segment ingestion → queue → worker → event visible
```

### Done when

```text
A failed event can be traced from RTSP input to model output.
```

---

## Phase 4: DeepStream CV worker

### Goal

Add accelerated detector/tracker worker.

### Deliverables

```text
DeepStream worker container
TensorRT detector config
tracker config
zone event mapper
event schema publisher
GPU metrics
```

### Tests

```text
Functional:
  process one clip
  output detections
  output tracks
  map objects to zones

Integration:
  consume ai_window.ready
  write detections/tracks/events
  publish suspicious hypothesis

Failure:
  corrupt clip
  missing model
  GPU unavailable
  invalid config

Performance:
  clips/min/GPU
  FPS/GPU
  p50/p95 latency
  GPU memory
```

### Done when

```text
DeepStream worker emits common event schema and is visible in Grafana.
```

---

## Phase 5: Rule engine and event intelligence

### Goal

Convert low-level detections into retail-relevant hypotheses.

### Deliverables

```text
zone crossing rules
dwell-time rules
person-object interaction rules
billing bypass hypothesis
camera health rules
incident creation
```

### Tests

```text
Unit:
  person in polygon
  dwell time
  event sequence matching

Integration:
  detection stream → rule engine → event
  multi-window event aggregation
  incident creation

Quality:
  golden test clips produce expected events
  false positives are logged
```

### Done when

```text
System can raise explainable suspicious-item-interaction events without VLM.
```

---

## Phase 6: VLM worker using Ollama/llama.cpp/vLLM

### Goal

Verify suspicious windows and produce structured explanations.

### Deliverables

```text
VLM worker
prompt templates
JSON parser
retry/repair logic
model metrics
prompt versioning
```

### Tests

```text
Unit:
  prompt rendering
  JSON parsing
  schema validation

Integration:
  suspicious event → VLM job
  VLM output → vlm_observations table
  event confidence updated

Quality:
  golden clips
  expected verdicts
  parse success rate > 95%

Performance:
  TTFT
  tokens/sec
  clips/min/GPU
  p95 latency
```

### Done when

```text
Suspicious events receive model explanation and reviewer-ready evidence.
```

---

## Phase 7: Evidence builder and human review UI

### Goal

Make incidents reviewable, auditable, and useful.

### Deliverables

```text
evidence package builder
pre/during/post event clip stitching
timeline generator
review UI
human verdict capture
export report
```

### Tests

```text
Functional:
  build 20-sec evidence clip
  attach model outputs
  attach timeline
  reviewer can approve/reject

Integration:
  event → incident → evidence → review

Audit:
  every verdict stores reviewer, timestamp, model version
```

### Done when

```text
A human can review a theft/loss hypothesis in under 60 seconds.
```

---

## Phase 8: Natural-language search and embeddings

### Goal

Enable search across clips/events/incidents.

### Deliverables

```text
embedding worker
pgvector tables
search API
search UI
event text representation
clip caption indexing
```

### Tests

```text
Functional:
  search by event type
  search by text
  search by time/camera/zone

Quality:
  recall@k on golden queries
  timestamp accuracy

Performance:
  embedding throughput
  query latency
```

### Done when

```text
User can search: “show where a person picked an item and walked away.”
```

---

## Phase 9: VSS 3.1 benchmark path

### Goal

Benchmark NVIDIA VSS against custom platform paths.

### Deliverables

```text
VSS eval worker
VSS dataset adapter
VSS results importer
comparison reports
```

### Tests

```text
Experiments:
  RT-CV comparison
  RT-Embedding comparison
  Long Video Summarization comparison
  Natural-language search comparison

Metrics:
  accuracy
  latency
  GPU usage
  ease of integration
  cost per video hour
```

### Done when

```text
We know which VSS components should be adopted, wrapped, or used only as benchmark.
```

---

## Phase 10: Digital Twin Reference Framework

### Goal

Create reusable store abstraction.

### Deliverables

```text
store twin YAML
camera FOV mapping
zone editor/importer
shelf/counter/entry/exit model
event-to-twin mapping
digital twin UI page
```

### Tests

```text
Functional:
  load store twin
  validate camera-zone relationships
  map events to zones

Integration:
  event timeline overlays onto store map

Reuse:
  onboard second store with same schema
```

### Done when

```text
A new store can be configured by YAML/layout, not by code changes.
```

---

## Phase 11: Benchmarking and TCO

### Goal

Measure across DGX/A6000 GPU configurations.

### Metrics

```text
streams/GPU
FPS/GPU
clips/min/GPU
video-sec/sec/GPU
VLM TTFT
VLM tokens/sec
p95 event latency
GPU memory
GPU utilization
power draw
cost per 1,000 clips
cost per camera/month
cost per store/month
```

### Tests

```text
Benchmark 1:
  5 camera streams

Benchmark 2:
  16 camera streams

Benchmark 3:
  32 camera streams

Benchmark 4:
  1 hour archived video summarization

Benchmark 5:
  100 suspicious-event VLM verifications

Benchmark 6:
  natural-language search over 24 hours of footage
```

### Done when

```text
We can tell which GPU/server profile gives best TCO.
```

---

## Phase 12: Production hardening

### Goal

Make it reliable enough for pilot deployment.

### Deliverables

```text
auth
RBAC
retention policies
backup/restore
disaster recovery
data cleanup
alert routing
model rollback
config versioning
incident audit logs
```

### Tests

```text
Security:
  API auth required
  object storage permissions
  secrets not logged

Reliability:
  restart services
  recover queue
  recover DB
  replay from stored clips

Data:
  retention cleanup
  evidence lock
  incident export
```

### Done when

```text
The system survives restarts, camera failures, bad clips, model failures, and reviewer workflows.
```

---

## 16. Test Strategy Summary

### 16.1 Unit tests

```text
config parsing
timestamp generation
polygon logic
event rule logic
prompt rendering
JSON parser
DB repository functions
```

### 16.2 Integration tests

```text
sample video → clips → DB → queue
queue → CV worker → detections
detections → rule engine → event
event → VLM worker → observation
event → evidence builder → incident package
```

### 16.3 End-to-end tests

```text
5-camera retail scenario
shelf interaction scenario
billing bypass scenario
camera health scenario
normal customer scenario
false-positive scenario
```

### 16.4 Performance tests

```text
clips/min/GPU
streams/GPU
p95 latency
GPU memory
queue backlog
DB write rate
search latency
```

### 16.5 Quality tests

```text
event precision
event recall
false positives per hour
false negatives on golden clips
VLM parse success
human reviewer agreement
```

---

## 17. Golden Dataset Plan

Create a small golden dataset immediately.

```text
golden_dataset/
├── normal_customer/
├── shelf_pick_and_bill/
├── shelf_pick_and_exit/
├── staff_restocking/
├── queue_at_counter/
├── camera_blocked/
├── low_light/
├── multi_camera_same_person/
└── ambiguous_cases/
```

Each clip should have:

```json
{
  "clip_id": "golden_001",
  "expected_events": [
    {
      "event_type": "suspicious_item_interaction",
      "start_sec": 3.2,
      "end_sec": 6.5,
      "zones": ["shelf_zone_A"]
    }
  ],
  "expected_verdict": "suspicious",
  "notes": "Person picks item and hides it in bag."
}
```

---

## 18. Navigation for Developers

The README should expose these commands:

```bash
make setup
make up
make down
make logs
make migrate
make seed
make ingest-sample
make run-workers
make test
make test-e2e
make bench
make grafana
make evidence-demo
```

Recommended first run:

```bash
cp .env.example .env
make setup
make up
make migrate
make seed
make ingest-sample
make run-workers
make grafana
```

---

## 19. Acceptance Criteria for the Whole Platform

The platform is acceptable when:

```text
1. Five camera feeds can be ingested.
2. 10-sec clips are stored durably.
3. 5-sec windows are created.
4. DeepStream worker emits detections/tracks/events.
5. Rule engine creates suspicious hypotheses.
6. VLM verifies suspicious clips.
7. Evidence package is generated.
8. Human reviewer can approve/reject.
9. Grafana shows ingestion, queue, AI, GPU, and incident metrics.
10. Custom UI shows store map, event timeline, and evidence replay.
11. Benchmark report compares DGX/A6000 configurations.
12. VSS 3.1 benchmark path is evaluated.
13. A second store can be onboarded without code refactoring.
```

---

## 20. Strong Engineering Warnings

### 20.1 Do not build model-first

Wrong:

```text
Choose VLM → feed all videos → hope for theft detection
```

Right:

```text
Define store twin → define events → ingest clips → detect/track → reason → verify
```

### 20.2 Do not put business logic only inside DeepStream configs

DeepStream should emit facts. The platform should decide incidents.

### 20.3 Do not skip evidence package

Without evidence, the platform is only a demo.

### 20.4 Do not optimize tokens/sec alone

Measure:

```text
video-sec/sec/GPU
clips/min/GPU
event precision/recall
p95 latency
cost per store
```

### 20.5 Do not delay observability

Grafana should be available from Phase 1.

---

## 21. Final Target Architecture Statement

The final platform should be described as:

```text
An OSS-ingestion-first, NVIDIA-accelerated, clip-evidence-based video intelligence platform
for retail store analytics, loss/theft detection, natural-language video search,
incident verification, and digital_twin-driven multi-store deployment.
```

The durable design is:

```text
OSS ingestion gives control.
DeepStream gives real-time GPU acceleration.
VSS gives reference workflows and benchmark comparison.
Ollama/llama.cpp/vLLM give model experimentation.
TensorRT-LLM/Triton/NIM give production serving options.
Postgres/MinIO/Redis/Grafana give platform durability and visibility.
Digital Twin gives repeatability across stores.
```

---

## 22. Appendix: Minimal Prompt for Agents/Coding Tools

Use this prompt when asking coding agents to generate the repository:

```text
Build a production-oriented retail video AI platform with OSS video ingestion and NVIDIA-accelerated AI layers.

Hard requirements:
1. Ingestion layer must use OSS tools such as FFmpeg/GStreamer.
2. Ingestion must save durable 10-second clips and create 5-second sliding AI windows.
3. Store all metadata in Postgres.
4. Store clips in MinIO/local object storage abstraction.
5. Use Redis Streams initially for job queue.
6. Add DeepStream worker for detection/tracking wherever possible.
7. Add VLM worker supporting Ollama first, with clean adapters for llama.cpp, vLLM, TensorRT-LLM, Triton, and NIM.
8. Add VSS 3.1 evaluation path as a benchmark/reference workflow.
9. Add common event schema so workers are pluggable.
10. Add rule engine for retail loss/theft hypotheses.
11. Add evidence package builder with pre/during/post event clips.
12. Add Grafana dashboards from Phase 1.
13. Add custom UI for store map, camera timeline, incidents, evidence replay, and search.
14. Add Digital Twin Reference Framework using YAML store/camera/zone definitions.
15. Add complete tests: unit, integration, end-to-end, performance, quality, and failure tests.
16. Add benchmark scripts for DGX/A6000-class GPU boxes.
17. No hardware-specific Intel/AMD/OpenVINO dependencies.
18. Keep all interfaces stable to avoid refactoring later.

Generate repo with clear phases, Makefile commands, Docker Compose files, migrations, schemas, configs, tests, and documentation.
```

---

## 23. Appendix: Initial Dashboard List

```text
01 Ingestion Health:
  RTSP state, reconnects, dropped frames, segment count

02 Camera Health:
  black frame, blur, frozen frame, FPS actual

03 Queue Health:
  pending jobs, failed jobs, oldest job age

04 DeepStream CV:
  clips/min, FPS, detections, tracker IDs, latency

05 VLM Worker:
  TTFT, tokens/sec, parse success, verdict distribution

06 GPU:
  utilization, memory, temperature, power, encoder/decoder

07 Event Intelligence:
  events by type, severity, confidence, false positives

08 Incidents:
  open/closed incidents, reviewer decisions, evidence status

09 Digital Twin:
  events by zone, camera coverage, blind spots

10 TCO:
  GPU profile, streams/GPU, clips/min/GPU, cost/camera/month
```

---

## 24. Appendix: First 2-Week Execution Plan

### Week 1

```text
Day 1:
  repo scaffold, Docker Compose, Postgres, Redis, MinIO, Grafana

Day 2:
  ingestion service for local video files

Day 3:
  RTSP ingestion and 10-sec clip segmentation

Day 4:
  metadata DB and queue messages

Day 5:
  Grafana ingestion dashboard and smoke tests
```

### Week 2

```text
Day 6:
  DeepStream worker skeleton

Day 7:
  detector/tracker event schema

Day 8:
  rule engine for zone/dwell/person-object events

Day 9:
  Ollama VLM worker and prompt/parser tests

Day 10:
  evidence builder, basic review UI, end-to-end demo
```

By the end of Week 2, the platform should demonstrate:

```text
sample store video
→ clips
→ detections
→ suspicious event
→ VLM verification
→ evidence package
→ Grafana + review UI
```


---

# Addendum A: NVIDIA-Only Open Model Plugin Layer and VSS-Inspired OSS Capabilities

**Design update:** The platform should use **OSS ingestion + OSS orchestration**, learn capability patterns from NVIDIA VSS, but implement those capabilities ourselves using **NVIDIA open models and NVIDIA acceleration paths**.

VSS should be treated as a reference for product capability design:

```text
VSS teaches us what capabilities matter:
  - real-time CV
  - real-time embeddings
  - long video summarization
  - video input/output/storage patterns
  - search agents
  - attribute search
  - multi-embedding fusion
  - result critic/reviewer agents
```

But the production implementation should remain:

```text
OSS platform
+ NVIDIA open models
+ pluggable model registry
+ DeepStream/TensorRT/Triton/NIM where useful
+ Grafana model performance dashboards
```

Do **not** hardwire VSS as the production dependency.

---

## A1. Revised Model Policy

### A1.1 Allowed model families

The model catalog should prioritize NVIDIA open model families and NVIDIA-published model assets.

```text
NVIDIA Nemotron family:
  - Nemotron language/reasoning models
  - Nemotron vision-language models
  - Nemotron Omni models where available
  - Nemotron retrieval/embedding/safety models where available

NVIDIA Cosmos family:
  - Cosmos video/world/physical-AI models
  - Cosmos reasoning/simulation/action models
  - Cosmos models for digital twin and synthetic scenario exploration

NVIDIA CV / Metropolis-style assets:
  - person detection models
  - object detection models
  - ReID models
  - action recognition models where available
  - DeepStream-compatible TAO/TensorRT models

NVIDIA embedding models:
  - NVIDIA-published visual/text/video embedding models
  - RADIO-CLIP / SigLIP-style models when exposed in NVIDIA workflows
  - NVIDIA retrieval models where available

NVIDIA safety/guard models:
  - content safety
  - policy filtering
  - output validation
```

### A1.2 Disallowed default model posture

Avoid making non-NVIDIA open models the default in this NVIDIA-facing benchmark branch.

Do not make these the default model families:

```text
Qwen
InternVL
LLaVA
Gemma
Phi
DeepSeek
OpenAI models
Gemini models
Claude models
```

They may be useful in separate research branches, but the current plan is:

```text
NVIDIA GPU stack
+ NVIDIA open models
+ NVIDIA acceleration paths
+ OSS control plane
```

### A1.3 Runtime and model are separate

The model should be changeable without refactoring code.

```text
model family  ≠ runtime
runtime       ≠ business logic
business rule ≠ model prompt
```

Example:

```text
Model:
  nvidia/nemotron-nano-v2-vl

Runtime:
  ollama / llama.cpp / vLLM / Triton / NIM / TensorRT-LLM

Task:
  suspicious_event_verification

Prompt:
  retail_loss_verifier_v3

Output schema:
  SuspiciousEventVerification
```

---

## A2. Model Plugin Architecture

### A2.1 Plugin interface

Every model plugin must implement the same interface.

```python
class ModelPlugin:
    def load(self, config: dict) -> None:
        ...

    def infer(self, request: dict) -> dict:
        ...

    def health(self) -> dict:
        ...

    def metrics(self) -> dict:
        ...

    def unload(self) -> None:
        ...
```

### A2.2 Task-specific plugin contracts

Use task contracts, not model-specific contracts.

```text
DetectionPlugin:
  input: frame batch or clip path
  output: detections

TrackingPlugin:
  input: detections + frames
  output: tracks

EmbeddingPlugin:
  input: clip/text/image/event
  output: embedding vector + metadata

VLMClipReasoningPlugin:
  input: clip path + prompt + event hypothesis
  output: structured JSON

SummarizationPlugin:
  input: list of clips/events/time range
  output: timestamped summary

SearchCriticPlugin:
  input: search query + candidate results
  output: reranked/critic-reviewed results
```

### A2.3 Plugin registry

The platform should load models from config.

```yaml
model_registry:
  active_profile: nvidia_a6000_retail_v1

  profiles:
    nvidia_a6000_retail_v1:
      detection:
        plugin: deepstream_detector
        model_id: nvidia/retail-person-object-detector
        runtime: deepstream_tensorrt
        precision: fp16
        batch_size: 8
        enabled: true

      tracking:
        plugin: deepstream_tracker
        tracker: nvdcf
        reid_model_id: nvidia/reid-retail-compatible
        enabled: true

      vlm_clip_reasoning:
        plugin: nvidia_vlm_openai_compatible
        model_id: nvidia/nemotron-nano-v2-vl
        runtime: vllm
        endpoint: http://vllm-nemotron-vl:8000/v1
        prompt_pack: retail_loss_v1
        max_input_frames: 16
        max_output_tokens: 512
        temperature: 0.1
        enabled: true

      embeddings:
        plugin: nvidia_embedding
        model_id: nvidia/radio-clip-or-compatible
        runtime: triton
        endpoint: http://triton-embedding:8000
        vector_dim: 768
        enabled: true

      summarization:
        plugin: nvidia_nemotron_summary
        model_id: nvidia/nemotron-reasoning-or-omni
        runtime: nim_or_vllm
        endpoint: http://nemotron-summary:8000/v1
        prompt_pack: retail_lvs_v1
        enabled: true

      digital_twin_simulation:
        plugin: nvidia_cosmos
        model_id: nvidia/cosmos-3-or-compatible
        runtime: nvidia_runtime
        endpoint: http://cosmos-worker:8000
        enabled: false
```

### A2.4 Hot-swapping models

The config should allow changing model without code changes.

```bash
MODEL_PROFILE=nvidia_a6000_retail_v1 make up
MODEL_PROFILE=nvidia_dgx_retail_high_throughput make up
MODEL_PROFILE=nvidia_cosmos_research make up
```

The API should expose:

```http
GET  /api/model-registry
GET  /api/model-registry/active
POST /api/model-registry/activate/{profile_name}
GET  /api/model-metrics
GET  /api/model-runs
```

---

## A3. NVIDIA Model Catalog

Create a versioned model catalog file:

```text
configs/model_catalog/nvidia_models.yaml
```

Example:

```yaml
catalog_version: 1

models:
  - id: nvidia/nemotron-nano-v2-vl
    family: nemotron
    modality:
      - image
      - video
      - text
    tasks:
      - clip_reasoning
      - event_verification
      - visual_question_answering
      - timestamped_observation
    precision:
      - bf16
      - fp8
      - fp4
    runtimes:
      - vllm
      - triton
      - tensorrt_llm
      - nim
    default_metrics:
      - clips_per_min_gpu
      - video_sec_per_gpu_sec
      - ttft_ms
      - output_tokens_per_sec
      - parse_success_rate
      - event_verification_accuracy

  - id: nvidia/nemotron-omni
    family: nemotron
    modality:
      - video
      - audio
      - image
      - text
    tasks:
      - multimodal_reasoning
      - alert_verification
      - incident_explanation
    runtimes:
      - nim
      - vllm
      - tensorrt_llm
    default_metrics:
      - multimodal_latency_ms
      - tokens_per_sec
      - gpu_memory_gb
      - incident_accuracy

  - id: nvidia/cosmos-3
    family: cosmos
    modality:
      - video
      - world_state
      - text
    tasks:
      - digital_twin_simulation
      - synthetic_scenario_generation
      - physical_reasoning
      - action_prediction
    runtimes:
      - nvidia_runtime
      - triton
    default_metrics:
      - scenario_generation_latency
      - synthetic_usefulness_score
      - physical_reasoning_accuracy

  - id: nvidia/radio-clip-or-compatible
    family: embedding
    modality:
      - image
      - video_frame
    tasks:
      - visual_embedding
      - object_embedding
      - search
      - reidentification_support
    runtimes:
      - triton
      - custom_fastapi
    default_metrics:
      - embeddings_per_sec_gpu
      - recall_at_5
      - recall_at_10
      - vector_latency_ms

  - id: nvidia/deepstream-detector-tao-compatible
    family: cv
    modality:
      - video
      - image
    tasks:
      - detection
      - person_detection
      - object_detection
    runtimes:
      - deepstream_tensorrt
    default_metrics:
      - fps_gpu
      - streams_gpu
      - detection_latency_ms
      - gpu_decoder_utilization

  - id: nvidia/deepstream-reid-compatible
    family: cv
    modality:
      - image
      - cropped_track
    tasks:
      - reidentification
      - multi_camera_association
    runtimes:
      - deepstream_tensorrt
      - triton
    default_metrics:
      - reid_embeddings_per_sec
      - id_switches
      - cross_camera_match_accuracy
```

---

## A4. VSS-Inspired Capabilities Built in OSS

### A4.1 Capability map

| VSS-style capability | OSS implementation in this platform | NVIDIA model/runtime |
|---|---|---|
| Video IO and Storage | FFmpeg/GStreamer + MinIO + Postgres | None required |
| Real-time CV | DeepStream worker | DeepStream + TensorRT NVIDIA models |
| Real-time Embeddings | Embedding worker | NVIDIA embedding model via Triton/NIM/custom |
| Long Video Summarization | Clip/event aggregator + summarizer worker | Nemotron/Nemotron Omni |
| Video Search | pgvector + hybrid metadata search | NVIDIA embeddings |
| Attribute Search | Event schema + detection attributes + indexed JSONB | NVIDIA detector + embedding |
| Multi-Embedding Fusion | pgvector tables for visual/text/object/event embeddings | NVIDIA embedding models |
| Search Critic Agent | Search critic worker | Nemotron reasoning model |
| Alert Verification | VLM worker | Nemotron VL / Nemotron Omni |
| Physical Reasoning | Digital twin + scenario reasoner | Cosmos |
| Synthetic Scenario Generation | Digital twin simulator | Cosmos |

### A4.2 OSS long video summarization design

Do not feed a full 1-hour video directly into one model.

Use this staged flow:

```text
10-sec clips
  → 5-sec AI windows
  → detector/tracker events
  → suspicious/important moments
  → per-clip captions
  → per-5-minute summary
  → per-hour summary
  → final incident-aware report
```

Summarization worker input:

```json
{
  "store_id": "kathirmani_01",
  "time_range": {
    "start": "2026-06-01T10:00:00+05:30",
    "end": "2026-06-01T11:00:00+05:30"
  },
  "cameras": ["left_aisle", "billing", "entry_exit"],
  "summary_mode": "loss_prevention",
  "include_events": true,
  "include_low_confidence": false
}
```

Summarization output:

```json
{
  "summary": "The store had moderate activity. One suspicious shelf interaction was detected at 10:32 near shelf zone A.",
  "timeline": [
    {
      "time": "10:32:04",
      "camera": "left_aisle",
      "event": "person handled item near shelf zone A",
      "evidence_segment_id": "uuid"
    }
  ],
  "open_questions": [
    "Billing camera should be checked for the same person between 10:32 and 10:34."
  ],
  "recommended_review_items": [
    "incident_uuid_1"
  ]
}
```

### A4.3 OSS search design

Implement search like this:

```text
query
  → query parser
  → metadata filter
  → text embedding
  → visual/event embedding search
  → multi-embedding fusion
  → critic reranker
  → timeline results
```

Example query:

```text
"show people picking up an item and walking away without billing"
```

Search pipeline:

```text
1. Extract filters:
   object=item, action=pick, route=exit, missing=billing

2. Search indexed event metadata:
   event_type IN suspicious_item_interaction, possible_billing_bypass

3. Search embeddings:
   clip captions, VLM observations, object crops, event text

4. Fuse results:
   visual similarity + event metadata + time proximity + camera path

5. Critic:
   Nemotron reasoning model reranks and explains why each result matched.
```

---

## A5. Config-Driven Model Selection

### A5.1 models.yaml

Replace the earlier model section with this stricter NVIDIA-only structure.

```yaml
model_policy:
  vendor_scope: nvidia_only
  allow_non_nvidia_models: false
  allow_vss_runtime_dependency: false
  vss_allowed_as_reference: true

active_profile: nvidia_a6000_retail_balanced

profiles:
  nvidia_a6000_retail_balanced:
    description: "Balanced profile for A6000-class retail video analytics."

    detection:
      enabled: true
      task: detection
      model_id: nvidia/deepstream-detector-tao-compatible
      plugin: deepstream_detector
      runtime: deepstream_tensorrt
      precision: fp16
      batch_size: 8
      input_size: [640, 640]

    tracking:
      enabled: true
      task: tracking
      plugin: deepstream_tracker
      runtime: deepstream
      tracker_type: nvdcf
      reid_model_id: nvidia/deepstream-reid-compatible

    clip_reasoning:
      enabled: true
      task: vlm_clip_reasoning
      model_id: nvidia/nemotron-nano-v2-vl
      plugin: nvidia_openai_compatible_vlm
      runtime: vllm
      endpoint: http://nemotron-vl:8000/v1
      prompt_pack: retail_loss_v1
      max_frames: 16
      frame_sampling: event_aware
      max_output_tokens: 512
      temperature: 0.1

    embeddings:
      enabled: true
      task: embedding
      model_id: nvidia/radio-clip-or-compatible
      plugin: nvidia_embedding
      runtime: triton
      endpoint: http://embedding-triton:8000
      vector_dim: 768

    summarization:
      enabled: true
      task: long_video_summarization
      model_id: nvidia/nemotron-reasoning-or-omni
      plugin: nvidia_summary
      runtime: vllm
      endpoint: http://nemotron-summary:8000/v1
      summary_chunk_min: 5

    search_critic:
      enabled: true
      task: search_critic
      model_id: nvidia/nemotron-reasoning
      plugin: nvidia_text_reasoner
      runtime: vllm
      endpoint: http://nemotron-reasoner:8000/v1

    digital_twin_simulation:
      enabled: false
      task: physical_reasoning
      model_id: nvidia/cosmos-3
      plugin: nvidia_cosmos
      runtime: nvidia_runtime
      endpoint: http://cosmos-worker:8000

  nvidia_dgx_retail_high_throughput:
    description: "High throughput profile for DGX-class benchmarking."

    detection:
      enabled: true
      task: detection
      model_id: nvidia/deepstream-detector-tao-compatible
      plugin: deepstream_detector
      runtime: deepstream_tensorrt
      precision: fp16
      batch_size: 32

    tracking:
      enabled: true
      plugin: deepstream_tracker
      runtime: deepstream
      tracker_type: nvdcf

    clip_reasoning:
      enabled: true
      model_id: nvidia/nemotron-nano-v2-vl
      plugin: nvidia_openai_compatible_vlm
      runtime: tensorrt_llm_or_nim
      endpoint: http://nemotron-vl-nim:8000/v1
      max_frames: 8
      max_output_tokens: 256
      concurrency: 16

    embeddings:
      enabled: true
      model_id: nvidia/radio-clip-or-compatible
      plugin: nvidia_embedding
      runtime: triton
      concurrency: 32
```

### A5.2 Config validation

Add a validator:

```bash
make validate-model-config
```

Validation rules:

```text
1. vendor_scope must be nvidia_only.
2. No model_id may start with non-NVIDIA provider names.
3. Each task must map to a known plugin.
4. Each plugin must declare supported runtimes.
5. Each runtime must have an endpoint or local launch command.
6. Each output schema must be declared.
7. Metrics must be enabled for every active model.
```

---

## A6. Model Run Tracking Schema

Add these tables.

### A6.1 model_profiles

```sql
CREATE TABLE model_profiles (
  id UUID PRIMARY KEY,
  profile_name TEXT UNIQUE NOT NULL,
  description TEXT,
  config_json JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### A6.2 model_registry

```sql
CREATE TABLE model_registry (
  id UUID PRIMARY KEY,
  model_id TEXT NOT NULL,
  model_family TEXT NOT NULL,
  vendor TEXT NOT NULL DEFAULT 'nvidia',
  modality TEXT[] NOT NULL,
  supported_tasks TEXT[] NOT NULL,
  supported_runtimes TEXT[] NOT NULL,
  precision TEXT[],
  license_notes TEXT,
  source_url TEXT,
  metadata_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### A6.3 model_runs

```sql
CREATE TABLE model_runs (
  id UUID PRIMARY KEY,
  model_profile_name TEXT NOT NULL,
  model_id TEXT NOT NULL,
  task TEXT NOT NULL,
  runtime TEXT NOT NULL,
  gpu_name TEXT,
  gpu_count INT,
  precision TEXT,
  batch_size INT,
  input_entity_type TEXT,
  input_entity_id UUID,
  output_entity_type TEXT,
  output_entity_id UUID,
  latency_ms NUMERIC,
  ttft_ms NUMERIC,
  input_tokens INT,
  output_tokens INT,
  tokens_per_sec NUMERIC,
  frames_processed INT,
  video_sec_processed NUMERIC,
  clips_processed INT,
  gpu_memory_used_mb NUMERIC,
  gpu_utilization_avg NUMERIC,
  parse_success BOOLEAN,
  quality_score NUMERIC,
  error TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

### A6.4 model_benchmark_runs

```sql
CREATE TABLE model_benchmark_runs (
  id UUID PRIMARY KEY,
  benchmark_name TEXT NOT NULL,
  model_profile_name TEXT NOT NULL,
  gpu_name TEXT,
  gpu_count INT,
  dataset_name TEXT,
  total_clips INT,
  total_video_sec NUMERIC,
  total_runtime_sec NUMERIC,
  clips_per_min_gpu NUMERIC,
  video_sec_per_gpu_sec NUMERIC,
  avg_latency_ms NUMERIC,
  p95_latency_ms NUMERIC,
  avg_ttft_ms NUMERIC,
  avg_tokens_per_sec NUMERIC,
  parse_success_rate NUMERIC,
  event_precision NUMERIC,
  event_recall NUMERIC,
  false_positives_per_hour NUMERIC,
  estimated_cost_per_1000_clips NUMERIC,
  metadata_json JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## A7. Grafana Dashboards for Model Comparison

Add a dedicated model benchmarking folder in Grafana:

```text
observability/grafana/dashboards/model-benchmarks/
```

### Dashboard 11: Model Registry and Active Profile

Panels:

```text
Active model profile
Active model by task
Runtime by task
Model health status
Model endpoint status
Model config validation status
Loaded model memory footprint
```

Variables:

```text
store_id
model_profile
task
runtime
gpu_name
model_id
```

### Dashboard 12: Model Throughput

Panels:

```text
clips/min/GPU by model
video-sec/sec/GPU by model
frames/sec/GPU by detector
embeddings/sec/GPU
tokens/sec by VLM
requests/sec by runtime
concurrency by runtime
```

Prometheus metric names:

```text
model_clips_processed_total
model_video_seconds_processed_total
model_frames_processed_total
model_embedding_requests_total
model_output_tokens_total
model_requests_total
```

### Dashboard 13: Model Latency

Panels:

```text
p50 latency by model/task
p95 latency by model/task
p99 latency by model/task
TTFT by VLM
queue wait time before model
model execution time
post-processing time
JSON parse/repair time
```

Metric names:

```text
model_latency_ms_bucket
model_ttft_ms_bucket
model_queue_wait_ms_bucket
model_postprocess_ms_bucket
model_json_parse_ms_bucket
```

### Dashboard 14: Model Quality

Panels:

```text
event precision by model profile
event recall by model profile
false positives/hour
false negatives/hour
VLM parse success rate
human reviewer agreement
verdict distribution
model confidence calibration
```

Metric names:

```text
model_event_precision
model_event_recall
model_false_positives_total
model_false_negatives_total
model_parse_success_total
model_human_agreement_total
model_verdict_total
model_confidence_bucket
```

### Dashboard 15: Model TCO

Panels:

```text
cost per 1,000 clips
cost per video hour
cost per camera per day
cost per store per month
GPU power draw per model
GPU memory efficiency
GPU utilization efficiency
idle GPU waste
```

Metric names:

```text
model_estimated_cost_per_1000_clips
model_estimated_cost_per_video_hour
model_gpu_power_watts
model_gpu_memory_used_bytes
model_gpu_utilization_ratio
```

### Dashboard 16: VSS-Inspired OSS Capability Parity

Track how close the OSS platform is to VSS-style capabilities.

Panels:

```text
RT-CV parity:
  detector/tracker throughput
  event schema coverage

RT-Embedding parity:
  embedding throughput
  search recall@k

LVS parity:
  summary latency
  timestamp accuracy
  human summary agreement

Search parity:
  query latency
  result relevance
  critic agreement

VIOS parity:
  clip recording success
  playback success
  storage latency
```

This dashboard should not call VSS in production. It tracks our OSS implementation maturity.

### Dashboard 17: Model Failure and Drift

Panels:

```text
model errors by type
timeout rate
OOM rate
bad JSON rate
low-confidence events
camera-specific failure distribution
zone-specific false positives
model drift over time
golden dataset regression score
```

### Dashboard 18: NVIDIA GPU Runtime Comparison

Compare the same NVIDIA model across runtimes.

Panels:

```text
Ollama vs llama.cpp vs vLLM vs Triton vs NIM vs TensorRT-LLM
tokens/sec
TTFT
GPU memory
concurrency
p95 latency
error rate
cost per 1,000 requests
```

---

## A8. Prometheus Metrics to Add

Every model worker must expose `/metrics`.

### A8.1 Common model metrics

```text
model_requests_total{model_id,task,runtime,profile,status}
model_latency_ms_bucket{model_id,task,runtime,profile}
model_errors_total{model_id,task,runtime,profile,error_type}
model_gpu_memory_used_bytes{model_id,task,runtime,profile,gpu}
model_gpu_utilization_ratio{model_id,task,runtime,profile,gpu}
```

### A8.2 VLM metrics

```text
model_ttft_ms_bucket{model_id,runtime,profile}
model_input_tokens_total{model_id,runtime,profile}
model_output_tokens_total{model_id,runtime,profile}
model_tokens_per_second{model_id,runtime,profile}
model_json_parse_success_total{model_id,runtime,profile}
model_json_parse_failure_total{model_id,runtime,profile}
model_verdict_total{model_id,verdict,profile}
```

### A8.3 Video metrics

```text
model_clips_processed_total{model_id,task,runtime,profile}
model_video_seconds_processed_total{model_id,task,runtime,profile}
model_frames_processed_total{model_id,task,runtime,profile}
model_fps{model_id,task,runtime,profile}
model_clips_per_min_gpu{model_id,task,runtime,profile,gpu}
model_video_sec_per_gpu_sec{model_id,task,runtime,profile,gpu}
```

### A8.4 Quality metrics

```text
model_event_precision{model_id,task,profile,dataset}
model_event_recall{model_id,task,profile,dataset}
model_false_positives_total{model_id,task,profile}
model_false_negatives_total{model_id,task,profile}
model_human_agreement_total{model_id,task,profile,decision}
model_golden_regression_score{model_id,task,profile,dataset}
```

---

## A9. Benchmark Matrix: NVIDIA Models Only

### A9.1 Detector benchmark

```yaml
benchmark:
  name: detector_retail_cv
  task: detection
  models:
    - nvidia/deepstream-detector-tao-compatible
  runtimes:
    - deepstream_tensorrt
  gpus:
    - RTX_A6000
    - DGX_PROFILE
  metrics:
    - fps_gpu
    - streams_gpu
    - p95_latency_ms
    - gpu_memory_gb
    - detection_precision
    - detection_recall
```

### A9.2 VLM benchmark

```yaml
benchmark:
  name: vlm_retail_loss_verification
  task: clip_reasoning
  models:
    - nvidia/nemotron-nano-v2-vl
    - nvidia/nemotron-omni
  runtimes:
    - ollama
    - llama_cpp
    - vllm
    - tensorrt_llm
    - nim
  datasets:
    - golden_retail_loss_v1
  metrics:
    - clips_per_min_gpu
    - video_sec_per_gpu_sec
    - ttft_ms
    - tokens_per_sec
    - p95_latency_ms
    - parse_success_rate
    - event_precision
    - event_recall
    - false_positives_per_hour
```

### A9.3 Embedding/search benchmark

```yaml
benchmark:
  name: nvidia_embedding_search
  task: video_search
  models:
    - nvidia/radio-clip-or-compatible
    - nvidia/nemotron-retrieval-compatible
  runtimes:
    - triton
    - nim
    - custom_fastapi
  metrics:
    - embeddings_per_sec_gpu
    - recall_at_5
    - recall_at_10
    - query_latency_ms
    - fusion_latency_ms
    - critic_latency_ms
```

### A9.4 Digital twin/Cosmos benchmark

```yaml
benchmark:
  name: cosmos_digital_twin_research
  task: physical_reasoning
  models:
    - nvidia/cosmos-3
  metrics:
    - scenario_generation_latency
    - scenario_diversity
    - synthetic_to_real_improvement
    - rare_case_coverage
    - human_usefulness_score
```

---

## A10. Model Plugin Folder Structure

Update repo structure:

```text
model_plugins/
├── README.md
├── base/
│   ├── plugin.py
│   ├── schemas.py
│   └── metrics.py
├── deepstream_detector/
│   ├── plugin.py
│   ├── config_schema.yaml
│   └── tests/
├── deepstream_tracker/
│   ├── plugin.py
│   ├── config_schema.yaml
│   └── tests/
├── nvidia_openai_compatible_vlm/
│   ├── plugin.py
│   ├── prompt_runner.py
│   ├── output_parser.py
│   └── tests/
├── nvidia_embedding/
│   ├── plugin.py
│   ├── fusion.py
│   └── tests/
├── nvidia_summary/
│   ├── plugin.py
│   ├── hierarchical_summary.py
│   └── tests/
├── nvidia_search_critic/
│   ├── plugin.py
│   ├── reranker.py
│   └── tests/
└── nvidia_cosmos/
    ├── plugin.py
    ├── scenario_generator.py
    └── tests/
```

---

## A11. Plugin Tests

Each plugin must pass:

```text
1. config schema validation
2. health endpoint test
3. fake inference test
4. real sample inference test
5. metrics emission test
6. timeout handling test
7. GPU OOM handling test
8. bad output handling test
9. golden dataset regression test
10. benchmark harness compatibility test
```

Example:

```bash
make test-plugin PLUGIN=nvidia_openai_compatible_vlm
make bench-plugin PLUGIN=nvidia_openai_compatible_vlm MODEL=nvidia/nemotron-nano-v2-vl
```

---

## A12. Updated Phase Plan

### Add to Phase 0

```text
Create model plugin interface.
Create NVIDIA-only model policy.
Create model catalog.
Create config validator.
Create model metrics naming convention.
```

### Add to Phase 3

```text
Add model registry dashboard.
Add model throughput dashboard.
Add model latency dashboard.
Add model quality dashboard.
Add model TCO dashboard.
```

### Add to Phase 6

```text
Implement VLM worker as plugin host, not hardcoded model worker.
Support NVIDIA model profiles from config.
Track every model run in model_runs table.
```

### Add to Phase 8

```text
Build VSS-inspired OSS search:
  metadata search
  embedding search
  multi-embedding fusion
  Nemotron critic reranking
```

### Replace Phase 9

Old:

```text
VSS 3.1 benchmark path as possible runtime.
```

New:

```text
VSS-inspired OSS parity benchmark:
  read VSS capability patterns
  implement equivalent OSS pipeline
  use only NVIDIA models and our own platform services
  track parity in Grafana
```

### Add Phase 13

```text
Phase 13: NVIDIA model bake-off and production profile selection

Goal:
  Select best NVIDIA model/runtime profile for A6000 and DGX.

Deliverables:
  model benchmark reports
  Grafana model comparison dashboards
  production model profile
  fallback model profile
  model rollback mechanism

Done when:
  A config-only change can switch between NVIDIA model profiles
  and Grafana shows throughput, quality, latency, and TCO differences.
```

---

## A13. Updated Final Architecture Statement

The final architecture should now be described as:

```text
An OSS-ingestion-first, NVIDIA-model-first, plugin-driven video intelligence platform
that learns from VSS capabilities but implements its own OSS-controlled video search,
summarization, embedding, alert verification, and digital_twin workflows.
```

The corrected direction is:

```text
Use VSS as a reference architecture.
Use OSS for orchestration and storage.
Use DeepStream for accelerated CV where useful.
Use NVIDIA open models for VLM, embeddings, reasoning, summarization, and digital twin research.
Use config-driven model plugins so models/runtimes can be changed without refactoring.
Use Grafana to compare model quality, latency, throughput, GPU efficiency, and TCO.
```

