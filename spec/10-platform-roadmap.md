# 10 — Platform Roadmap (OSS ingestion + NVIDIA model plugins)

> **Forward-looking.** Docs `01`–`09` describe the **current system** (the offline
> Marlin/Qwen cascade). This doc and [11-model-plugin-policy.md](11-model-plugin-policy.md)
> describe the **target platform** the repo is heading toward, per the master plan
> [`00-master-plan-v2.md`](00-master-plan-v2.md)
> (full 3.5k-line detail). This is the **execution plan** an agent
> follows; the master plan is the *why*, this is the ordered *what-next*.

## The shift in one line

From an **offline research cascade** (one process reads 5 `.mkv` → `results/*.json`)
to an **OSS-ingestion-first, NVIDIA-model-first, plugin-driven video intelligence
platform**: durable 10-sec clips → 5-sec AI windows → queue → plugin-host AI workers
→ deterministic rule engine → VLM verification → evidence package → human review →
search → digital twin → Grafana. The current `src/kathirmani` pipeline becomes the
**AI-worker seed**; nothing is thrown away.

## What exists today vs. what the platform needs

| Layer | Today (`src/kathirmani`) | Target | Reuse |
|-------|----------------------|--------|-------|
| Ingestion | none — `cli/run_inference.py` reads `.mkv` directly | FFmpeg/GStreamer service: 10-sec clips, 5-sec windows, RTSP reconnect, camera health | PyAV preload (`ffmpeg_preload.py`) |
| Object store | raw `.mkv` + `results/*.json` on disk | MinIO / NVMe abstraction, evidence clips | `RESULTS_DIR` anchor |
| Queue | none — synchronous in-process | Redis Streams (`video_segment.ready`, `ai_window.ready`) | — |
| Metadata DB | **none, by design** (file-first) | Postgres + pgvector (master plan §6 + A6) | JSON→rows backfill adapter |
| CV worker | `kathirmani.locate` (YOLOE, optional stub) | DeepStream+TensorRT **and** OSS YOLOE worker, one plugin contract | `pipeline.py` locate/route logic |
| Rule engine | none | deterministic zone/dwell/interaction/bypass/health rules + incidents | — |
| VLM worker | `qwen_vl.py`, hardcoded | **plugin host**, NVIDIA Nemotron-VL default, every run tracked | `qwen_vl.py` → baseline plugin |
| Evidence + review | none (Streamlit = viz only) | pre/during/post stitching, timeline, audited verdicts | — |
| Search | none | pgvector + multi-embedding fusion + Nemotron critic | — |
| Digital twin | implicit (5 cameras, same area) | YAML store/camera/FOV/zone, event→twin mapping | camera roles |
| Plugin layer | none — code is imported Python | `ModelPlugin` ABC + registry + profiles | `metrics.py` registry pattern |
| Observability | 8 dashboards, `kathirmani_*` | +dashboards 01–18, `model_*` namespace, OTel, DCGM | the whole `04`/`08` stack |
| Makefile / compose | none — `start_stack.sh` only | Makefile + 3 compose files, one-command stack | `start_stack.sh` logic |

Strongest existing asset = observability ([04](04-observability-stack.md),
[08-dashboards.md](08-dashboards.md)); extend it, don't replace it. The cascade-
gating thesis ([05-performance-and-optimizations.md](05-performance-and-optimizations.md))
survives: detector + rules gate the VLM — never run the VLM on every window.

## The one hard reconciliation — model policy

The master plan (Addendum A) mandates **NVIDIA-only default models** and explicitly
disallows Qwen/LLaVA/InternVL as defaults. The current cascade (Marlin = Qwen3-VL
based, Qwen2.5-VL, YOLOE) is **all disallowed-as-default**. Resolution: keep it as a
runnable, comparison-only `research_qwen_baseline` profile behind the plugin layer;
NVIDIA models (possibly stubbed until weights land) are the production/benchmark
default. Because everything sits behind the plugin contract, swapping NVIDIA weights
in is a config flip, not a refactor. Full detail in
[11-model-plugin-policy.md](11-model-plugin-policy.md).

## Data stack — Postgres + filesystem (decision, 2026-06-03)

The master plan named Postgres + Redis + MinIO. For a single-box pilot we **collapse
to Postgres + the local filesystem**, keeping Redis/MinIO as optional backends behind
the ingestion abstractions (so nothing is wasted and scale can re-enable them):

- **Postgres (+pgvector) holds ALL structured state** — metadata, events, vectors,
  `model_runs`, **and the job queue** via `SELECT … FOR UPDATE SKIP LOCKED`. At
  30-camera scale the message rate is modest, so a Postgres queue is ample; it adds
  transactional enqueue+state (no dual-write), one fewer service, and an auditable
  queue-as-table (good for the trust posture). **Redis** stays an *optional*
  high-fanout backend (`RedisStreamQueue`) — revisit only at Kafka-scale fan-out.
- **Clip blobs live on local NVMe**, not in Postgres. Storing video bytes in Postgres
  (BYTEA/large objects) bloats the DB + WAL, wrecks backup/restore, and can't
  range-serve video to the review UI. Postgres stores **paths + checksums**; the
  filesystem stores the bytes (`LocalFSStorage`). **MinIO** stays an *optional* backend
  for when S3 API / multi-node / lifecycle tiering is needed.
- Net: the pilot runs on **one service (Postgres) + files** — smaller ops + security
  surface, and it matches the repo's file-first ethos (spec/02). The ingestion backends
  are pluggable (`ingestion/{storage,bus,metadata}.py`), so this is config, not a rewrite.

## Temporal correlation — why pg-cataloged spans (2026-06-05)

Ingestion is not a fire-and-forget pipe; it's the **durable, time-indexed substrate**
for multi-scale, *connected* reasoning. Subtle retail-loss actions have **variable
extent** (palming ≈3s · tag-swap 5–15s · sweethearting/basket-to-bag 10–40s · return
fraud = minutes) and each micro-action looks innocent — only the **sequence**, often
**across clip + camera boundaries**, is suspicious. A fire-and-forget design (decode →
infer once → discard) is blind to anything longer than its window, can't take a second
look, and has no shared timeline to link views. So the same stream must be re-windowable
at any scale, replayable byte-for-byte, and correlatable — which is exactly what the
Postgres catalog provides: the hierarchy `video_segments` (abs span + sha256) →
`ai_windows` (offsets) → `events` (partitioned by `event_time`) → **`incidents`
(arbitrary span)** linked via **`incident_events` (M:N)** and `tracks`
(`first_seen`/`last_seen`); `embeddings vector(768)` for "similar moments"; `job_queue`
(SKIP LOCKED) to **replay** a span on a new model; `sha256` + append-only `audit_log`
for tamper-evidence. Figures: `design/figures/ingest_multiscale_time.png`,
`ingest_coverage_quadrant.png`, `ingest_reasoning.png`, `data_ingestion_layer.png`.

**Resolved (migration 0004, 2026-06-10):** `video_segments`/`incidents` got generated
absolute `tstzrange span` columns and `ai_windows` a trigger-maintained `span` (derived
from the parent segment + offsets); composite **`btree_gist` GiST overlap indexes**
`(camera_id|store_id, span)` make cross-clip / cross-camera "everything overlapping
[t0,t1]" a first-class index-served join. Migration applied + verified against real
Postgres on 2026-06-10. (`events` stays point-in-time; its btree suffices.)

## Phased build (master plan §15 + Addendum A12)

Each phase = one branch = one PR, ends green (its test gate passes, observability
shows the new signal, a `agents-memory/sessions/` entry is written). **Contracts
before code** — write each layer's schema/interface + validator before its
implementation.

| Phase | Goal | Done-when |
|-------|------|-----------|
| **0 ✅** Skeleton | Makefile, 3 compose files (base/gpu/observability — wrap the existing stack), `configs/` (cameras/stores/zones/models/rules/retention), `ModelPlugin` interface, NVIDIA model policy + `validate-model-config` | `make up` green; policy validator passes NVIDIA / fails non-NVIDIA default |
| **1 ✅** OSS ingestion | PyAV → 10-sec clips + 5-sec windows @2s stride → filesystem + JSONL + queue; camera health. **Files done; live RTSP (GStreamer `splitmuxsink`, Phase 1.5) now implemented** (`GStreamerSource` + `catalog_live_clips`). | 5-camera footage segmented, registered, queued; `ingest_*` metrics |
| **2 ✅** DB + API | `db/migrations` + `schema.sql` (§6 + A6, 19 tables, partitioned events, job_queue); `PgQueue` (SKIP LOCKED); JSON→rows backfill; FastAPI (+ model-registry endpoints); seed kathirmani | workers/UI use stable API contracts; `make migrate && seed && backfill` idempotent |
| **3 ✅** Observability | dashboards 01–18 (`observability/grafana/`, generator + 18 JSONs), `model_*`/`ingest_*` scrape, OTel + promtail configs, additive to `kathirmani_*` | dashboards generated + validated; scrape wiring documented (panels light up as producers expose /metrics) |
| **4 ✅** CV worker | `ai_workers/cv_oss_worker` — YOLOE behind `DetectionPlugin` (fake-infer fallback), consumes `ai_window.ready`, writes detections/events/model_runs, zone-maps, emits `suspicious_item_interaction`. DeepStream worker later. | CV worker emits common event schema; 23 tests green |
| **5 ✅** Rule engine | `services/rule_engine/` — deterministic per-track context windows over `event_rules.yaml`; consumes/republishes `event.created`, raises explainable events + `possible_loss` incidents; golden fixtures | suspicious-item events raised **without** a VLM; 24 tests green |
| **6 ✅** VLM worker | `ai_workers/vlm_worker/` — plugin host (Nemotron-VL / Qwen baseline + fake_infer); prompt pack `retail_loss_v1`, JSON repair, writes `vlm_observations` + `model_runs`, updates event confidence | suspicious events get explanation; profile-selected; 23 tests green |
| **7 ✅** Evidence + review | `services/evidence_builder/` (PyAV 20-sec stitch + sha256 + timeline, manifest-only fallback) + `services/review_ui/` (:8010 approve/reject, audit reviewer/ts/model_version) | reviewable incidents; 24 tests green |
| **8 ✅** Search | `ai_workers/embedding_worker/` — C-RADIOv4-H embedding plugin (fake 768-d) + indexer + parse→metadata→pgvector→fuse→critic search | NL search returns ranked, time-stamped hits; 29 tests green (ran vs live pgvector). C-RADIO dim≠768 **resolved**: cosine-preserving random-projection head (`plugin._project`) maps to the `vector(768)` column, `vector_dim: 768` |
| **9 ✅** VSS-parity | `ai_workers/vss_eval_worker/` — `nvidia_summary` plugin + §A4.2 staged LVS (clip→5min→hour→report) + `parity.py`→`parity_report.json` (rt_cv/embedding/lvs/search/vios). VSS reference-only | parity report (honest measured/fake/pending); 24 tests green |
| **10 ✅** Digital twin | `services/digital_twin/` — `StoreTwin` (load/validate/zones_for_point/map_event/summary); `configs/digital_twin/second_store.yaml` proves YAML-only onboarding | second store loads + validates with the same code; 5 tests green |
| **11 ✅** Benchmark + TCO | `benchmarks/` harness (p50/p95, clips-min/GPU, tokens/sec) + `tco.py` (cost/1000-clips, /camera-month, /store-month) + 6 configs → `model_benchmark_runs`. Fake-mode default (synthetic until GPU run); real runner pluggable | harness + TCO; 28 tests green |
| **12 ✅** Hardening | `services/security/` — auth (PLATFORM_API_KEY) + RBAC (viewer/reviewer/admin) wired into the API, audit log (`audit_log` table, migration 0002), secret redaction, retention CLI (evidence-lock, dry-run default), backup | building blocks + API auth/RBAC; 38 tests green |
| **13 ✅** NVIDIA bake-off | `benchmarks/bakeoff/` — runtime matrix reusing the Phase-11 harness; `select_profiles()` → production+fallback by weighted score; `rollback.py` reuses the API's `model_profiles.active` switch; rows → `model_benchmark_runs` (dashboard 18) | config-only profile swap + rollback; 20 tests. Numbers synthetic until real runtimes on GPU |

## Entry point — the Console (2026-06-10)

All 14 phases shipped many surfaces (api, review_ui, Grafana, search) but **no single
front door**. Added `services/console/` — a BFF/gateway on **:8080** that serves one
unified SPA and proxies the upstreams server-side — plus `Dockerfile.app` +
`docker-compose.platform.yml` + **`make platform`** for a one-command launch of the
whole stack behind that URL. Design + ports + env: [14-console-and-deployment.md](14-console-and-deployment.md).

## What's to be done (roadmap)

The 14 phases + the code-doable real-hardware follow-ups + the entry-point Console are
**done**. The full suite is green — **298 passed, 1 skipped, 1 deselected** in one flat
`make test` run (verified 2026-07-08 post-refactor; the one remaining failure,
`test_venv_packages`, is this box's torchcodec↔FFmpeg provisioning, not platform code).
What remains:

**A. Blocked on infra / user action (cannot be coded here):**
- `make setup-nvidia-docker` — needs **sudo** (not passwordless); registers the NVIDIA
  Docker runtime → required for GPU containers.
- **No NGC key** (`nvcr.io` 401) → DeepStream / NIM / TAO still blocked; YOLOE/RT-DETR
  remains the free CV path.
- **Real-GPU numbers** — Phase 11/13 benchmark + bake-off run in fake mode, and the VSS
  hierarchical summary (Phase 9) is still synthetic; both need real weights on the GPU.
  (The Nemotron-VL chat path itself is real + verified on a clip.)
- **Apply migrations on each deployment's Postgres** — `make migrate` (0001–0004).
  Verified applying cleanly on 2026-06-10; just needs running wherever PG lives.

**B. Deferred refactor:**
- ✅ **Hyphen→underscore package rename — DONE 2026-07-08.** The worker/service dirs
  (`ai_workers/*`, `services/rule_engine|review_ui|evidence_builder|digital_twin`,
  `model_plugins`) are real importable packages; the ~27 `sys.path` shims, the dual
  package/flat-import `try/except` blocks, and all per-dir `conftest.py`/`pytest.ini`
  hacks are gone. `make test` is one flat pytest run (see spec/09 → Tests). Makefile
  targets run `python -m <package>` / dotted uvicorn app paths from the repo root;
  compose services run from `working_dir: /app`. Compose *service names* and dashboard
  *filenames* keep their hyphens (identities, not paths).
- Possible relocation of the legacy `src/kathirmani` layer — currently **invariant-protected**
  (keep root shims + path anchors + `learning.md`); only after the baseline is no longer
  needed as a live comparison arm.

**C. Open code follow-ups:**
- **Cross-camera track handoff** — `Tracker` gives per-camera persistent `track_id`s;
  linking a person across cameras (and the rule_engine consuming it) is the next step.
- **Nemotron-VL production edge cases** — very long / multi-language prompts, batching.
- **`serve_stream.py` parallel-GPU frame batching** — design-only (spec/02), no code yet.

## Critical path & parallelism

- **Critical path** = 0 → 1 → 2 → (3 ∥ 4) → 5 → 6 → 7. That is the master plan §24
  two-week demo: footage → clips → detections → suspicious event → VLM verification
  → evidence → Grafana + review UI.
- **Parallel once 0–2 land:** Phase 3 (observability) ∥ Phase 4 (CV worker); Phase 8
  (search) and Phase 10 (twin) are independent after Phase 5 events exist.
- **Acquirable now (free/OSS on HF):** the NVIDIA VLM/LLM/embedding/Cosmos models
  (Nemotron-Nano-VL, Nemotron-3-Nano, C-RADIO, Cosmos-Reason2) — the HF token works,
  so the production-default profile is real, not stubbed
  ([11-model-plugin-policy.md](11-model-plugin-policy.md)).
- **Still gated on external access:** DeepStream/NIM/TAO CV (need an **NGC key**,
  `nvcr.io` is 401) → YOLOE/RT-DETR is the free CV path meanwhile; the Docker NVIDIA
  runtime needs registering (`make setup-nvidia-docker`, needs sudo). Live RTSP capture
  (GStreamer Phase 1.5) is now implemented. Build the wiring with what's free; add NGC
  assets when a key exists. Full remaining-work list: *What's to be done* above.
- **Always-on from Phase 1:** Grafana (master plan §20.5 "do not delay observability").

## Anti-goals (master plan §20 — do not violate)

1. Not model-first — twin → events → ingest → detect/track → reason → verify.
2. Business logic is **not** inside DeepStream configs — DeepStream emits facts; the
   platform decides incidents.
3. Never skip the evidence package — without it, it's a demo.
4. Don't optimize tokens/sec alone — measure video-sec/sec/GPU, clips/min/GPU,
   precision/recall, p95, cost/store.
5. Don't delay observability — Grafana from Phase 1.
6. ~~Don't make a non-NVIDIA model the production/benchmark default — Qwen is
   comparison-only.~~ **Relaxed by v3 (2026-06-16):** model selection is governed by
   measured cost/quality parity, not vendor; NVIDIA-only is one *recommended* profile,
   not mandated ([16-capability-hooks-profiles-router.md](16-capability-hooks-profiles-router.md)
   §Policy reconciliation). Provenance pinning + the parity gate replace it as the trust
   control ([11-model-plugin-policy.md](11-model-plugin-policy.md)).

## Security & trust (cross-cutting — utmost priority, not just Phase 12)

This is a **platform**, so security and trust are first-class and built into **every
phase**, not deferred to the Phase 12 hardening bucket. It handles retail CCTV
(faces/people = **PII/biometric data**), produces loss/theft **evidence** that must
be tamper-evident and admissible, and is **multi-store** (tenant isolation). Trust in
the data, the models, and the verdicts *is* the product.

| Concern | Control | Lands in |
|---------|---------|----------|
| **Secrets** | RTSP creds / HF tokens / DB passwords never in logs or git; `.env` + secret store; `validate-config` rejects inline secrets | Phase 0 |
| **AuthN/AuthZ** | every API call authenticated; RBAC least-privilege; object-storage (MinIO) scoped perms; no anonymous footage access | Phase 2 |
| **Encryption** | TLS in transit; encryption at rest for clips (MinIO) + DB (Postgres) | Phase 1–2 |
| **PII & retention** | minimize/redact where possible; enforce `configs/retention.yaml` limits; privacy-by-design on footage | Phase 1, 12 |
| **Tenant isolation** | all data store-scoped; one store cannot read another's clips/incidents | Phase 2, 10 |
| **Evidence integrity** | `video_segments.checksum`; immutable/locked evidence; full chain-of-custody audit (reviewer + timestamp + **model version**) | Phase 7 |
| **Model trust** | pin provenance (`model_registry.source_url`/`license_notes`); **vendor scope advisory + measured cost/quality parity gate** (v3 relaxed the hard NVIDIA-only default — [16](16-capability-hooks-profiles-router.md) §Policy reconciliation); safety/guard models + output validation (A1.1) | Phase 0, 6 |
| **Auditability** | `model_runs` + incident audit logs make every decision traceable — trust through legibility | Phase 3, 6, 12 |

**Gate:** no phase is "done" until its relevant security/trust control above is in
place. Auditability + legibility (already a core principle —
[02-architecture.md](02-architecture.md), `04`/`08`) is how trust is *demonstrated*.

## Acceptance (whole platform)

Master plan §19's 13 criteria **plus** A12's profile-swap: 5 feeds ingested · 10-sec
clips durable · 5-sec windows · CV worker emits detections/tracks/events · rules
raise hypotheses · VLM verifies · evidence built · human approve/reject · Grafana
shows ingestion/queue/AI/GPU/incident metrics · custom UI shows map/timeline/replay ·
benchmark report compares GPU profiles · VSS-parity evaluated · second store
onboards code-free · **+** config-only NVIDIA model-profile swap with Grafana deltas.

## Related

[11-model-plugin-policy.md](11-model-plugin-policy.md) · [02-architecture.md](02-architecture.md) · [05-performance-and-optimizations.md](05-performance-and-optimizations.md) · [09-repo-structure.md](09-repo-structure.md)
