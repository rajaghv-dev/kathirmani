# Agent memory

Terse edit/run context. *What/why* lives in `spec/` — follow the links.

**What:** Marlin-2B + Qwen2.5-VL video intelligence over 5 store cameras →
Prometheus/Grafana/Loki + Streamlit viewer. → `spec/01`, `spec/02`.

## Repo map → `spec/09`
- Code in `src/kathirmani/`; root `*.py` are thin shims (don't add logic there).
- Entry points: `run_inference.py`, `serve_metrics.py`, `download_model.py`,
  `viz_app.py`, `start_stack.sh`. Commands unchanged post-refactor.
- Paths anchor on `kathirmani.PROJECT_ROOT/MODELS_DIR/RESULTS_DIR` (not cwd).

## Invariants — don't break
- Device is runtime-detected; load via `kathirmani.device.detect_device()`, place
  tensors with `.to(model.device)`. Never hardcode `"cuda"`/dtype. → `spec/06`.
- Keep root shims + path anchors working; keep `learning.md` as-is.
- Guard NVIDIA/Linux-only calls (nvidia-smi, NVML, `/sys`,`/proc`, `hostname -I`).

## Run / test (dev box = CPU, no model)
- `bash setup.sh` (picks torch per machine), `bash run.sh`. Override:
  `KATHIRMANI_DEVICE=cuda|mps|cpu`.
- `pytest -q tests/` → expect ~18 passed, ~7 skipped (torch/model/server absent).
  `test_device.py` + `test_structure.py` are torch-free.

## Pitfalls (detail in spec)
- "Re-downloading" = `transformers_modules` + compile cache regen, not weights
  (offline). → `spec/04`.
- Marlin is Qwen3-VL: pre-decoded frames need `video_metadata` or find spans
  break. → `spec/05`.
- One GPU; `_GPU_LOCK` serializes forward passes (only decode/IO overlap). →
  `spec/02`, `spec/05`.
- `inference/locate.py` (`--locate`) absent — optional lazy stub (`kathirmani.locate`).

## Roadmap (future platform — not built yet) → `spec/10`, `spec/11`, `spec/15`–`17`
- **v3 capability-hook layer (forward-looking, doc-only as of 2026-06-16).**
  Plug-and-play QUALITY/WHEN/WHERE/WHAT hooks + POS + cost-tiered profiles
  (economy/balanced/benchmark) + router + small-model parity. **v3 RELAXES the hard
  NVIDIA-only default** (spec/11): selection = measured cost/quality parity on retail
  KPIs, NVIDIA-only demoted to one optional/recommended profile. Not yet built — no
  `ingestion/quality/` module, no POS adapter. → `spec/15` (quality gate), `spec/16`
  (hooks/profiles/router/parity), `spec/17` (POS + wall-clock time alignment).
- Target: OSS-ingestion-first, NVIDIA-model-first, plugin platform (master plan
  `oss_ingestion_nvidia_model_plugin_master_plan_v2.md` at root). Current
  `src/kathirmani` cascade = the AI-worker seed. 14 phases; start at Phase 0.
- **Invariant — security & trust are utmost (it's a platform).** PII CCTV +
  tamper-evident loss/theft evidence + multi-store isolation. Treat security
  **cross-cutting every phase**, not just Phase 12: API auth/RBAC, secrets never
  in logs/repo, encrypt at rest+transit, evidence integrity (checksum + immutable
  + audit: reviewer/ts/model_version), model provenance. → `spec/10` Security.
- **Models = free/OSS NVIDIA, HF-hosted (verified 2026-06-03).** VLM
  `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1`; LLM `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`
  (or `-4B`); embed `nvidia/C-RADIOv4-H`; twin `nvidia/Cosmos-Reason2-2B`; CV =
  YOLOE/RT-DETR (TAO needs NGC). HF token works → acquirable now. → `spec/11` table.
- **Box/env (GB10 aarch64, CUDA13, 1.1T free).** HF token present+working; **no NGC
  key** (`nvcr.io` 401 → DeepStream/NIM/TAO blocked). Docker + nvidia-ctk installed
  but **nvidia runtime not registered** — `make setup-nvidia-docker` (needs **sudo**,
  not passwordless here → user must run it). Default generative runtime =
  **Ollama/llama.cpp (GGUF)**, not vLLM (unproven on aarch64+Blackwell); RADIO/Cosmos
  via transformers.
- **Phase 0 done (on main).** Models fetched+pinned in `models/PROVENANCE.json` (all 3:
  Nemotron-Nano-VL-8B, C-RADIOv4-H, Cosmos-Reason2-2B). Base services
  (postgres/redis/minio) come up via `docker compose -f docker-compose.yml up -d`
  (no port clash with the start_stack.sh observability stack). `make validate-model-config`
  / `config-check` / `docker-config` green. Next: Phase 1 ingestion.
- **New dashboard:** `grafana/dashboard_model_performance.json` (uid `marlin-model-perf`)
  — model performance & usefulness from the tested 5-camera results. Imported via the
  now-fixed `grafana/setup_grafana.sh` (loops all `dashboard_*.json`). → `spec/08`.
- **Invariant — NVIDIA-only default models.** Qwen/Marlin/YOLOE are disallowed as
  defaults → kept as comparison-only `research_qwen_baseline` behind the plugin
  layer; NVIDIA models (maybe stubbed) are production default. Swap = config, not
  code. → `spec/11`.

## Grafana image renderer (live infra note, 2026-06-04)
- The live `grafana` container was compose-managed by `~/Documents/raja/monitoring/`
  but that compose file is **gone** (dir root-owned, only leftovers). State is in the
  named volume `monitoring_grafanadata` on net `monitoring_default`; no provisioning
  bind mount (dashboards live in the volume).
- Added a `renderer` (grafana/grafana-image-renderer, arm64) sidecar on that net and
  **recreated `grafana`** with `GF_RENDERING_SERVER_URL=http://renderer:8081/render`
  + `GF_RENDERING_CALLBACK_URL=http://grafana:3000/` (same image/volume/port → data
  preserved). Original config saved at `/tmp/grafana_original_inspect.json`.
- Render dashboards → PNG: `bash scripts/render_dashboards.sh`. Codified in
  `docker-compose.observability.yml` (renderer service + grafana rendering env).

## Data stack — Postgres + filesystem, drop Redis/MinIO from default (decision)
- **Postgres (+pgvector) = ALL structured state incl. the job queue** (`SELECT … FOR
  UPDATE SKIP LOCKED`), replacing Redis. Modest msg rate at 30 cams → ample;
  transactional + auditable. Redis kept OPTIONAL (high-fanout only).
- **Clip blobs on local NVMe** (`LocalFSStorage`), NOT in Postgres (BYTEA bloats
  DB/WAL/backup, no range-serve). Postgres stores paths+checksums. MinIO OPTIONAL
  (S3/multi-node only). → `spec/10` Data stack; docker-compose redis/minio = profile
  "optional" (off by default). Pilot = one service (Postgres) + files.

## Ingestion media stack — PyAV now, GStreamer for live/HW (decision)
- **Files (Phase 1): PyAV** (libav) — vetted repo dep, bundled-FFmpeg preload, no
  system ffmpeg, in-process + testable. `ingestion/` segments .mkv → 10-sec clips.
- **Live RTSP: GStreamer** (`gst 1.24` on box; Python `gi` NOT installed → drive via
  gst-launch/subprocess). `rtspsrc ! rtph264depay ! h264parse ! splitmuxsink
  max-size-time=10s` = stream-copy 10-sec clips, no re-encode. The live clip writer.
- **Frame gating = mechanism vs policy.** GStreamer/DeepStream supply the *mechanism*
  to ignore frames for inference (`videorate` fps cap, keyframe-only, DeepStream
  `nvinfer interval=N`, `motioncells`); the *policy* of which frames matter (motion,
  zone activity, cascade `route_to_vlm`) stays OUR deterministic logic. = "do less
  work" + hot/cold two-tier. `ingestion/sources.py` Source ABC: File/Rtsp (PyAV) now,
  GStreamerSource is the live/Phase-4 slot.

## Front door — the Console (2026-06-10) → sessions/2026-06-10, PLATFORM.md
- **Single entry = `services/console/` (BFF gateway) on :8080.** Serves one cohesive
  SPA (`static/index.html`) + proxies server-side to API:8000 (`/backend/*`),
  review-ui:8010 (`/review/*`), embeds Grafana:3000 — one origin, no CORS, auth
  passthrough. `make console`. Was: scattered ports (8000/8010/3000/streamlit) with
  no cohesive UX. Sections: Overview/Incidents(approve-reject)/Events/Search/Segments/
  Cameras/Models/Observability; offline-degrading. `GET /search?q=` added to
  services/api (wraps embedding-worker query_run). **PLATFORM.md** = the start-here
  guide (README is still the old src/kathirmani doc).

## Real-hardware follow-ups CLEARED (2026-06-10, code-doable ones) → sessions/2026-06-10
- C-RADIO: random-projection head (`plugin._project`, cosine-preserving) replaces
  truncate/pad; vector_dim tbd→768 (kept the pgvector index). TrackingPlugin
  (`cv-oss-worker/tracking.py`) → real per-camera track_ids into events + `tracks`.
  GStreamerSource live-RTSP splitmuxsink recorder implemented. Migration **0003**
  audit_log append-only GRANTs (role `kathirmani_app`); **0004** tstzrange + btree_gist
  overlap indexes (segments/incidents generated, ai_windows trigger-maintained).
- STILL blocked on user/infra: `setup-nvidia-docker` (sudo), NGC key (DeepStream/
  NIM/TAO), real-GPU benchmark numbers + real VSS summary (weights+GPU), and applying
  0003/0004 (`make migrate`, no PG on dev box).

## Phases status (parallel-agent build)
- **ALL 14 PHASES (0–13) BUILT ✅** via 5 dependency-ordered waves of parallel agents.
  `make test` = **299 passed, 1 deselected** across 16 TESTDIRS (re-verified 2026-06-11;
  was 264). Phase 13 W5:
  `benchmarks/bakeoff/` (runtime matrix + select_profiles production/fallback +
  rollback reusing API model_profiles.active; 20 tests). `make bakeoff`.
- **Real Nemotron-VL WIRED (2026-06-04).** `ai-workers/vlm-worker/plugin.py` runs the
  real `nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1` from local weights (verified on a real
  clip, faked=False): PyAV frames → multi-image `model.chat`. Needed: `timm`/`einops`/
  `open_clip_torch` (vision tower); `attn_implementation="eager"` (no FA2); a transformers-5.9
  `all_tied_weights_keys` compat shim; `repetition_penalty=1.3`; parser repairs truncated
  JSON. C-RADIO tower *code* fetched from HF on first load (network once). Qwen baseline
  + fake_infer still intact. → `spec/13`.
- **Honest caveats / real-hardware follow-ups** (code is wired + tested with fakes;
  these need real weights/runtimes/GPU): real Nemotron-VL chat-API + summary (fake
  now); GPU-real benchmark + bake-off numbers (synthetic now); C-RADIO dim≠768 →
  projection or dim migration; DeepStream/NIM/TAO need an NGC key; multi-object
  tracking (TrackingPlugin) for true track_ids; audit_log app-role GRANTs; live-RTSP
  GStreamer splitmuxsink (Phase 1.5).
- **0–12 ✅ (13 next).** W4 (parallel): Phase 7 `services/evidence-builder/`
  (PyAV 20s stitch + sha256 + timeline) + `services/review-ui/` (:8010 audited
  approve/reject); Phase 9 `ai-workers/vss-eval-worker/` (nvidia_summary + §A4.2
  staged LVS + `parity_report.json`); Phase 11 `benchmarks/` (harness + TCO + 6
  configs → model_benchmark_runs, **fake-mode default**, real runner pluggable);
  Phase 12 `services/security/` (auth+RBAC **wired into services/api**:
  PLATFORM_API_KEY unset=dev-open via `_optional_auth`, set=enforced; admin-gate on
  /activate; audit_log = migration **0002**; retention CLI evidence-lock dry-run;
  redaction; backup). `make` targets: run-review-ui/summarize/parity/bench/
  retention-dryrun/-apply/backup/evidence-demo. Benchmarks results gitignored.
- **0✅1✅2✅3✅4✅5✅6✅8✅10✅.** W3 (parallel): Phase 6 VLM worker
  `ai-workers/vlm-worker/` (plugin host: Nemotron-VL/Qwen baseline + fake_infer,
  prompt pack, JSON repair, vlm_observations+model_runs; 22 tests; publishes
  `vlm.verified`); Phase 8 `ai-workers/embedding-worker/` (C-RADIOv4-H plugin fake
  768-d, indexer, A4.3 parse→metadata→pgvector→fuse→critic; 26 tests, ran vs live
  pgvector). FOLLOW-UPS: real Nemotron-VL chat-API shape unconfirmed; **C-RADIO native
  dim ≠ 768 → needs projection head or a dim migration** (configs vector_dim: tbd).
  Remaining: W4 (7 evidence/9 vss/11 bench/12 hardening) → W5 (13 bake-off).
- **0✅1✅2✅3✅4✅5✅10✅.** W2: Phase 5 rule engine `services/rule-engine/`
  (deterministic per-track windows over event_rules.yaml; consumes/republishes
  `event.created`; raises events + `possible_loss` incidents; golden fixtures; 22
  tests). Integration gap: CV worker emits zone *ids* + no track_id/ts hints — engine
  has fallbacks (cam:<id>); true tracking = future TrackingPlugin.
- Wave-1 (parallel agents): `observability/` (dashboard
  generator + 18 JSONs + scrape/otel/promtail), `ai-workers/cv-oss-worker/` (YOLOE→
  DetectionPlugin, fake fallback, consumes ai_window.ready), `services/digital-twin/`
  (StoreTwin + second-store demo). `make run-cv-worker`/`dashboards`/`twin-validate`.
  `make test` runs all component test dirs (TESTDIRS). ultralytics added to
  inference.txt (optional). NEXT: W2 Phase 5 rule-engine → W3 6 vlm + 8 search →
  W4 7 evidence/9 vss/11 bench/12 hardening → W5 13 bake-off.
- Hyphenated dirs (`ai-workers/cv-oss-worker`, `services/digital-twin`) import flat
  via per-dir conftest/pytest.ini (not valid package names).

## Design assets + AI image-gen (2026-06-05) → sessions/2026-06-05
- Figures live in `design/figures/` via `design/make_figures.py` (`make figures`); NV green
  `#76B900`. New: `data_ingestion_layer.png` + 3 "why pg-cataloged ingestion" figs
  (`ingest_multiscale_time/coverage_quadrant/reasoning.png`) — grounded in `db/schema.sql`.
- **AI image gen = design-time only, NOT inference.** `design/ai_images.py` wraps Google
  **Nano Banana** (`gemini-2.5-flash-image`, `google-genai`) + OpenAI **`gpt-image-1`**
  (`openai`); text→image + `--ref` editing. `requirements/design.txt`, `make setup-design`/
  `gen-image`. Keys env-only (`GEMINI_API_KEY`/`OPENAI_API_KEY`); `generated/` gitignored.
  NVIDIA-only policy (spec/11) is unaffected — it governs the inference plugins. → spec/12.
- **Temporal-correlation follow-up (spec/10):** `ai_windows` use relative sec-offsets +
  btree indexes; true overlap joins for long/cross-clip subtle actions need an absolute
  `tstzrange` + `btree_gist` index on segments/events/incidents. Not built yet.

## What's to be done (roadmap) → spec/10 "What's to be done", spec/14
- **A. Blocked on infra/user:** `make setup-nvidia-docker` (sudo) · NGC key (DeepStream/
  NIM/TAO) · real-GPU benchmark+bakeoff numbers + real VSS summary (fake now) · `make
  migrate` on each deploy's PG (0001–0004; verified applying clean 2026-06-10).
- **B. Deferred refactor (planned, NOT done — keeps suite green + invariants):**
  hyphen→underscore package rename (`cv-oss-worker`→`cv_oss_worker` …; ~27 sys.path +
  8 conftest/pytest.ini + Makefile/compose `working_dir`; ~40 files, do as 1 isolated
  PR) · legacy `src/kathirmani` relocation (invariant-protected until baseline not needed).
- **C. Open code follow-ups:** cross-camera track handoff (Tracker = per-camera ids
  only) · Nemotron-VL prod edge cases (long/multilingual, batching) · serve_stream
  parallel-GPU batching (design-only, spec/02).

## Spec index
`01` overview · `02` architecture · `03` models/queries · `04` observability ·
`05` performance · `06` hardware-portability · `07` runbook · `08` dashboards ·
`09` repo-structure · `10` platform-roadmap (+ What's-to-be-done) · `11` model-plugin-policy ·
`12` frameworks (requirements/ split + setup/validate) · `13` models (catalog/provenance) ·
`14` console + one-command deployment (the front door).

## Setup + validation framework (per component)
- Frameworks per component: `requirements/{base,inference,ingestion,db,api,models,
  observability,dev,optional}.txt`; root `requirements.txt` `-r`'s them.
- Setup: `scripts/setup/<c>.sh` + `make setup-<c>` (env/ingestion/db/models/api/
  observability/gpu). `make setup` = all Python deps; `make setup-inference` = torch.
- Validate: `scripts/validate/<c>.py` + `make validate-<c>`; `make validate`/`doctor`
  = status matrix. Rule: import-missing = FAIL, service-down = WARN. → `spec/12`.
- DB now real: `make migrate`/`migrate-down`/`seed` (scripts/db_migrate.py, db_seed.py);
  schema `db/migrations/0001_init.up.sql` + `db/schema.sql` (19 tables incl. partitioned
  events + job_queue). `db/` is a package (`db/__init__.py`) — needed so the `db`
  validator doesn't shadow it. DSN default in `db/db.py`.
- **Phase 2 complete (0✅1✅2✅).** `PgQueue` (SKIP LOCKED, `ingestion/bus.py`:
  publish/consume/ack/fail) = the platform queue (Redis optional). `make backfill`
  (scripts/backfill_ingest.py) loads ingestion JSONL → video_segments/ai_windows.
  `make api` = FastAPI `services/api/app.py` (/health,/cameras,/segments,/events,
  /incidents + A2.5 model-registry endpoints). 12 tests pass. Verified: migrate→seed→
  ingest→backfill = 3 seg+10 win in PG. Next: Phase 3 (scrape `ingest_*`/`model_*`) ∥
  Phase 4 (cv-oss-worker consuming PgQueue).
