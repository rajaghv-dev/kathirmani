# 09 · Repository Structure

The repo has two layers: the **platform** (services, AI workers, ingestion, DB, the
Console front door) and the **legacy Marlin research cascade** (`src/kathirmani/` + root
shims), kept as the non-default comparison baseline. Both share one checkout; the
legacy layer is *not* a hard dependency of the platform (only lazy/optional imports —
`ffmpeg_preload`, the Qwen baseline plugin).

## Layout

```
kathirmani/
├── PLATFORM.md             # the start-here guide (front door = the Console :8080)
├── README.md               # platform overview + legacy cascade
├── Makefile                # all entry points (make platform / api / console / test …)
├── docker-compose.yml                 # base: postgres (+ optional redis/minio)
├── docker-compose.observability.yml   # prometheus/grafana/loki/renderer/netdata/dcgm
├── docker-compose.platform.yml        # app tier: migrate→api→review-ui→console
├── docker-compose.gpu.yml             # GPU/worker overlay
├── Dockerfile.app          # torch-free image for the api/review-ui/console tier
│
├── services/               # platform services (FastAPI)
│   ├── console/            #  ☜ FRONT DOOR — BFF/gateway: SPA + proxy (:8080)  → spec/14
│   ├── api/                #  read/query API + model-registry + /search (:8000)
│   ├── review-ui/          #  human approve/reject of loss hypotheses (:8010)
│   ├── rule-engine/        #  deterministic zone/dwell/interaction rules → incidents
│   ├── evidence-builder/   #  clip stitch + sha256 + timeline (chain-of-custody)
│   ├── digital-twin/       #  YAML store/camera/zone model; event→twin mapping
│   └── security/           #  auth + RBAC + audit + redaction + retention + backup
│
├── ai-workers/             # plugin-host inference workers (consume the queue)
│   ├── cv-oss-worker/      #  YOLOE detection + Tracker (real track_ids) → events
│   ├── vlm-worker/         #  Nemotron-VL (default) / Qwen baseline / fake_infer
│   ├── embedding-worker/   #  C-RADIO embeddings (+ random-projection head) + search
│   └── vss-eval-worker/    #  hierarchical summary + VSS-parity report
├── model-plugins/base/     # the ModelPlugin ABC + common schemas (Detection, …)
│
├── ingestion/              # PyAV File/Rtsp + GStreamerSource (live RTSP, Phase 1.5)
│   │                       #   segmenter, windows, health, storage, PgQueue bus
├── db/                     # schema.sql + migrations 0001–0004 + tiny psycopg helper
│   └── migrations/         #   0001 init · 0002 audit_log · 0003 audit GRANTs · 0004 tstzrange+gist
├── configs/                # cameras/stores/zones/models/event_rules/retention (YAML)
│
├── observability/          # dashboard generator + 18 JSONs + scrape/otel/promtail
├── benchmarks/             # perf + TCO harness; benchmarks/bakeoff/ (Phase 13)
├── scripts/                # db_migrate/seed/backfill, setup/<c>.sh, validate/<c>.py
├── design/                 # figures + AI image-gen (design-time only)
├── spec/                   # this documentation set
├── agents-memory/          # AI-agent session notes + durable memory
│
└── src/kathirmani/             # LEGACY research cascade (see below)
    ├── __init__.py         #   PROJECT_ROOT / MODELS_DIR / RESULTS_DIR anchors
    ├── pipeline.py qwen_vl.py device.py ffmpeg_preload.py metrics.py loki.py …
    ├── cli/{run_inference,download_model}.py
    └── viz/app.py          #   Streamlit viewer
   root shims → src/kathirmani:  run_inference.py · download_model.py · serve_metrics.py
                             · viz_app.py · run.sh · setup.sh · start_stack.sh · cleanup.sh
   learning.md (original working notes, kept verbatim)
```

## Import conventions

- **Platform services/workers run flat.** The worker/service dirs are **hyphenated**
  (`ai-workers/cv-oss-worker`, `services/rule-engine`, …), which are *not* valid Python
  package names. So each runs with its own dir on `sys.path` and imports flat
  (`import worker`, `import plugin`), with a `try: from .x import … except ImportError:
  from x import …` shim in `__init__.py`. Per-dir `conftest.py` + `pytest.ini` put the
  package dir (and `model-plugins/`) on `sys.path` so tests collect in isolation.
- **Shared contracts**: `from base.plugin import ModelPlugin, …` / `from base.schemas
  import Detection, CommonEvent` (the `model-plugins/base` package).
- **Repo-root modules**: `from db import connect`, `from ingestion.bus import …`.
- No code imports these dirs dotted (`services.review_ui`) — only `services.security`
  and `services.api` are real (underscore/plain) packages.

> **Planned (deferred) refactor:** rename the hyphenated dirs to underscore packages
> (`cv_oss_worker`, `rule_engine`, `review_ui`, …) so they become real importable
> packages and the ~27 `sys.path` shims + 8 `conftest.py`/`pytest.ini` hacks can go.
> Tracked in [10-platform-roadmap.md](10-platform-roadmap.md) → *What's to be done*.

## Path anchoring (legacy)

Videos, `models/`, and `results/` live at the **repo root**. `src/kathirmani/__init__.py`
exposes the anchors so nothing depends on the cwd:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/kathirmani → repo root
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
```

`serve_metrics.py` re-derives the root locally (no `kathirmani` import) so the slim metrics
container can run standalone. `ingestion/config.py` and `scripts/fetch_models.py`
likewise re-derive the root locally — the platform never imports the anchors from
`kathirmani`. **Invariant:** keep the root shims + path anchors working and `learning.md`
verbatim (these are the legacy commands' contract).

## Deployment & the front door

The app tier (api / review-ui / console) is containerized by `Dockerfile.app` (a
torch-free image) and orchestrated by `docker-compose.platform.yml`; `make platform`
brings the whole stack up behind the Console on `:8080`. Design + ports + env:
**[14-console-and-deployment.md](14-console-and-deployment.md)**.

## Tests

`make test` runs each component dir in isolation (the `TESTDIRS` list) because the
hyphenated dirs share module basenames (`plugin.py`/`worker.py`) and can't be collected
together. The legacy `tests/` (device/structure/setup) + `tests/conftest.py` put `src/`
on `sys.path` so they import `kathirmani.*` without an editable install. Full suite as of
2026-06-11: **299 passed, 1 deselected** (the live-Grafana datasource check).

## Related

[14-console-and-deployment.md](14-console-and-deployment.md) ·
[10-platform-roadmap.md](10-platform-roadmap.md) ·
[06-hardware-portability.md](06-hardware-portability.md) · [07-runbook.md](07-runbook.md)
