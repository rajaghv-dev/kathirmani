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
├── docker-compose.platform.yml        # app tier: migrate→api→review_ui→console
├── docker-compose.gpu.yml             # GPU/worker overlay
├── Dockerfile.app          # torch-free image for the api/review_ui/console tier
│
├── services/               # platform services (FastAPI)
│   ├── console/            #  ☜ FRONT DOOR — BFF/gateway: SPA + proxy (:8080)  → spec/14
│   ├── api/                #  read/query API + model-registry + /search (:8000)
│   ├── review_ui/          #  human approve/reject of loss hypotheses (:8010)
│   ├── rule_engine/        #  deterministic zone/dwell/interaction rules → incidents
│   ├── evidence_builder/   #  clip stitch + sha256 + timeline (chain-of-custody)
│   ├── digital_twin/       #  YAML store/camera/zone model; event→twin mapping
│   └── security/           #  auth + RBAC + audit + redaction + retention + backup
│
├── ai_workers/             # plugin-host inference workers (consume the queue)
│   ├── cv_oss_worker/      #  YOLOE detection + Tracker (real track_ids) → events
│   ├── vlm_worker/         #  Nemotron-VL (default) / Qwen baseline / fake_infer
│   ├── embedding_worker/   #  C-RADIO embeddings (+ random-projection head) + search
│   └── vss_eval_worker/    #  hierarchical summary + VSS-parity report
├── model_plugins/base/     # the ModelPlugin ABC + common schemas (Detection, …)
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
├── spec/                   # this documentation set (00 = the master plan)
├── agents-memory/          # AI-agent session notes + durable memory
│
├── store-videos/           # SOURCE FOOTAGE (.mkv, gitignored) — configs/cameras.yaml
│                           #   source_file names must match these files exactly
├── models/                 # model weights + PROVENANCE.json (fetched, not committed)
├── data/                   # runtime output: clips/thumbnails/queue/metadata (gitignored)
├── results/                # legacy-cascade JSON results (gitignored)
├── grafana/                # LEGACY-cascade dashboards + setup_grafana.sh (live-mounted
├── prometheus/             #   by compose/containers — do not move these two)
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

- **Everything is a real package** (2026-07-08 refactor: the formerly hyphenated
  dirs are underscore packages). Workers/services import package-qualified —
  siblings relatively (`from .plugin import CvOssDetector`), cross-package
  absolutely (`from ai_workers.embedding_worker.worker import query_run`).
- **Shared contracts**: `from model_plugins.base.plugin import ModelPlugin, …` /
  `from model_plugins.base.schemas import Detection, CommonEvent`.
- **Repo-root modules**: `from db import connect`, `from ingestion.bus import …`.
- **Entry points run from the repo root** — `python -m ai_workers.cv_oss_worker.worker`,
  `uvicorn services.review_ui.app:app` (uvicorn's default `--app-dir` puts the cwd on
  `sys.path`; the Makefile and compose files always run from the root / `/app`).
  There are no per-package `sys.path` shims, `pytest.ini`s, or dual
  package/flat-import `try/except` blocks anymore; the only path bootstrap left is
  the repo-root `conftest.py` (adds root + `src/` for pytest) and the legacy root
  shims (`run_inference.py` …) that anchor `src/`.

> The refactor previously deferred here (hyphenated dirs → underscore packages,
> dropping the ~27 `sys.path` shims + per-dir `conftest.py`/`pytest.ini` hacks)
> **landed 2026-07-08**. Compose *service names* (`review-ui`, `console`) and the
> generated dashboard *filenames* (`05-vlm-worker.json`) keep their hyphens — they
> are network/artifact identities, not Python paths.

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

The app tier (api / review_ui / console) is containerized by `Dockerfile.app` (a
torch-free image) and orchestrated by `docker-compose.platform.yml`; `make platform`
brings the whole stack up behind the Console on `:8080`. Design + ports + env:
**[14-console-and-deployment.md](14-console-and-deployment.md)**.

## Tests

`make test` is a single flat `pytest` run — every component's `tests/` dir is a
real package (unique module paths, no basename clashes), enumerated in
`pyproject.toml` `[tool.pytest.ini_options] testpaths`. The repo-root `conftest.py`
puts the root and `src/` on `sys.path`; `tests/conftest.py` additionally sets the
video-reader env vars + PyAV FFmpeg preload for the legacy suite. One deselect
remains (the live-Grafana datasource check, env-dependent).

## Related

[14-console-and-deployment.md](14-console-and-deployment.md) ·
[10-platform-roadmap.md](10-platform-roadmap.md) ·
[06-hardware-portability.md](06-hardware-portability.md) · [07-runbook.md](07-runbook.md)
