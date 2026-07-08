# 2026-07-08 — Hyphen→underscore package refactor (repo-wide)

Executed the refactor deferred in spec/10 §B / spec/09: every hyphenated
worker/service dir is now a real importable Python package, and all docs/tests/
entry points were re-aligned in the same pass.

## Renames (git mv — history preserved)

- `ai-workers/` → `ai_workers/` (`cv_oss_worker`, `vlm_worker`, `embedding_worker`,
  `vss_eval_worker`)
- `model-plugins/` → `model_plugins/` (+ new `__init__.py` for `model_plugins` and
  `model_plugins/base` — `base` was previously a bare dir found via sys.path)
- `services/rule-engine|review-ui|evidence-builder|digital-twin` →
  `services/rule_engine|review_ui|evidence_builder|digital_twin`

**Kept hyphenated (identities, not Python paths):** compose service names
(`review-ui`, `console`), generated dashboard filenames (`05-vlm-worker.json`),
Makefile target names (`run-review-ui`, `run-cv-worker`).

## Code changes

- Removed the ~27 `sys.path.insert` shims and every dual package/flat
  `try: from .x import … except ImportError:` block. Siblings import relatively
  (`from .plugin import CvOssDetector`); cross-package imports are absolute
  (`from model_plugins.base.schemas import Detection`,
  `from ai_workers.embedding_worker.worker import query_run` in services/api).
- Caught a lazy in-function flat import too: `plugin.detections_to_events` did
  `from zones import …` at call time → `from .zones import …` (the full-suite run
  exposed it; per-dir runs had masked it via conftest sys.path).
- `services/review_ui/__init__` no longer re-exports the FastAPI instance as
  `app` — it was shadowing the `app` **submodule**, which broke
  `from services.review_ui import app` module imports. uvicorn target is
  `services.review_ui.app:app`.

## Test config

- Deleted all 9 per-dir `pytest.ini` + 14 sys-path-shim `conftest.py`s. Each
  `tests/` dir got an `__init__.py` (unique module paths → no basename clashes).
- New repo-root `conftest.py` (root + `src/` on sys.path). `tests/conftest.py`
  (env vars + PyAV FFmpeg preload) kept.
- `pyproject.toml` `testpaths` lists all 16 test dirs; `make test` = ONE flat
  pytest run (was: 16 isolated per-dir invocations).

## Entry points

- Makefile: workers via `python -m ai_workers.<w>.worker` / dotted uvicorn
  (`services.console.app:app`) — no more `cd <dir> && python -m worker`;
  `twin-validate` imports `services.digital_twin.loader` directly.
- docker-compose.platform.yml: review-ui/console run from `working_dir: /app`
  with dotted app paths (uvicorn's default `--app-dir` handles sys.path).
- All CLIs smoke-verified: cv/rule/vlm/vss workers `--help`, `make twin-validate`
  end-to-end, `docker compose config -q`, `make lint`, `make dashboards`.

## Live-env fix found along the way

`tests/test_setup.py::test_metrics_server_up` was failing because the
`serve_metrics` Docker container still bind-mounted the **deleted** pre-rename
repo path (`~/raja/kathir`) and served `marlin_*` metric names, while all 8
`grafana/dashboard_*.json` query `kathirmani_*`. Recreated the container against
the current checkout (same image/network/port) → 493 `kathirmani_*` series, test
green, dashboards live again.

## Known-remaining (pre-existing, NOT from this refactor)

- `tests/test_setup.py::test_venv_packages` — torchcodec can't load
  `libtorchcodec` against this box's FFmpeg (env/provisioning, predates the
  refactor; the rest of the legacy suite passes).

Docs re-aligned: README, PLATFORM.md, spec/02/08/09 (rewritten import
conventions + Tests), spec/10 (§B marked DONE), 11/13/14/16/17, master plan,
observability/README, design/INFOGRAPHIC-BRIEF, memory.md. Dated session notes
left verbatim (historical record).

---

# Same day, follow-up — store-videos/ wiring + root reorg

## store-videos/ (source footage home)

User dropped the camera footage into `store-videos/` at the repo root (4 of the
5 files; Center Camera missing). Wired everywhere, gitignored:

- `configs/cameras.yaml` `source_file` values updated to the ACTUAL filenames —
  incl. `"Kathirmani Store Near Bill Counter.mkv"` and the typo'd
  `"Kathirmni Entry Cam.mkv"` (match files exactly, don't "fix" the typo).
- `ingestion.sources.source_for_camera` resolves `store-videos/<file>` first,
  repo root as fallback. `ingestion.config.VIDEOS_DIR` added.
- Legacy layer: `kathirmani.VIDEOS_DIR` anchor (falls back to PROJECT_ROOT);
  `run_inference --video-dir` defaults to it; viz app scans it.
- Verified end-to-end: `python -m ingestion --duration 10 --camera bill_counter`
  → 1 clip + 4 windows from the real footage.

## Root reorg (conservative — live infra pinned in place)

- Master plan `oss_ingestion_nvidia_model_plugin_master_plan_v2.md` →
  `spec/00-master-plan-v2.md`; all non-session references + spec/README index
  updated.
- A stray 26 GB Xilinx FPGA SDK installer tar (`test@zlabslen-d12-3`, unrelated
  to this repo) moved out to `~/`.
- `grafana/` and `prometheus/` deliberately NOT moved: bind-mounted by live
  containers/compose (`./grafana:/etc/grafana/provisioning`, prometheus TSDB).
- Makefile now pins `COMPOSE_PROJECT_NAME=kathirmani-platform` so `make platform`
  from the renamed repo dir reuses the existing project + pgdata volume instead
  of creating a parallel stack.

## Live-stack staleness found (user action needed)

ALL app-tier containers (api/review-ui/console, up 4 weeks) + the prometheus
container bind-mount the pre-rename path `~/raja/kathir` (deleted) and run
pre-refactor in-memory code. Fix = `make platform` (project name now pinned) and
re-run `bash start_stack.sh` for prometheus/serve_metrics. serve_metrics was
already recreated (2026-07-08) — it was serving `marlin_*` names while all
dashboards query `kathirmani_*`.
