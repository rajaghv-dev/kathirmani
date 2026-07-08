# Kathirmani platform — single entrypoint (master plan §18 + Addendum A).
# Targets marked [stub] print intent until their phase lands (see spec/10-platform-roadmap).
# Project name pinned so the stack (and its pgdata volume) survives repo-dir renames
# (the live containers/volume were created as `kathirmani-platform`).
export COMPOSE_PROJECT_NAME ?= kathirmani-platform
COMPOSE := docker compose -f docker-compose.yml -f docker-compose.observability.yml
COMPOSE_GPU := $(COMPOSE) -f docker-compose.gpu.yml
COMPOSE_PLATFORM := $(COMPOSE) -f docker-compose.platform.yml
PROFILE ?= $(or $(MODEL_PROFILE),nvidia_gb10_retail_balanced)

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- Setup (per component — see scripts/setup/, frameworks in requirements/) ----
.PHONY: setup setup-env setup-ingestion setup-db setup-models setup-api setup-observability setup-gpu setup-inference setup-nvidia-docker
setup: ## Install all component Python deps (env/ingestion/models/api)
	bash scripts/setup/all.sh
setup-env: ## venv + shared libs + .env
	bash scripts/setup/env.sh
setup-ingestion: ## ingestion deps (PyAV)
	bash scripts/setup/ingestion.sh
setup-db: ## db client + migrate + seed (needs postgres up)
	bash scripts/setup/db.sh
setup-models: ## HF client + policy check (FETCH=1 to download weights)
	bash scripts/setup/models.sh $(if $(FETCH),--fetch,)
setup-api: ## FastAPI/uvicorn
	bash scripts/setup/api.sh
setup-observability: ## (re)start serve_metrics + prometheus
	bash scripts/setup/observability.sh
setup-gpu setup-nvidia-docker: ## Register the Docker NVIDIA runtime (needs sudo)
	bash scripts/setup/gpu.sh
setup-inference: ## Heavy inference stack (torch/transformers) via machine-aware setup.sh
	bash setup.sh

# ---- Validation (per component — see scripts/validate/) ---------------------
.PHONY: validate doctor validate-env validate-ingestion validate-db validate-models validate-observability validate-api validate-gpu config-check validate-model-config docker-config fetch-models lint
validate doctor: ## Run all component validators (status matrix)
	python3 scripts/validate/doctor.py
validate-env: ; python3 scripts/validate/env.py
validate-ingestion: ; python3 scripts/validate/ingestion.py
validate-db: ; python3 scripts/validate/db.py
validate-models: ; python3 scripts/validate/models.py
validate-observability: ; python3 scripts/validate/observability.py
validate-api: ; python3 scripts/validate/api.py
validate-gpu: ; python3 scripts/validate/gpu.py
config-check: validate-model-config ## Parse all configs + run policy validator
	@python3 -c "import sys,glob,yaml; [yaml.safe_load(open(f)) for f in glob.glob('configs/**/*.yaml',recursive=True)]; print('configs parse OK')"
validate-model-config: ## Enforce the NVIDIA-only model policy (A5.2)
	python3 scripts/validate_model_config.py
docker-config: ## Validate the compose stack parses
	$(COMPOSE_GPU) config -q && echo "compose config OK"
fetch-models: ## Download + pin the NVIDIA model shortlist (writes models/PROVENANCE.json)
	python3 scripts/fetch_models.py
lint: ## Byte-compile python sources
	@python3 -m py_compile scripts/*.py scripts/validate/*.py model_plugins/base/*.py ingestion/*.py db/*.py && echo "lint OK"

# ---- Stack lifecycle --------------------------------------------------------
.PHONY: up down logs grafana observability platform platform-down platform-logs
platform: ## ☜ ONE COMMAND: bring up the whole platform behind http://localhost:8080
	$(COMPOSE_PLATFORM) up -d --build
	@echo ""; echo "  ┌────────────────────────────────────────────────────┐"
	@echo "  │  Kathirmani platform is up.                            │"
	@echo "  │  ▶ Console (front door):  http://localhost:8080        │"
	@echo "  │    API + Swagger /docs:   http://localhost:8000/docs   │"
	@echo "  │    Grafana:               http://localhost:3000        │"
	@echo "  └────────────────────────────────────────────────────┘"
platform-down: ## Stop the whole platform (app tier + base + observability)
	$(COMPOSE_PLATFORM) down
platform-logs: ## Tail the app-tier logs (console/api/review-ui)
	$(COMPOSE_PLATFORM) logs -f --tail=100 console api review-ui
up: ## Bring up base + observability stack
	$(COMPOSE) up -d
down: ## Stop the stack
	$(COMPOSE_GPU) down
logs: ## Tail stack logs
	$(COMPOSE_GPU) logs -f --tail=100
observability: ## Current observability stack (existing script until compose migration)
	bash start_stack.sh
grafana: ## Print the Grafana URL
	@echo "Grafana: http://localhost:3000 (admin/admin) — dashboards 01-18"

# ---- DB + workers + tests (filled per phase) --------------------------------
.PHONY: migrate migrate-down migrate-status seed backfill api run-cv-worker run-rule-engine run-vlm-worker index search run-review-ui console run-console summarize parity retention-dryrun retention-apply backup dashboards twin-validate ingest-sample run-workers test test-e2e bench bakeoff evidence-demo
migrate: ## Apply db/migrations (Postgres)
	python3 scripts/db_migrate.py up
migrate-down: ## Roll back the latest migration
	python3 scripts/db_migrate.py down
migrate-status: ## Show applied migrations
	python3 scripts/db_migrate.py status
seed: ## Seed kathirmani store/cameras/zones + model profiles/registry
	python3 scripts/db_seed.py
ingest-sample: ## Segment store-videos/*.mkv into 10-sec clips + 5-sec windows (DURATION=secs)
	python3 -m ingestion $(if $(DURATION),--duration $(DURATION),--duration 30) $(if $(CAMERA),--camera $(CAMERA),)
backfill: ## Load ingestion JSONL (data/metadata) into Postgres
	python3 scripts/backfill_ingest.py
api: ## Run the platform API (FastAPI/uvicorn on :8000)
	.venv/bin/uvicorn services.api.app:app --host 0.0.0.0 --port 8000
run-cv-worker: ## Phase 4: OSS CV worker — consume ai_window.ready, emit detections/events
	INGEST_QUEUE=pg .venv/bin/python -m ai_workers.cv_oss_worker.worker $(if $(ONCE),--once,) --limit $(or $(LIMIT),8)
run-rule-engine: ## Phase 5: rule engine — consume event.created, raise hypotheses + incidents
	INGEST_QUEUE=pg .venv/bin/python -m services.rule_engine.worker $(if $(ONCE),--once,) --limit $(or $(LIMIT),8)
run-vlm-worker: ## Phase 6: VLM worker — verify needs_vlm events → vlm_observations
	INGEST_QUEUE=pg .venv/bin/python -m ai_workers.vlm_worker.worker $(if $(ONCE),--once,) --limit $(or $(LIMIT),8)
run-workers: run-cv-worker run-rule-engine run-vlm-worker ## Start the AI workers (cv + rules + vlm)
index: ## Phase 8: embed + index events/observations into pgvector
	.venv/bin/python -m ai_workers.embedding_worker.worker index
search: ## Phase 8: natural-language search (QUERY="...")
	.venv/bin/python -m ai_workers.embedding_worker.worker query "$(QUERY)"
run-review-ui: ## Phase 7: human review UI (FastAPI :8010)
	.venv/bin/uvicorn services.review_ui.app:app --host 0.0.0.0 --port 8010
console: ## THE FRONT DOOR — unified console gateway (SPA + proxy) on :8080
	.venv/bin/uvicorn services.console.app:app --host 0.0.0.0 --port 8080
run-console: console ## alias for `make console`
summarize: ## Phase 9: hierarchical long-video summary (REQUEST=path.json optional)
	.venv/bin/python -m ai_workers.vss_eval_worker.worker summarize $(if $(REQUEST),--request $(REQUEST),)
parity: ## Phase 9: build the VSS-parity report → parity_report.json
	.venv/bin/python -m ai_workers.vss_eval_worker.worker parity
retention-dryrun: ## Phase 12: show what retention cleanup would delete (dry run)
	.venv/bin/python -m services.security.retention
retention-apply: ## Phase 12: apply retention cleanup (evidence-locked rows skipped)
	.venv/bin/python -m services.security.retention --apply
backup: ## Phase 12: pg_dump + clips manifest backup
	.venv/bin/python -c "from services.security.backup import run_backup; print(run_backup('backups','data/clips',execute=True))"
dashboards: ## Phase 3: (re)generate the 01-18 Grafana dashboard JSONs
	python3 observability/grafana/make_dashboards.py
figures: ## Design: (re)render the matplotlib infographics into design/figures/
	.venv/bin/python design/make_figures.py
setup-design: ## Design: install AI image-gen deps (Nano Banana + gpt-image-1)
	.venv/bin/pip install -r requirements/design.txt
gen-image: ## Design: AI image (PROVIDER=gemini|openai|both OUT=name PROMPT="..."|PROMPT_FILE=path [REF=path])
	.venv/bin/python design/ai_images.py --provider $(or $(PROVIDER),gemini) \
		--out $(or $(OUT),generated) \
		$(if $(PROMPT_FILE),--prompt-file $(PROMPT_FILE),--prompt "$(PROMPT)") \
		$(if $(REF),--ref $(REF),)
twin-validate: ## Phase 10: validate a store digital twin (STORE=configs/stores/kathirmani.yaml)
	.venv/bin/python -c "import sys; from services.digital_twin.loader import load_twin; t=load_twin('$(or $(STORE),configs/stores/kathirmani.yaml)'); p=t.validate(); print(t.summary()); print('problems:',p); sys.exit(1 if p else 0)"
# One flat run — every tests/ dir is a real package (see pyproject testpaths). The
# deselect skips a live-Grafana integration test (env-dependent, not platform code).
test: ## Run the full test suite
	.venv/bin/pytest -q --deselect tests/test_setup.py::test_grafana_datasource_marlin
test-e2e: ## Full-scenario chain: ingest → backfill → cv → rules → vlm → evidence
	@echo "e2e: make ingest-sample && make backfill && make run-workers ONCE=1 && make evidence-demo INCIDENT=<id>"
bench: ## Phase 11: run all benchmarks (fake mode) → model_benchmark_runs + reports
	.venv/bin/python -m benchmarks.run --all
bakeoff: ## Phase 13: NVIDIA runtime bake-off (fake) → ranked production/fallback profile
	.venv/bin/python -m benchmarks.bakeoff.run --task vlm_clip_reasoning
evidence-demo: ## Phase 7: build an evidence package (INCIDENT=<incident_id>)
	.venv/bin/python -m services.evidence_builder.builder $(INCIDENT)

# ---- Plugins ----------------------------------------------------------------
.PHONY: test-plugin bench-plugin
test-plugin: ## [stub→Phase 6] 10-point plugin test (PLUGIN=<name>)
	@echo "[stub] A11 plugin test for PLUGIN=$(PLUGIN)"
bench-plugin: ## [stub→Phase 13] plugin benchmark (PLUGIN=<name> MODEL=<id>)
	@echo "[stub] A9 bench for PLUGIN=$(PLUGIN) MODEL=$(MODEL)"
