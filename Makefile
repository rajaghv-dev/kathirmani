# Kathirmani platform — single entrypoint (master plan §18 + Addendum A).
# Targets marked [stub] print intent until their phase lands (see spec/10-platform-roadmap).
COMPOSE := docker compose -f docker-compose.yml -f docker-compose.observability.yml
COMPOSE_GPU := $(COMPOSE) -f docker-compose.gpu.yml
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
	@python3 -m py_compile scripts/*.py scripts/validate/*.py model-plugins/base/*.py ingestion/*.py db/*.py && echo "lint OK"

# ---- Stack lifecycle --------------------------------------------------------
.PHONY: up down logs grafana observability
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
.PHONY: migrate migrate-down migrate-status seed ingest-sample run-workers test test-e2e bench evidence-demo
migrate: ## Apply db/migrations (Postgres)
	python3 scripts/db_migrate.py up
migrate-down: ## Roll back the latest migration
	python3 scripts/db_migrate.py down
migrate-status: ## Show applied migrations
	python3 scripts/db_migrate.py status
seed: ## Seed kathirmani store/cameras/zones + model profiles/registry
	python3 scripts/db_seed.py
ingest-sample: ## Segment the 5 .mkv into 10-sec clips + 5-sec windows (DURATION=secs)
	python3 -m ingestion $(if $(DURATION),--duration $(DURATION),--duration 30) $(if $(CAMERA),--camera $(CAMERA),)
run-workers: ## [stub→Phase 4/6] start cv/vlm/embedding plugin-host workers
	@echo "[stub] Phase 4/6: start workers for profile=$(PROFILE)"
test: ## Run the test suite (platform + ingestion)
	pytest -q tests/ ingestion/tests/
test-e2e: ## [stub→Phase 7] full-scenario gate
	@echo "[stub] Phase 7: end-to-end scenario"
bench: ## [stub→Phase 11] benchmarks → model_benchmark_runs
	@echo "[stub] Phase 11: streams/GPU, clips-min/GPU, TCO"
evidence-demo: ## [stub→Phase 7] build + show one evidence package
	@echo "[stub] Phase 7: evidence package demo"

# ---- Plugins ----------------------------------------------------------------
.PHONY: test-plugin bench-plugin
test-plugin: ## [stub→Phase 6] 10-point plugin test (PLUGIN=<name>)
	@echo "[stub] A11 plugin test for PLUGIN=$(PLUGIN)"
bench-plugin: ## [stub→Phase 13] plugin benchmark (PLUGIN=<name> MODEL=<id>)
	@echo "[stub] A9 bench for PLUGIN=$(PLUGIN) MODEL=$(MODEL)"
