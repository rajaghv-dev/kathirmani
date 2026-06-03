# Kathirmani platform — single entrypoint (master plan §18 + Addendum A).
# Targets marked [stub] print intent until their phase lands (see spec/10-platform-roadmap).
COMPOSE := docker compose -f docker-compose.yml -f docker-compose.observability.yml
COMPOSE_GPU := $(COMPOSE) -f docker-compose.gpu.yml
PROFILE ?= $(or $(MODEL_PROFILE),nvidia_gb10_retail_balanced)

.PHONY: help
help: ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---- Phase 0: setup, config, models ----------------------------------------
.PHONY: setup setup-nvidia-docker config-check validate-model-config docker-config fetch-models lint
setup: ## Host prerequisites (delegates to existing setup.sh)
	bash setup.sh
setup-nvidia-docker: ## Register the Docker NVIDIA runtime (needs sudo)
	bash scripts/setup_nvidia_docker.sh
config-check: validate-model-config ## Parse all configs + run policy validator
	@python3 -c "import sys,glob,yaml; [yaml.safe_load(open(f)) for f in glob.glob('configs/**/*.yaml',recursive=True)]; print('configs parse OK')"
validate-model-config: ## Enforce the NVIDIA-only model policy (A5.2)
	python3 scripts/validate_model_config.py
docker-config: ## Validate the compose stack parses
	$(COMPOSE_GPU) config -q && echo "compose config OK"
fetch-models: ## Download + pin the NVIDIA model shortlist (writes models/PROVENANCE.json)
	python3 scripts/fetch_models.py
lint: ## Byte-compile python sources
	@python3 -m py_compile scripts/*.py model-plugins/base/*.py && echo "lint OK"

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
.PHONY: migrate seed ingest-sample run-workers test test-e2e bench evidence-demo
migrate: ## [stub→Phase 2] apply db/migrations
	@echo "[stub] Phase 2: apply db/migrations to Postgres"
seed: ## [stub→Phase 2] seed kathirmani store/cameras/zones
	@echo "[stub] Phase 2: seed from configs/stores/kathirmani.yaml"
ingest-sample: ## [stub→Phase 1] segment the 5 .mkv into clips/windows
	@echo "[stub] Phase 1: ingest configs/cameras.yaml source files"
run-workers: ## [stub→Phase 4/6] start cv/vlm/embedding plugin-host workers
	@echo "[stub] Phase 4/6: start workers for profile=$(PROFILE)"
test: ## Run the test suite
	pytest -q tests/
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
