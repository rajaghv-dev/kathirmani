#!/usr/bin/env bash
# Models: HF client + policy check. Pass --fetch to download the pinned NVIDIA shortlist.
set -euo pipefail; cd "$(dirname "$0")/../.."
.venv/bin/pip install -q -r requirements/models.txt
.venv/bin/python scripts/validate_model_config.py
if [ "${1:-}" = "--fetch" ]; then .venv/bin/python scripts/fetch_models.py; else
  echo "to fetch weights: .venv/bin/python scripts/fetch_models.py"; fi
echo "setup/models: OK"
