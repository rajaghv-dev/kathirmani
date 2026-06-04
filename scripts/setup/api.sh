#!/usr/bin/env bash
# API (FastAPI/uvicorn).
set -euo pipefail; cd "$(dirname "$0")/../.."
.venv/bin/pip install -q -r requirements/api.txt
echo "setup/api: OK"
