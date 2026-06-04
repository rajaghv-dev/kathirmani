#!/usr/bin/env bash
# Ingestion (PyAV). Inference deps (torch/transformers) install separately on the GPU box.
set -euo pipefail; cd "$(dirname "$0")/../.."
.venv/bin/pip install -q -r requirements/ingestion.txt
echo "setup/ingestion: OK"
