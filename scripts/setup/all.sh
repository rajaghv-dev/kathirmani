#!/usr/bin/env bash
# Install all Python component deps. Service/sudo steps are called out, not auto-run.
set -euo pipefail; cd "$(dirname "$0")/../.."
for c in env ingestion models api; do bash "scripts/setup/$c.sh"; done
echo
echo "next (need services / sudo):"
echo "  make up                       # postgres + base services"
echo "  bash scripts/setup/db.sh      # migrate + seed (needs postgres up)"
echo "  bash scripts/setup/observability.sh"
echo "  bash scripts/setup/gpu.sh     # sudo: Docker NVIDIA runtime"
echo "  bash scripts/setup/models.sh --fetch   # ~22GB NVIDIA weights"
echo "then: make validate"
