#!/usr/bin/env bash
# Observability: (re)start serve_metrics + prometheus; first-time datasources/dashboards.
set -euo pipefail; cd "$(dirname "$0")/../.."
bash start_stack.sh
echo "first run only: bash grafana/setup_grafana.sh  (datasources + dashboards + renderer)"
echo "setup/observability: OK"
