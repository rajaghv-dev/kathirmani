#!/usr/bin/env bash
# Render Grafana dashboards to PNG (needs the renderer sidecar — see
# docker-compose.observability.yml). Outputs to design/figures/dashboard_<name>.png.
#
# Usage: bash scripts/render_dashboards.sh [from] [to]   (defaults: now-7d now)
set -euo pipefail

GRAFANA="${GRAFANA_URL:-http://localhost:3000}"
AUTH="${GRAFANA_USER:-admin}:${GRAFANA_PASS:-admin}"
FROM="${1:-now-7d}"; TO="${2:-now}"
OUT="$(dirname "$0")/../design/figures"; mkdir -p "$OUT"

# name:uid  (the curated set worth screenshotting for docs/infographics)
DASHBOARDS=(
  "model_performance:marlin-model-perf:1600:1400"
  "economy:marlin-economy:1600:1100"
  "pipeline:marlin-pipeline:1600:1100"
  "cameras:marlin-cameras:1600:1100"
  "fused:marlin-fused:1600:1100"
  "qwen:marlin-qwen:1600:1100"
)

for entry in "${DASHBOARDS[@]}"; do
  IFS=":" read -r name uid w h <<< "$entry"
  out="$OUT/dashboard_${name}.png"
  curl -s -u "$AUTH" --max-time 90 \
    "${GRAFANA}/render/d/${uid}/x?orgId=1&width=${w}&height=${h}&theme=light&from=${FROM}&to=${TO}" \
    -o "$out"
  if file "$out" | grep -q PNG; then echo "  ✓ ${name} -> $out"; else echo "  ✗ ${name} (not a PNG — renderer up? data present?)"; fi
done
