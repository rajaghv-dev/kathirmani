#!/usr/bin/env bash
# Configure Grafana: add Netdata + Marlin metrics data sources, import dashboards.
#
# Run once after Grafana is up:
#   bash grafana/setup_grafana.sh
#
# Then start inference:
#   source .venv/bin/activate
#   python run_inference.py --video "Bill Counter" --duration 10 --no-compile
#
# Note: run_inference.py self-patches the FFmpeg path via RTLD_GLOBAL preload
# (no sudo / system FFmpeg required).  Just activate the venv and run.

set -euo pipefail

GRAFANA_URL="http://localhost:3000"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASS="${GRAFANA_PASS:-admin}"
NETDATA_URL="http://localhost:19999"
MARLIN_METRICS_URL="http://localhost:8900"

AUTH="${GRAFANA_USER}:${GRAFANA_PASS}"
CURL="curl -sS -u ${AUTH}"

echo "==> Checking Grafana ..."
${CURL} "${GRAFANA_URL}/api/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Grafana', d['version'])"

echo "==> Adding Netdata Prometheus data source ..."
${CURL} -X POST "${GRAFANA_URL}/api/datasources" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Netdata",
    "uid": "netdata-prom",
    "type": "prometheus",
    "url": "'"${NETDATA_URL}/api/v1/allmetrics?format=prometheus"'",
    "access": "proxy",
    "isDefault": false,
    "jsonData": { "httpMethod": "GET", "timeInterval": "5s" }
  }' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ', d.get('message', d.get('name','?')))" || true

echo "==> Adding Marlin inference Prometheus data source ..."
${CURL} -X POST "${GRAFANA_URL}/api/datasources" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "KathirmaniMetrics",
    "uid": "marlin-metrics",
    "type": "prometheus",
    "url": "'"${MARLIN_METRICS_URL}"'",
    "access": "proxy",
    "isDefault": true,
    "jsonData": { "httpMethod": "GET", "timeInterval": "5s" }
  }' 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print('  ', d.get('message', d.get('name','?')))" || true

echo "==> Creating Marlin folder ..."
FOLDER_UID=$(${CURL} -X POST "${GRAFANA_URL}/api/folders" \
  -H "Content-Type: application/json" \
  -d '{"uid":"marlin","title":"Marlin Inference"}' 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('uid','marlin'))" 2>/dev/null || echo "marlin")

echo "==> Importing dashboards ..."
for f in "$(dirname "$0")"/dashboard_*.json; do
  [ -e "$f" ] || continue
  DASHBOARD_JSON=$(cat "$f")
  ${CURL} -X POST "${GRAFANA_URL}/api/dashboards/db" \
    -H "Content-Type: application/json" \
    -d '{
      "dashboard": '"${DASHBOARD_JSON}"',
      "overwrite": true,
      "folderUid": "marlin"
    }' | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print('  $(basename "$f") ->', d.get('url') or d.get('message','(check Grafana UI)'))
except Exception:
    print('  $(basename "$f") imported (check Grafana UI)')
" 2>/dev/null || echo "  $(basename "$f") imported (check Grafana UI)"
done

echo ""
echo "===================================================="
echo "  Setup complete!"
echo "  Grafana:          http://localhost:3000"
echo "  Login:            admin / admin"
echo "  Dashboards:       Dashboards > Marlin Inference >"
echo "    - Marlin-2B — Model & Runtime"
echo "    - Marlin-2B — Application / Pipeline"
echo "    - Marlin — Model Performance & Usefulness   (uid marlin-model-perf)"
echo ""
echo "  Run inference (10s test on one camera):"
echo "    source .venv/bin/activate"
echo "    python run_inference.py --video 'Bill Counter' --duration 10 --no-compile"
echo ""
echo "  Run all cameras (full video):"
echo "    python run_inference.py"
echo ""
echo "  Metrics endpoint: http://localhost:8900/metrics"
echo "  (metrics server runs while run_inference.py is alive)"
echo "===================================================="
