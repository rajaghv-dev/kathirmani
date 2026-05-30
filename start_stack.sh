#!/usr/bin/env bash
# Start the full observability stack for Kathirmani Marlin inference.
# All services run as Docker containers on the monitoring_default network.
# Run from the project root:  bash start_stack.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NETWORK="monitoring_default"

echo "=== Kathirmani Marlin Observability Stack ==="

# ── 1. serve_metrics ─────────────────────────────────────────────────────────
echo "[serve_metrics] restarting ..."
docker rm -f serve_metrics 2>/dev/null || true
docker run -d \
    --name serve_metrics \
    --network "$NETWORK" \
    -v "$HERE":/app \
    -w /app \
    -p 8900:8900 \
    python:3.12-slim \
    sh -c "pip install prometheus_client -q && python serve_metrics.py"
echo "[serve_metrics] started on :8900 (Docker, network=$NETWORK)"

# ── 2. Prometheus ─────────────────────────────────────────────────────────────
echo "[prometheus]    restarting ..."
docker rm -f prometheus 2>/dev/null || true
chmod 777 "$HERE/prometheus/data" 2>/dev/null || true
docker run -d \
    --name prometheus \
    --network "$NETWORK" \
    --user "$(id -u):$(id -g)" \
    -v "$HERE/prometheus/prometheus.yml":/etc/prometheus/prometheus.yml:ro \
    -v "$HERE/prometheus/data":/prometheus \
    -p 9090:9090 \
    prom/prometheus:v2.52.0 \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.cors.origin=".*"
echo "[prometheus]    started on :9090 (Docker, network=$NETWORK)"

# ── 3. Grafana (already running as Docker container) ─────────────────────────
if docker ps --format '{{.Names}}' | grep -q "^grafana$"; then
    echo "[grafana]       already running on :3000"
else
    echo "[grafana]       not running — start your docker-compose stack"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
# Portable primary-IP detection: `hostname -I` is Linux-only (fails on macOS /
# Mac Studio), so fall back to ipconfig (macOS) and finally localhost.
host_ip() {
    if hostname -I >/dev/null 2>&1; then
        hostname -I | awk '{print $1}'
    elif command -v ipconfig >/dev/null 2>&1; then
        ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo localhost
    else
        echo localhost
    fi
}
HOST_IP=$(host_ip)
[ -z "$HOST_IP" ] && HOST_IP=localhost
echo ""
echo "Stack ready:"
echo "  Grafana    →  http://${HOST_IP}:3000  (admin / admin)"
echo "  Prometheus →  http://${HOST_IP}:9090"
echo "  Metrics    →  http://${HOST_IP}:8900/metrics"
echo ""
echo "Datasource: MarlinMetrics → http://prometheus:9090 (proxy, Docker DNS)"
echo ""
echo "To run inference:"
echo "  source .venv/bin/activate"
echo "  python run_inference.py --video 'Bill Counter' --duration 10 --no-compile"
echo "  python run_inference.py   # all cameras, full video"
