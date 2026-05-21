#!/usr/bin/env bash
# Start the full observability stack: serve_metrics + Prometheus + (Grafana already running)
# Run from the project root:  bash start_stack.sh

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== Kathirmani Marlin Observability Stack ==="

# 1. Activate venv
source "$HERE/.venv/bin/activate"

# 2. serve_metrics.py — reads results/ JSONs, exposes :8900/metrics
if curl -s http://127.0.0.1:8900/metrics >/dev/null 2>&1; then
    echo "[serve_metrics] already running on :8900"
else
    nohup python "$HERE/serve_metrics.py" > /tmp/serve_metrics.log 2>&1 &
    sleep 2
    echo "[serve_metrics] started → http://127.0.0.1:8900/metrics"
fi

# 3. Prometheus — scrapes :8900 and Netdata :19999
if curl -s http://127.0.0.1:9090/-/ready >/dev/null 2>&1; then
    echo "[prometheus]    already running on :9090"
else
    nohup "$HERE/prometheus/prometheus" \
        --config.file="$HERE/prometheus/prometheus.yml" \
        --storage.tsdb.path="$HERE/prometheus/data" \
        --web.listen-address=0.0.0.0:9090 \
        > "$HERE/prometheus/prometheus.log" 2>&1 &
    sleep 3
    echo "[prometheus]    started → http://127.0.0.1:9090"
fi

# 4. Grafana (systemd)
if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
    echo "[grafana]       already running on :3000"
else
    echo "[grafana]       start with: sudo systemctl start grafana-server"
fi

HOST_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "Stack ready:"
echo "  Grafana   →  http://${HOST_IP}:3000   (admin / admin)"
echo "  Prometheus → http://${HOST_IP}:9090"
echo "  Metrics    → http://${HOST_IP}:8900/metrics"
echo ""
echo "Datasources in Grafana point to Prometheus at http://${HOST_IP}:9090 (browser mode)"
echo ""
echo "To run inference:"
echo "  python run_inference.py --video 'Bill Counter' --duration 10 --no-compile"
echo "  python run_inference.py  # all cameras, full video"
