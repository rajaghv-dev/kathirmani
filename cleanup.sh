#!/usr/bin/env bash
# Reset all inference data: Prometheus TSDB, serve_metrics, results JSONs.
# Dashboards and datasources in Grafana are kept — only data is wiped.
#
# Usage:
#   bash cleanup.sh          # wipes everything, restarts stack
#   bash cleanup.sh --dry-run  # show what would be deleted

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

log()  { echo "[cleanup] $*"; }
run()  { $DRY_RUN && echo "  DRY: $*" || eval "$*"; }

echo "=================================================="
echo "  Kathirmani — Data Cleanup"
$DRY_RUN && echo "  (DRY RUN — nothing will be changed)"
echo "=================================================="

# ── 1. Stop containers that hold data in memory ────────────────────────────
log "Stopping serve_metrics and prometheus containers ..."
run "docker stop serve_metrics prometheus 2>/dev/null || true"

# ── 2. Wipe Prometheus TSDB ────────────────────────────────────────────────
PROM_DATA="$HERE/prometheus/data"
log "Clearing Prometheus TSDB: $PROM_DATA"
if $DRY_RUN; then
    du -sh "$PROM_DATA" 2>/dev/null && echo "  DRY: would delete contents"
else
    if [[ -d "$PROM_DATA" ]]; then
        SIZE=$(du -sh "$PROM_DATA" 2>/dev/null | cut -f1)
        rm -rf "${PROM_DATA:?}"/*
        log "  Cleared ${SIZE} of TSDB data"
    fi
fi

# ── 3. Wipe inference result JSONs ─────────────────────────────────────────
RESULTS="$HERE/results"
log "Clearing inference results: $RESULTS"
if $DRY_RUN; then
    ls "$RESULTS"/*.json 2>/dev/null && echo "  DRY: would delete above files" || echo "  (no results yet)"
else
    if [[ -d "$RESULTS" ]]; then
        COUNT=$(ls "$RESULTS"/*.json 2>/dev/null | wc -l)
        rm -f "$RESULTS"/*.json
        log "  Deleted ${COUNT} result file(s)"
    fi
fi

# ── 4. Restart containers ──────────────────────────────────────────────────
if ! $DRY_RUN; then
    log "Restarting serve_metrics ..."
    docker start serve_metrics 2>/dev/null && sleep 3 || log "  serve_metrics not found — run start_stack.sh"

    log "Restarting prometheus ..."
    docker start prometheus 2>/dev/null && sleep 3 || log "  prometheus not found — run start_stack.sh"
fi

# ── 5. Grafana: clear annotations (keep dashboards + datasources) ──────────
log "Clearing Grafana annotations ..."
if $DRY_RUN; then
    echo "  DRY: would DELETE /api/annotations (all)"
else
    COUNT=$(curl -s -u admin:admin http://localhost:3000/api/annotations \
      | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo 0)
    if [[ "$COUNT" -gt 0 ]]; then
        curl -s -u admin:admin -X DELETE \
            "http://localhost:3000/api/annotations" \
            -H "Content-Type: application/json" \
            -d '{"dashboardId":0,"panelId":0}' >/dev/null 2>&1 || true
        log "  Cleared ${COUNT} annotation(s)"
    else
        log "  No annotations to clear"
    fi
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
if $DRY_RUN; then
    echo "  DRY RUN complete — nothing changed."
else
    echo "  Cleanup complete."
    echo ""
    echo "  Prometheus TSDB : empty"
    echo "  Results JSONs   : deleted"
    echo "  Grafana         : annotations cleared, dashboards kept"
    echo ""
    echo "  Ready for next run:"
    echo "    source .venv/bin/activate"
    echo "    python run_inference.py --duration 10 --no-compile"
fi
echo "=================================================="
