# Platform Observability (Phase 3)

Delivers the master-plan dashboard set **01–18** plus the scrape / trace / log
configs that feed them. **Additive** to the existing `kathirmani_*` stack — it does not
modify `grafana/dashboard_*.json`, `prometheus/prometheus.yml`, or the compose files.

| Layer | What | Source of truth |
|-------|------|-----------------|
| Dashboards 01–10 (platform) | ingestion / camera / queue / CV / VLM / GPU / event-quality / incidents / twin / TCO | `ingest_*` (`ingestion/metrics.py`), `model_*`, `kathirmani_*` |
| Dashboards 11–18 (model bake-off) | registry / throughput / latency / quality / TCO / VSS-parity / failure-drift / runtime-comparison | `model_*` (`model-plugins/base/metrics.py`, spec/11 A8) |
| Scrape | `ingest_*` (:8901), `model_*` (:8902), DCGM (:9400) | `prometheus/platform_scrape.yml` |
| Traces | segment→queue→worker→event (OTLP) | `otel-collector/config.yaml` |
| Logs | per-service JSON → Loki, searchable by `trace_id` / `camera_id` / `segment_id` | `promtail/config.yml` |

## Layout

```
observability/
  grafana/make_dashboards.py        # headless generator (emits JSON, no Grafana needed)
  grafana/dashboards/*.json         # generated; 01-ingestion.json … 18-runtime-comparison.json
  prometheus/platform_scrape.yml    # ADDITIVE scrape jobs (ingest_* / model_* / dcgm)
  otel-collector/config.yaml        # OTLP receiver -> logging (+ commented prometheus/loki)
  promtail/config.yml               # container + JSONL logs -> Loki
  tests/test_dashboards.py          # pytest gate
```

## 1. Generate / regenerate dashboards

```bash
.venv/bin/python observability/grafana/make_dashboards.py
# writes observability/grafana/dashboards/*.json (18 files)
```

Each dashboard: `schemaVersion 39`, unique `uid` (`kathir-NN-…`), an **About this
dashboard** text panel (durable explain-meaning preference), datasource
`{type: prometheus, uid: marlin-metrics}`, and panels whose `targets[].expr` use the
real documented metric names. Edit panels/queries in `make_dashboards.py` and rerun —
the JSON is generated, never hand-edited.

## 2. Import into Grafana

Reuse the existing `grafana/setup_grafana.sh` pattern — POST each file to
`/api/dashboards/db`. The `marlin-metrics` datasource and renderer sidecar are already
wired by that script / `docker-compose.observability.yml`. One-liner against a running
Grafana (admin/admin):

```bash
GRAFANA_URL=http://localhost:3000
for f in observability/grafana/dashboards/*.json; do
  curl -sS -u admin:admin -X POST "$GRAFANA_URL/api/dashboards/db" \
    -H 'Content-Type: application/json' \
    -d "{\"dashboard\": $(cat "$f"), \"overwrite\": true, \"folderUid\": \"marlin\"}" \
    >/dev/null && echo "imported $(basename "$f")"
done
```

(Or copy these files next to `grafana/dashboard_*.json` only if you want
`setup_grafana.sh`'s glob to pick them up — but that script's loop targets
`grafana/dashboard_*.json`, so the standalone loop above is the clean path.)

## 3. Add the scrape targets to Prometheus

Prometheus has no `include` directive, so merge the jobs in `platform_scrape.yml`
into the live config. Append its three `scrape_configs` entries under the existing
`scrape_configs:` in `prometheus/prometheus.yml`:

```yaml
# prometheus/prometheus.yml  (append under scrape_configs:)
  - job_name: kathir_ingestion
    static_configs: [{ targets: ["ingestion:8901"] }]
  - job_name: kathir_model_worker
    static_configs: [{ targets: ["worker:8902"] }]
  - job_name: kathir_dcgm
    static_configs: [{ targets: ["dcgm-exporter:9400"] }]
```

**Ports:** ingestion `/metrics` = **8901** (`ingestion/config.py METRICS_PORT`),
model worker `/metrics` = **8902** (chosen here for the plugin-host worker), DCGM
= **9400** (`docker-compose.observability.yml`, `gpu-metrics` profile). Bare-metal:
replace the compose hostnames with `localhost`. Verify with
`curl localhost:9090/api/v1/targets`.

## 4. Traces (OTel) and logs (Promtail)

- **OTel:** `otelcol --config observability/otel-collector/config.yaml`. Services set
  `OTEL_EXPORTER_OTLP_ENDPOINT` to the collector (gRPC :4317 / HTTP :4318). Default
  exporter is `logging` (zero deps); uncomment the `prometheus` / `loki` exporters when
  those sinks are wired. The `trace_id` ties an RTSP frame → segment → window → model output.
- **Promtail:** `promtail -config.file=observability/promtail/config.yml`. Ships
  container stdout + `/var/log/kathir/*.jsonl` to the existing Loki, promoting
  `service` / `camera_id` / `level` to labels and keeping `trace_id` / `segment_id`
  as JSON fields (searchable via `| json | trace_id="…"`).

## 5. Test gate

```bash
.venv/bin/pytest observability/tests/ -q   # asserts 18 valid JSON, unique uids, About panel each
```

## Status: template vs. wired

- **Wired now:** the generator + 18 JSON files + the pytest gate are real and green.
  Dashboard import reuses the proven `setup_grafana.sh` / `/api/dashboards/db` path.
- **Template / pending live data:** panels render once the `ingest_*` (Phase 1) and
  `model_*` (Phase 4/6 workers) producers actually expose `/metrics` on :8901 / :8902.
  Dashboards **09 (twin)**, **15 (model TCO)** and **16 (VSS-parity)** use proxy queries
  over existing metrics (noted in their About panel) until their dedicated metrics land
  (Phases 9/10/13). The OTel `prometheus`/`loki` exporters and DCGM GPU panels are
  commented/opt-in until those services are up.
