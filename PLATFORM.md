# Kathirmani Platform — Start Here

The platform is several cooperating services. **You only need one URL.**

## ① The front door — the Console

```bash
make up          # base stack: postgres (+ observability)   [needs docker]
make api         # platform read/query API           → :8000   (terminal 1)
make run-review-ui   # human review service           → :8010   (terminal 2)
make run-workers     # cv + rule_engine + vlm workers          (terminal 3, optional)
make console     # ☜ THE UNIFIED CONSOLE             → :8080   (terminal 4)

open http://localhost:8080
```

`http://localhost:8080` is the **single cohesive UI**. It's a gateway (BFF): it serves
one single-page console and proxies to the API, the review service, and Grafana —
so the browser talks to one origin and you never juggle ports.

Everything still runs standalone (you can hit `:8000`, `:8010`, `:3000` directly),
but the console is the product surface.

**Source footage** lives in `store-videos/` at the repo root (gitignored;
filenames must match `configs/cameras.yaml` → `source_file`). Populate the
pipeline from it with `make ingest-sample` → `make backfill` → `make run-workers`.

## ② What's in the console (left sidebar)

| Section | What it shows | Backed by |
|---|---|---|
| **Overview** | KPIs (cameras, segments, open incidents) + recent events | `GET /cameras /segments /incidents /events` |
| **Incidents** | Loss hypotheses queue → click to review → **Approve / Reject** (audited) | review service `/incidents/{id}` + `/incidents/{id}/review` |
| **Events** | Perception + rule_engine signal stream | `GET /events` |
| **Search** | Natural-language search over events/observations/embeddings | `GET /search?q=…` (Phase 8 pipeline) |
| **Segments** | Ingested 30-second clips, 5-sec overlap (path + checksum) | `GET /segments` |
| **Cameras** | Store camera fleet | `GET /cameras` |
| **Models** | Active profile, model registry, recent runs | `GET /api/model-registry*`, `/api/model-runs` |
| **Observability** | Embedded Grafana dashboards | Grafana `:3000` |

The console degrades gracefully: if a backend isn't running it shows an inline
"service offline — start it" state instead of breaking.

## ③ Querying results without the UI

- **Interactive API console (Swagger):** `http://localhost:8000/docs` — try every
  endpoint in the browser.
- **Natural-language search:** `make search QUERY="person leaving without paying"`
  or `curl 'localhost:8000/search?q=tag+swap+at+a+shelf&k=10'`.
- **Raw reads:** `curl localhost:8000/incidents`, `/events`, `/segments?camera_id=…`.

## ④ Auth

The API is **open in dev**. Set `PLATFORM_API_KEY` to enforce auth + RBAC; the
console forwards `Authorization` / `X-API-Key` upstream, so a key set on the API is
honored through the console too.

## Ports at a glance

| Port | Service | Start |
|---|---|---|
| **8080** | **Console (front door)** | `make console` |
| 8000 | Platform API (+ `/docs`, `/search`) | `make api` |
| 8010 | Review service | `make run-review-ui` |
| 3000 | Grafana | `make up` / observability stack |

Override upstream locations for the console with `CONSOLE_API_URL`,
`CONSOLE_REVIEW_URL`, `CONSOLE_GRAFANA_URL`.
