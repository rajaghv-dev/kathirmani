# 14 · Console & One-Command Deployment

The platform is several cooperating services (read API, review service, Grafana,
search). Before this, an operator had to know each port and start each process — there
was **no single entry point**. The **Console** is that front door, and `make platform`
brings the whole stack up behind it with one command.

## The Console — a BFF/gateway (`services/console/`, :8080)

`http://localhost:8080` serves one cohesive single-page app **and proxies the
upstreams server-side**, so the browser talks to a single origin (no CORS, auth header
passthrough, one URL to share).

```
browser ──▶ console :8080 ──▶ /backend/*  ─▶ api :8000        (read/query + /search)
                          └─▶ /review/*   ─▶ review-ui :8010  (incident detail + approve/reject)
            (iframe)      └─▶ Grafana :3000 embedded for Observability
```

Routes (`services/console/app.py`):

| Route | Purpose |
|---|---|
| `GET /` , `/static/*` | the SPA shell (`static/index.html`) + assets |
| `GET /config` | upstream URLs the SPA needs (Grafana embed) |
| `GET /healthz` | console + best-effort upstream reachability (drives the status pill) |
| `* /backend/{path}` | proxied to the platform API (`CONSOLE_API_URL`) |
| `* /review/{path}` | proxied to the review service (`CONSOLE_REVIEW_URL`) |

SPA sections: **Overview** (KPIs + recent events), **Incidents** (queue → drawer →
approve/reject), **Events**, **Search** (NL → ranked hits), **Segments**, **Cameras**,
**Models** (active profile / registry / recent runs), **Observability** (embedded
Grafana). Every panel **degrades to an inline "offline" state** when its upstream is
down, so the console always loads. A dead upstream surfaces as a structured `502`
(`{"error":"upstream_unreachable"}`), never an unhandled 500.

The `/search` capability is an addition to the API (`GET /search?q=&k=`) that wraps the
embedding-worker `query_run` and never 500s — so search is HTTP-queryable and proxied
into the console.

## One-command launch

- **`Dockerfile.app`** — a deliberately **torch-free** image for the read/query/console
  tier (`fastapi`, `uvicorn`, `httpx`, `psycopg[binary]`, `pyyaml`, `numpy`). The repo
  is bind-mounted at `/app`, so code edits need no rebuild — only a dependency change does.
- **`docker-compose.platform.yml`** — layers the app tier on the base stack:
  - `migrate` (one-shot): `db_migrate.py up && db_seed.py` once Postgres is healthy, then exits.
  - `api` → `review-ui` → `console`, started in dependency order (console waits for api healthy).
  - All share the compose network; the console reaches `api`/`review-ui` by service name.
- **`make platform`** = `docker compose -f docker-compose.yml -f
  docker-compose.observability.yml -f docker-compose.platform.yml up -d --build`, then
  prints the URL map. `make platform-down` / `make platform-logs` to stop / tail.

### Ports & env

| Port | Service | | Env override | Default |
|---|---|---|---|---|
| 8080 | console | | `CONSOLE_API_URL` | `http://api:8000` |
| 8000 | api (+ `/docs`, `/search`) | | `CONSOLE_REVIEW_URL` | `http://review-ui:8010` |
| 8010 | review-ui | | `CONSOLE_GRAFANA_URL` | `http://localhost:3000` (browser-facing) |
| 3000 | grafana | | `PLATFORM_API_KEY` | unset → API open in dev |

## Auth

The API is **open in dev** (no `PLATFORM_API_KEY`). Set it to enforce auth + RBAC; the
console forwards `Authorization` / `X-API-Key` upstream, so a key set on the API is
honored through the console too.

## Verification (done 2026-06-10)

`make platform`'s subset (base + app tier) was brought up live on the dev box: Postgres
healthy → `migrate` applied **all four migrations (incl. the new 0003 GRANTs + 0004
tstzrange/btree_gist) exit 0** and seeded 5 cameras / 6 zones / model registry → api
healthy → console serving. `GET /backend/cameras` returned the real seeded Kathirmani
cameras through console→api→Postgres; `/healthz` reported api+review healthy. The
console's own suite (`services/console/tests/`) covers SPA serving, JSON relay, and the
502-on-dead-upstream path. (Grafana skipped in that run to avoid the box's existing :3000.)

## Related

[09-repo-structure.md](09-repo-structure.md) · [10-platform-roadmap.md](10-platform-roadmap.md) ·
[04-observability-stack.md](04-observability-stack.md) · [12-frameworks.md](12-frameworks.md)
