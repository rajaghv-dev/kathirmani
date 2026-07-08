# Kathirmani — Video-Intelligence Platform

An NVIDIA-model, OSS-ingestion video-intelligence **platform** over the 5 Kathirmani
store cameras. Footage is segmented into durable 10-sec clips → 5-sec AI windows →
a Postgres-backed queue → plugin-host AI workers (detection → deterministic rule
engine → VLM verification) → evidence package → human approve/reject → natural-language
search — all observable in Grafana. Models are config-driven and NVIDIA-only by default
(spec/11); the original Marlin/Qwen cascade is kept as a comparison baseline.

## Start here — one command, one URL

```bash
make platform                 # brings up Postgres → migrate+seed → API → review → Console
open http://localhost:8080    # the Console — the single front door
```

`http://localhost:8080` is the **Console**: a unified UI (Overview · Incidents ·
Events · Search · Segments · Cameras · Models · Observability) that proxies every
backend behind one origin. Full guide: **[PLATFORM.md](PLATFORM.md)**. Repo map:
**[spec/09-repo-structure.md](spec/09-repo-structure.md)**.

Run the tiers individually instead (dev):

```bash
make up               # base stack: Postgres (+ observability)        [needs docker]
make migrate && make seed
make api              # platform read/query API + Swagger /docs        → :8000
make run-review-ui    # human approve/reject service                   → :8010
make console          # the Console (SPA + gateway)                    → :8080
make run-workers      # cv + rule_engine + vlm workers (optional)
```

| Port | Service | Start |
|---|---|---|
| **8080** | **Console (front door)** | `make console` / `make platform` |
| 8000 | Platform API (+ `/docs`, `/search`) | `make api` |
| 8010 | Review service | `make run-review-ui` |
| 3000 | Grafana | `make up` |

## How to query results

- **Console** (`:8080`) — Search page for natural-language queries; Incidents/Events/Segments tables.
- **Swagger** — `http://localhost:8000/docs`, an interactive console for every API endpoint.
- **NL search** — `make search QUERY="person leaving without paying"` or `curl 'localhost:8000/search?q=tag+swap&k=10'`.
- **Raw reads** — `curl localhost:8000/{cameras,events,incidents,segments}`.

## Documentation

Full design docs live under [`spec/`](spec/) (start at [spec/README.md](spec/README.md));
remaining work is tracked in [spec/10-platform-roadmap.md](spec/10-platform-roadmap.md)
("What's to be done"). AI-agent session notes + durable memory live in
[`agents-memory/`](agents-memory/).

---

## Legacy research cascade (Marlin-2B)

The platform grew out of an offline research cascade that runs
[NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B) (a 2B-param video
VLM) + Qwen2.5-VL across the 5 cameras to produce dense captions, timestamps, and event
detections in `results/*.json`. That code lives in the **`src/kathirmani`** package and is
preserved as the non-default `research_qwen_baseline` comparison arm (spec/11). The
original working notes are kept verbatim in [`learning.md`](learning.md). It still runs
unchanged via thin root shims (auto-detects CUDA / Apple-MPS / CPU; override with
`KATHIRMANI_DEVICE`):

```bash
bash setup.sh && source .venv/bin/activate
python download_model.py --model all          # offline thereafter
bash run.sh                                    # == python run_inference.py (all cameras)
streamlit run viz_app.py                       # the legacy viewer
```

Stack: PyTorch (CUDA on DGX GB10 / Metal-MPS on Mac Studio / CPU), `prometheus_client`
→ Grafana, Netdata + nvidia-smi/NVML for system + GPU telemetry. See
[spec/06-hardware-portability.md](spec/06-hardware-portability.md) and
[spec/09-repo-structure.md](spec/09-repo-structure.md).
