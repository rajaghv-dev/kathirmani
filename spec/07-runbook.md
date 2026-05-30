# 07 — Runbook (Running & Viewing)

> **NOTE:** after the src/ refactor the entry-point commands are unchanged (`python run_inference.py` / `python serve_metrics.py` / `streamlit run viz_app.py` still work via thin root shims). See spec/09-repo-structure.md.

This runbook reflects decode-once, parallel cameras (`--max-workers`), the opt-in `--locate` YOLOE gate, and the Streamlit Optimizations tab.

## Setup (per shell)

```bash
cd <repo>
source .venv/bin/activate
docker compose up -d      # Prometheus, Loki, Grafana(:3000), Netdata(:19999)
```

## Run inference

```bash
python run_inference.py                                   # all 5 cameras, Marlin only, parallel
python run_inference.py --locate                          # + YOLOE Locate gate → Qwen only on routed frames
python run_inference.py --video "Bill Counter" --duration 12 --no-compile --locate   # fast smoke test
python run_inference.py --max-workers 3                   # cap parallel camera workers (default: one/video)
```

### Flags

| Flag | Meaning |
|---|---|
| `--video` | Run a single named camera. Uses `process_video` (sequential). |
| `--duration N` | Limit clip to N seconds. |
| `--no-compile` | Skip the ~3-min torch.compile step. |
| `--max-workers N` | Cap parallel camera workers (default: one per video). |
| `--locate` | Opt-in Locate stage; default backend `yoloe`. |
| `--locate-backend` | Choose the Locate backend (default `yoloe`). |
| `--skip-qwen` | Skip the Qwen cascade. |
| `--keep-alive` | Keep the process alive after the run. |

A plain `python run_inference.py` runs **Marlin only** — no Locate, no Qwen cascade gating. The multi-camera path uses `run_all` (parallel); a single `--video` uses `process_video` (sequential). Locate + Qwen run **after** Marlin, looping over results in `main()`.

## Outputs (`results/`)

- `<camera>.json` — caption, events, `find_results` (+ `locate_analysis` / `qwen_analysis` when those stages ran)
- `fused.json` — best-camera find spans + deduped store events
- `summary.json`
- `economy.json`

## View — Grafana

`http://<host>:3000`, login `admin` / `admin`; URLs are printed at run end.

Dashboards: Model & Runtime · Pipeline · Camera Views · Store (Fused) · Visual Analysis (Qwen) · Inference Logs (Loki) · Compute Economy. (See [08-dashboards.md](08-dashboards.md).)

## View — Streamlit

Visualization ONLY; annotation is CVAT/Label Studio.

```bash
streamlit run viz_app.py        # http://localhost:8501
```

Tabs: Overview · Detections · Timeline · Qwen Q&A · Store (fused) · Backend compare · **Optimizations** (decode-once win + Tier 1/2/3 roadmap; renders even with no results on disk).

## Related

[01-overview.md](01-overview.md) · [06-hardware-portability.md](06-hardware-portability.md) · [08-dashboards.md](08-dashboards.md) · [09-repo-structure.md](09-repo-structure.md)
