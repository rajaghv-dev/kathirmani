# Kathirmani Store — Marlin-2B Video Intelligence

Runs [NemoStation/Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B), a 2B-parameter video VLM, across 5 security camera feeds to extract dense captions, timestamps, and event detections. Metrics are exposed to Grafana via Prometheus.

## Cameras

| File | Location |
|---|---|
| Kathirmani Bill Counter.mkv | Cash register area |
| Kathirmani Center Camera.mkv | Center of store |
| Kathirmani Left Side Camera.mkv | Left aisle |
| Kathirmani Near Entry Door Camera.mkv | Entrance |
| Kathirmani Right Side Cam.mkv | Right aisle |

## What the model extracts

**Caption mode** (per video):
- Full scene description
- Dense event list with second-precise timestamps

**Find mode** (12 queries per video):
- Customer enters / leaves store
- Cash payment transaction at counter
- Person picks up items from shelf
- Employee interacts with customer
- Person using mobile phone
- Group of people gathered
- Person walking through aisle
- Suspicious or unusual behavior
- Person standing near entry door
- Crowded area with multiple customers
- Person carrying a bag

Results are saved to `results/<camera_name>.json` + `results/summary.json`.

## Runs on two machines — auto-detected

This repo is deployed on an **NVIDIA DGX** (CUDA) and an **Apple Mac Studio**
(Apple Silicon / Metal-MPS), and falls back to **CPU** on dev/CI boxes. The
accelerator is detected at **runtime** — the same checkout runs unchanged on all
three. Override with `MARLIN_DEVICE=cuda|mps|cpu`. See
[spec/06-hardware-portability.md](spec/06-hardware-portability.md).

## Quick start

```bash
# 1. Machine-aware setup — detects the box, installs the right PyTorch build
bash setup.sh
source .venv/bin/activate

# 2. Fetch the models (weights are loaded offline thereafter)
python download_model.py --model all

# 3. Run inference (device auto-selected and printed at startup)
bash run.sh                 # all cameras   (== python run_inference.py)
# bash run.sh --video "Bill Counter" --no-compile   # one camera, quick

# 4. Watch the dashboard
open http://localhost:3000   # Grafana — login: admin / admin
# Navigate: Dashboards > Marlin Inference > Marlin-2B Inference — Kathirmani Store
```

> The code lives in the `src/marlin` package; `python run_inference.py`,
> `serve_metrics.py`, `download_model.py`, and `streamlit run viz_app.py` still
> work via thin root shims. See [spec/09-repo-structure.md](spec/09-repo-structure.md).

## Documentation

Full docs are under [`spec/`](spec/) (start at [spec/README.md](spec/README.md)).
The original working notes remain in [`learning.md`](learning.md). AI-agent
session notes and durable memory live in [`agents-memory/`](agents-memory/).

## Dashboard panels

| Panel | Data source |
|---|---|
| Videos Processed / Events Detected / Errors | MarlinMetrics (Prometheus) |
| GPU Memory, Utilization, Temperature | MarlinMetrics (live nvidia-smi polling) |
| Inference Duration by Video & Mode | MarlinMetrics histogram |
| Events per Camera (bar gauge) | MarlinMetrics |
| Find Query Results table | MarlinMetrics |
| System CPU & RAM | Netdata |
| Inference Duration Heatmap | MarlinMetrics |

## Stack

- **Model**: Marlin-2B (Qwen3.5-2B base, fine-tuned for video VLM)
- **Runtime**: PyTorch — CUDA on NVIDIA GB10 (DGX) / Metal-MPS on Apple Mac Studio / CPU fallback, auto-detected
- **Metrics**: `prometheus_client` → Grafana 12.4.1
- **System monitoring**: Netdata v2.9.0 (Prometheus endpoint); GPU power/util via nvidia-smi+NVML on NVIDIA boxes only
