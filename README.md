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

## Quick start

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Run inference (downloads model ~5GB on first run)
python run_inference.py

# 3. Watch the dashboard
open http://localhost:3000   # Grafana — login: admin / admin
# Navigate: Dashboards > Marlin Inference > Marlin-2B Inference — Kathirmani Store
```

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
- **Runtime**: PyTorch 2.12 + CUDA 13.0 on NVIDIA GB10
- **Metrics**: `prometheus_client` → Grafana 12.4.1
- **System monitoring**: Netdata v2.9.0 (Prometheus endpoint)
