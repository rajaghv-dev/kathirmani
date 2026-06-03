#!/usr/bin/env bash
# Register the NVIDIA runtime with the Docker daemon so GPU containers
# (ollama, dcgm-exporter, future NIM/Triton/DeepStream) can use the GPU.
# Idempotent. Needs sudo. The NVIDIA Container Toolkit (nvidia-ctk) must be
# installed already — it is on this box (v1.18).
set -euo pipefail

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "ERROR: nvidia-ctk not found. Install the NVIDIA Container Toolkit first." >&2
  exit 1
fi

if docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia'; then
  echo "NVIDIA Docker runtime already registered. Nothing to do."
  exit 0
fi

echo "Registering the NVIDIA runtime with Docker (sudo)..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "Verifying..."
docker info 2>/dev/null | grep -iE 'Runtimes' || true
echo "Done. Test with:  docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi"
