#!/usr/bin/env bash
# GPU: register the Docker NVIDIA runtime (needs sudo).
set -euo pipefail; cd "$(dirname "$0")/../.."
bash scripts/setup_nvidia_docker.sh
echo "setup/gpu: OK"
