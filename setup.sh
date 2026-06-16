#!/usr/bin/env bash
# Machine-aware environment setup.
#
# Detects the box and installs the matching PyTorch build:
#   - NVIDIA GPU present (DGX)        -> default CUDA wheels
#   - Apple Silicon macOS (Mac Studio)-> default wheels (Metal/MPS capable)
#   - anything else                  -> CPU-only wheels
#
# Usage:  bash setup.sh            (creates .venv and installs deps)
#         PYTHON=python3.12 bash setup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"

uname_s="$(uname -s)"
uname_m="$(uname -m)"

if command -v nvidia-smi >/dev/null 2>&1; then
    PLATFORM="cuda"
    LABEL="NVIDIA GPU / CUDA (e.g. DGX)"
elif [ "$uname_s" = "Darwin" ] && [ "$uname_m" = "arm64" ]; then
    PLATFORM="mps"
    LABEL="Apple Silicon / Metal MPS (e.g. Mac Studio)"
else
    PLATFORM="cpu"
    LABEL="CPU ($uname_s/$uname_m)"
fi
echo "[setup] Detected: $LABEL  ->  platform=$PLATFORM"

if [ ! -d "$ROOT/.venv" ]; then
    echo "[setup] Creating venv at $ROOT/.venv"
    "$PY" -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -U pip

# Install the right torch first so the requirements.txt torch constraint is
# already satisfied with the correct backend.
case "$PLATFORM" in
    cuda) python -m pip install torch ;;   # default Linux wheels are CUDA
    mps)  python -m pip install torch ;;   # default macOS arm64 wheels are MPS-capable
    cpu)  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu ;;
esac

python -m pip install -r "$ROOT/requirements.txt"

echo ""
echo "[setup] Done ($LABEL)."
echo "[setup] The pipeline auto-detects the accelerator at runtime (CUDA -> MPS -> CPU);"
echo "[setup] override with KATHIRMANI_DEVICE=cuda|mps|cpu if needed."
echo "[setup] Next:  source .venv/bin/activate  &&  bash run.sh   (or: python download_model.py)"
