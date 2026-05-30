#!/usr/bin/env bash
# Machine-aware run wrapper. Activates the venv (if present) and launches the
# pipeline; marlin.device picks the accelerator (CUDA -> MPS -> CPU) at runtime
# and prints what it chose. All run_inference.py flags pass straight through.
#
# Usage:  bash run.sh                                  # all cameras
#         bash run.sh --video "Bill Counter" --no-compile
#         MARLIN_DEVICE=cpu bash run.sh --duration 10  # force a device
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -d "$ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
fi
exec python "$ROOT/run_inference.py" "$@"
