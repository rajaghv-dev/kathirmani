# 06 · Hardware Portability — one repo, two machines

This pipeline is deployed on **two very different boxes**, and is developed on a
third kind:

| Box | Accelerator | torch device | dtype | `device_map` | `torch.compile` |
|---|---|---|---|---|---|
| **NVIDIA DGX** (GB10) | CUDA GPU | `cuda` | `bfloat16`¹ | `{"": "cuda"}` | on |
| **Apple Mac Studio** | Apple Silicon / Metal | `mps` | `float16` | none (load → `.to`) | off² |
| Dev / CI box | none | `cpu` | `float32` | none (load → `.to`) | off |

¹ `bfloat16` only when the GPU reports `torch.cuda.is_bf16_supported()`, else `float16`.
² `torch.compile` is unreliable on the MPS backend, so it is skipped there.

The code does **not** hardcode any of this. The accelerator is detected at
**runtime**, so the same checkout runs unchanged on all three. Nothing in the
pipeline assumes CUDA anymore.

## The detection layer — `marlin.device`

`src/marlin/device.py` is the single source of truth. It splits into:

- **`select_kind(prefer, has_cuda, has_mps, system, arch, cuda_name)`** — a
  *pure* function (no torch import) that returns `(kind, machine_label,
  warnings)`. Auto-detection order is **CUDA → MPS → CPU**. Because it is pure,
  it is fully unit-tested without torch or a GPU (see `tests/test_device.py`).
- **`detect_device(prefer=None, quiet=False)`** — imports torch lazily, asks it
  what is actually available, calls `select_kind`, then attaches the matching
  `dtype`, `device_map`, and `supports_compile` and returns a frozen
  `DeviceConfig`.

```python
from marlin.device import detect_device
cfg = detect_device()
print(cfg.describe())
# [device] device=cuda, machine=NVIDIA GPU / CUDA (NVIDIA GB10), dtype=bfloat16, compile=on
```

### Overriding the choice

Set the **`MARLIN_DEVICE`** environment variable (or pass `prefer=`) to one of
`cuda` / `mps` / `cpu` / `auto`. If the requested device isn't available the
code prints a warning and falls back to auto-detection — it never hard-fails on
a wrong override.

```bash
MARLIN_DEVICE=cpu bash run.sh --duration 10   # force CPU for a quick smoke test
```

## How the loaders use it

Both model loaders go through `detect_device()` and branch on `device_map`:

- **CUDA**: pass `device_map={"": "cuda"}` to `from_pretrained` — accelerate
  places the weights.
- **MPS / CPU**: load without `device_map`, then `model.to(cfg.torch_device)`.

`marlin/pipeline.py::load_model` (Marlin-2B) and
`marlin/qwen_vl.py::load_qwen_model` (Qwen2.5-VL) both follow this pattern.
Per-call tensor placement uses **`.to(model.device)`** (not a literal `"cuda"`),
so inputs always land wherever the model actually is.

## Metrics are portable too

`marlin/metrics.py` polls hardware telemetry. The NVIDIA-specific sources are
now guarded:

- **`nvidia-smi` power** and **NVML compute-util** are probed once via
  `shutil.which("nvidia-smi")`. Absent (Mac Studio / dev box) → those gauges
  stay at `0` and no failing subprocess is spawned each interval.
- **Thermal zones** (`/sys/class/thermal`) and **`/proc`** CPU/mem reads are
  Linux-only and already degrade to `None`/defaults elsewhere; Netdata (if
  reachable) is the cross-platform source for CPU/mem/disk.

The poll loop logs the detected device on startup so an operator can tell at a
glance whether the GPU gauges will carry data on this box.

## FFmpeg preload is Linux-only

`marlin/ffmpeg_preload.py::preload_av_ffmpeg` dlopens PyAV's bundled FFmpeg
`.so`s with `RTLD_GLOBAL` so torchcodec resolves them. It is a **no-op on
macOS**, where torchcodec ships and resolves its own dylibs. The soname version
is globbed (`libavcodec.so*`) rather than pinned, and the Python version is
derived from `sys.version_info` rather than hardcoded to 3.12.

## Shell entry points

- **`hostname -I`** (Linux-only) in `start_stack.sh` now falls back to macOS
  `ipconfig getifaddr en0/en1` and finally `localhost`.
- **`setup.sh`** detects the box (nvidia-smi → CUDA wheels; Darwin/arm64 → MPS
  wheels; else CPU wheels) and installs the matching PyTorch build.
- **`run.sh`** activates the venv and launches the pipeline; the device is then
  auto-selected at runtime and printed.

## Where each former CUDA assumption lived (before → after)

| Was | Now |
|---|---|
| `pipeline.py` `device_map={"": "cuda"}`, `dtype=torch.bfloat16` | `detect_device()` config + load/`.to` branch |
| `qwen_vl.py` `device_map={"": "cuda"}`, `.to("cuda")` | `detect_device()` config + `.to(model.device)` |
| `metrics.py` unconditional `nvidia-smi` / NVML | guarded by `shutil.which("nvidia-smi")` |
| `start_stack.sh` `hostname -I` | portable `host_ip()` fallback |
| `run_inference.py` hardcoded `python3.12` av.libs preload | `marlin.ffmpeg_preload`, version-globbed, Linux-guarded |

See also: [09-repo-structure.md](09-repo-structure.md) for the package layout.

## Related

[04-observability-stack.md](04-observability-stack.md) · [05-performance-and-optimizations.md](05-performance-and-optimizations.md) · [09-repo-structure.md](09-repo-structure.md)
