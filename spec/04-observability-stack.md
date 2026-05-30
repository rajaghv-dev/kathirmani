# 04 — Observability Stack

## Runtime metrics come from Netdata

CPU / memory / disk are read from the **Netdata REST API** (`_netdata_dims()` in `inference/metrics.py`, `NETDATA_URL` default `http://localhost:19999`, override to `http://netdata:19999` inside Docker).

Charts used:

| Chart | What it provides |
|---|---|
| `system.cpu` | user / system % |
| `system.ram` | MiB → GB |
| `disk_space./` | GiB used / avail |
| `system.io` | reads / writes KiB/s |

Falls back to `/proc` if Netdata is unreachable.

## GPU metrics

- **GPU power** still comes from `nvidia-smi`.
- **Compute utilization** comes from NVML.

## Why the model "downloads repeatedly" (it doesn't) + caching

All `from_pretrained` calls use `local_files_only=True` and `HF_HUB_OFFLINE=1`, and `models/Marlin-2B/` is complete — so **weights are never re-downloaded**.

What *did* regenerate every run:

- the `trust_remote_code` module cache (`transformers_modules`), and
- `torch.compile` artifacts (recompiled from scratch each process, ~3 min).

On an ephemeral `$HOME` these vanish between runs and look like "downloading again". Fixed by pinning both into the repo (`run_inference.py`):

```bash
HF_HOME=models/.cache/hf
TORCHINDUCTOR_CACHE_DIR=models/.cache/inductor
TORCHINDUCTOR_FX_GRAPH_CACHE=1
```

(All gitignored.) First run still compiles once; every run after reuses the cache.

## Related

[02-architecture.md](02-architecture.md) · [06-hardware-portability.md](06-hardware-portability.md) · [08-dashboards.md](08-dashboards.md)
