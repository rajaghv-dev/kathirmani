# 09 · Repository Structure — the `src/` package refactor

The Python code lives in an installable **`marlin`** package under `src/`. The
old flat layout (`run_inference.py` + an `inference/` package at the repo root)
was reorganized; thin root **shims** preserve every existing command.

## Layout

```
kathirmani-marlin/
├── run_inference.py        # shim → marlin.cli.run_inference:main
├── download_model.py       # shim → marlin.cli.download_model:main
├── serve_metrics.py        # shim → marlin.serve_metrics:main
├── viz_app.py              # shim → marlin.viz.app:main (streamlit entry)
├── setup.sh                # machine-aware venv + torch install
├── run.sh                  # machine-aware run wrapper
├── start_stack.sh          # observability docker stack (portable host-IP)
├── cleanup.sh
├── pyproject.toml          # declares the marlin package + console scripts
├── requirements.txt
├── learning.md             # original working notes (kept verbatim)
│
├── src/marlin/
│   ├── __init__.py         # PROJECT_ROOT / MODELS_DIR / RESULTS_DIR anchors
│   ├── device.py           # accelerator detection  (see 06-hardware-portability)
│   ├── ffmpeg_preload.py   # portable PyAV/FFmpeg RTLD_GLOBAL preload
│   ├── pipeline.py         # Marlin-2B inference  (was inference/pipeline.py)
│   ├── qwen_vl.py          # Qwen2.5-VL spatial analysis
│   ├── metrics.py          # Prometheus gauges + hardware polling
│   ├── loki.py             # Loki push client
│   ├── annotations.py      # Grafana annotations
│   ├── serve_metrics.py    # standalone Prometheus exporter (was root)
│   ├── cli/
│   │   ├── run_inference.py
│   │   └── download_model.py
│   └── viz/
│       └── app.py          # Streamlit viewer (was viz_app.py)
│
├── spec/                   # this documentation set
├── agents-memory/          # AI-agent session notes + durable memory
├── tests/
├── results/                # inference JSONs (git-tracked, on-disk source of truth)
├── grafana/  prometheus/   # dashboards + scrape config
```

> `inference/locate.py` is referenced by `run_inference.py --locate` and the
> Streamlit "Backend compare" tab but is **not present** in the repo — the
> locate stage is an opt-in stub. Both call sites import it lazily inside
> `try/except`, so its absence is harmless. After the refactor the import path
> is `marlin.locate` (still optional).

## Path anchoring

Videos, `models/`, and `results/` live at the **repo root**, not next to the
moved code. `marlin/__init__.py` exposes the anchors so nothing depends on the
current working directory:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/marlin → repo root
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
```

`serve_metrics.py` re-derives the root locally (`parents[2]`) without importing
`marlin`, so the slim Docker container in `start_stack.sh` (which only
`pip install prometheus_client`) can still run it.

## How the commands keep working

Each root shim prepends `src/` to `sys.path` and calls the package entry point,
so all documented commands are unchanged:

```bash
python run_inference.py --video "Bill Counter" --no-compile
python download_model.py --model all
python serve_metrics.py --port 8900
streamlit run viz_app.py
bash start_stack.sh
```

The CLI modules also self-bootstrap `src/` onto `sys.path`, so `python -m
marlin.cli.run_inference` works too. Streamlit re-executes `viz_app.py` on every
interaction, so that shim calls `main()` directly rather than relying on an
`__main__` guard.

### Optional: install as a package

```bash
pip install -e .            # then use the console scripts:
marlin-infer  --video "Bill Counter"
marlin-download --model all
marlin-metrics --port 8900
```

## Tests

`tests/conftest.py` puts `src/` on `sys.path`, so tests import `marlin.*`
without an editable install.

- `tests/test_device.py` — accelerator-selection logic (no torch needed).
- `tests/test_structure.py` — package layout, shims, no-hardcoded-CUDA guard,
  all sources parse.
- `tests/test_setup.py` — result-schema + integration checks; the torch/model/
  server-dependent ones **skip** cleanly on a bare dev box and assert fully on a
  provisioned DGX / Mac Studio venv.

See also: [06-hardware-portability.md](06-hardware-portability.md).

## Related

[06-hardware-portability.md](06-hardware-portability.md) · [07-runbook.md](07-runbook.md)
