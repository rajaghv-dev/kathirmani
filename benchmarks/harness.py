"""Benchmark harness — time a pluggable runner over N synthetic items.

Core idea: a benchmark drives a *runner* callable over a list of synthetic
items, timing each item end-to-end (and optionally per-stage). From the timings
it computes the Phase 11 throughput/latency metrics:

  - p50 / p95 / mean latency (ms)
  - clips_per_min, video_sec_per_sec (throughput vs wall time)
  - tokens_per_sec, ttft_ms (VLM tasks, if the runner reports them)
  - + carries avg_power_w / gpu so tco.py can turn it into ₹.

The runner is pluggable so the same harness can drive a real worker/plugin OR
the default **fake timing model** (`FakeRunner`) which needs no GPU/DB/weights.
Numbers from the fake runner are SYNTHETIC until a real runner is run on a GPU.

A "runner" is any callable: runner(item: dict) -> RunnerResult, where
RunnerResult carries latency_ms and optional stage_ms / tokens / ttft_ms /
parse_success / power_w. The harness never imports numpy at module scope for the
percentile math — it uses a tiny pure-python percentile so tests stay light, but
falls back to numpy if present for parity with the rest of the repo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


# ---- Result containers ------------------------------------------------------
@dataclass
class RunnerResult:
    """What a single runner invocation reports back to the harness."""
    latency_ms: float                       # end-to-end latency for this item
    stage_ms: dict[str, float] = field(default_factory=dict)  # per-stage split
    output_tokens: int = 0                  # VLM decode tokens (0 if N/A)
    ttft_ms: float | None = None            # time-to-first-token (VLM)
    tokens_per_sec: float | None = None     # decode throughput (VLM)
    parse_success: bool | None = None       # JSON parse outcome (VLM)
    video_sec: float = 0.0                  # video-seconds this item represents
    power_w: float | None = None            # instantaneous GPU power sample (W)


@dataclass
class BenchmarkResult:
    """Aggregated metrics for a whole benchmark run."""
    name: str
    task: str
    n_items: int
    wall_time_sec: float
    # latency
    p50_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    # throughput
    clips_per_min: float
    video_sec_per_sec: float
    # vlm (None when task has no tokens)
    ttft_ms_p95: float | None
    tokens_per_sec: float | None
    parse_success_rate: float | None
    # resource / quality (carried through for TCO + the DB row)
    avg_power_w: float
    gpu_name: str | None
    total_video_sec: float
    total_output_tokens: int
    stage_ms_mean: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "n_items": self.n_items,
            "wall_time_sec": self.wall_time_sec,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "clips_per_min": self.clips_per_min,
            "video_sec_per_sec": self.video_sec_per_sec,
            "ttft_ms_p95": self.ttft_ms_p95,
            "tokens_per_sec": self.tokens_per_sec,
            "parse_success_rate": self.parse_success_rate,
            "avg_power_w": self.avg_power_w,
            "gpu_name": self.gpu_name,
            "total_video_sec": self.total_video_sec,
            "total_output_tokens": self.total_output_tokens,
            "stage_ms_mean": self.stage_ms_mean,
            "extra": self.extra,
        }


# ---- Percentile (pure-python, numpy-parity 'linear' interpolation) ----------
def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile (matches numpy.percentile default).

    Kept dependency-free so the math is auditable in tests; numpy is used if
    importable purely as a cross-check, but this implementation is canonical.
    """
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (pct / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


# ---- The fake timing model (default runner; no GPU) -------------------------
class FakeRunner:
    """Deterministic synthetic timing model — drives the harness without a GPU.

    Per item it returns a latency derived from the item's `base_latency_ms` and a
    deterministic per-index jitter (no RNG → reproducible tests). It mimics a
    detector or VLM depending on `task`. ALL numbers it produces are synthetic
    placeholders for what the real GPU run will measure.
    """

    def __init__(
        self,
        *,
        task: str = "detection",
        base_latency_ms: float = 40.0,
        jitter_ms: float = 10.0,
        power_w: float = 60.0,
        tokens_per_item: int = 0,
        ttft_ms: float = 0.0,
        stages: Sequence[str] = ("preprocess", "infer", "postprocess"),
        sleep: bool = False,
    ) -> None:
        self.task = task
        self.base_latency_ms = base_latency_ms
        self.jitter_ms = jitter_ms
        self.power_w = power_w
        self.tokens_per_item = tokens_per_item
        self.ttft_ms = ttft_ms
        self.stages = tuple(stages)
        self.sleep = sleep  # if True, actually sleep (real wall time) — off in tests

    def __call__(self, item: dict[str, Any]) -> RunnerResult:
        idx = int(item.get("index", 0))
        n = max(int(item.get("n", 1)), 1)
        # Triangular, deterministic jitter in [-jitter, +jitter] across the run.
        frac = (idx % n) / n if n > 1 else 0.0
        latency = self.base_latency_ms + self.jitter_ms * (2.0 * frac - 1.0)
        # Split latency across declared stages (infer gets the lion's share).
        weights = {s: (3.0 if s == "infer" else 1.0) for s in self.stages}
        tot_w = sum(weights.values()) or 1.0
        stage_ms = {s: latency * w / tot_w for s, w in weights.items()}
        if self.sleep:
            time.sleep(latency / 1000.0)
        tps = None
        if self.tokens_per_item:
            decode_ms = max(latency - self.ttft_ms, 1.0)
            tps = self.tokens_per_item / (decode_ms / 1000.0)
        return RunnerResult(
            latency_ms=latency,
            stage_ms=stage_ms,
            output_tokens=self.tokens_per_item,
            ttft_ms=self.ttft_ms if self.tokens_per_item else None,
            tokens_per_sec=tps,
            parse_success=True if self.tokens_per_item else None,
            video_sec=float(item.get("video_sec", 0.0)),
            power_w=self.power_w,
        )


# ---- The harness ------------------------------------------------------------
def run_benchmark(
    name: str,
    task: str,
    items: Sequence[dict[str, Any]],
    runner: Callable[[dict[str, Any]], RunnerResult],
    *,
    gpu_name: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Drive `runner` over `items`, aggregate Phase 11 metrics.

    `clock` is injectable so tests can supply a deterministic wall clock instead
    of relying on real elapsed time.
    """
    latencies: list[float] = []
    ttfts: list[float] = []
    tps_vals: list[float] = []
    parse_ok = 0
    parse_total = 0
    power_samples: list[float] = []
    total_video_sec = 0.0
    total_tokens = 0
    stage_accum: dict[str, float] = {}
    stage_count: dict[str, int] = {}

    t0 = clock()
    for item in items:
        r = runner(item)
        latencies.append(r.latency_ms)
        if r.ttft_ms is not None:
            ttfts.append(r.ttft_ms)
        if r.tokens_per_sec is not None:
            tps_vals.append(r.tokens_per_sec)
        if r.parse_success is not None:
            parse_total += 1
            parse_ok += 1 if r.parse_success else 0
        if r.power_w is not None:
            power_samples.append(r.power_w)
        total_video_sec += r.video_sec
        total_tokens += r.output_tokens
        for s, ms in r.stage_ms.items():
            stage_accum[s] = stage_accum.get(s, 0.0) + ms
            stage_count[s] = stage_count.get(s, 0) + 1
    wall = clock() - t0

    # Throughput basis: a real (GPU) runner blocks, so wall time reflects the
    # work and dominates. A fake runner returns instantly, so the harness's own
    # near-zero wall time would inflate throughput — fall back to the summed
    # modeled latency. We take max(wall, summed-latency) so fake mode reports the
    # *modeled* throughput while real mode reports the *measured* one. (Serial
    # model: parallel/batched throughput needs a concurrency-aware runner.)
    summed_latency_sec = sum(latencies) / 1000.0
    effective_sec = max(wall, summed_latency_sec, 1e-9)

    n = len(latencies)
    clips_per_min = (n / effective_sec) * 60.0
    video_sec_per_sec = total_video_sec / effective_sec
    avg_power = sum(power_samples) / len(power_samples) if power_samples else 0.0
    tps = sum(tps_vals) / len(tps_vals) if tps_vals else None
    parse_rate = (parse_ok / parse_total) if parse_total else None
    stage_mean = {s: stage_accum[s] / stage_count[s] for s in stage_accum}

    return BenchmarkResult(
        name=name,
        task=task,
        n_items=n,
        wall_time_sec=effective_sec,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        mean_latency_ms=(sum(latencies) / n) if n else 0.0,
        clips_per_min=clips_per_min,
        video_sec_per_sec=video_sec_per_sec,
        ttft_ms_p95=percentile(ttfts, 95) if ttfts else None,
        tokens_per_sec=tps,
        parse_success_rate=parse_rate,
        avg_power_w=avg_power,
        gpu_name=gpu_name,
        total_video_sec=total_video_sec,
        total_output_tokens=total_tokens,
        stage_ms_mean=stage_mean,
    )
