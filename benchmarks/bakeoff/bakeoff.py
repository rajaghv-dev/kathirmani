"""Phase-13 NVIDIA bake-off — same model, many runtimes, ranked + selected.

Builds ON the Phase-11 harness: each candidate runtime (see `runtimes.py`) is run
through `benchmarks.harness.run_benchmark` with its synthetic `FakeProfile` driving
a `FakeRunner`, then costed with `benchmarks.tco`. The result is a ranked table and
an explicit selection of a **production** profile (best balanced score) plus a
**fallback** profile (most-available / most-robust).

Everything here is deterministic and GPU/DB-free. The throughput/latency numbers
are SYNTHETIC (fake timing model) until the runtimes are actually run on the GPU —
the *structure*, *scoring*, and *selection* are real.

Scoring (documented, tunable — see `DEFAULT_WEIGHTS`):
  We min-max normalize each metric across the candidates to 0..1 where 1 = best:
    throughput : clips_per_min        (higher better)
    latency    : p95_latency_ms       (lower  better -> inverted)
    quality    : quality_score        (higher better)
    cost       : cost_per_1000_clips  (lower  better -> inverted)
  balanced_score = Σ weight_k * norm_k   (weights sum to 1).
  Production = highest balanced_score among *available* candidates if any are
  available, else highest overall (so the bake-off still recommends on a bare box).
  Fallback  = the most-available / most-robust candidate that is NOT production:
  prefer truly-available + portable (local/server over container), break ties by
  balanced_score. Fallback guarantees a graceful degrade target for rollback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from benchmarks.harness import FakeRunner, run_benchmark
    from benchmarks import tco
    from benchmarks.bakeoff import runtimes as rt
except ImportError:  # direct-script / path fallback
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmarks.harness import FakeRunner, run_benchmark
    from benchmarks import tco
    from benchmarks.bakeoff import runtimes as rt


# ---- Scoring weights (sum to 1.0; tune to bias the production pick) ----------
DEFAULT_WEIGHTS: dict[str, float] = {
    "throughput": 0.35,   # clips/min — primary capacity lever
    "latency": 0.25,      # p95 — UX / SLA
    "quality": 0.25,      # output quality proxy — don't ship a faster-but-worse model
    "cost": 0.15,         # ₹/1000-clips — TCO
}

# Default workload for the per-runtime synthetic run (mirrors benchmark 5 shape).
DEFAULT_N_ITEMS = 100
DEFAULT_VIDEO_SEC_PER_ITEM = 20.0
DEFAULT_N_CAMERAS = 5
DEFAULT_GPU = "DGX_SPARK_GB10"


@dataclass
class RuntimeResult:
    """Per-runtime bake-off result: harness metrics + TCO + availability + score."""
    runtime: str
    available: bool
    kind: str
    notes: str
    # headline metrics (from harness, SYNTHETIC in fake mode)
    clips_per_min: float
    p95_latency_ms: float
    ttft_ms_p95: float | None
    tokens_per_sec: float | None
    quality_score: float
    avg_power_w: float
    cost_per_1000_clips: float
    tco: dict[str, float]
    # scoring (filled by rank())
    norm: dict[str, float] = field(default_factory=dict)
    balanced_score: float = 0.0
    synthetic: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "available": self.available,
            "kind": self.kind,
            "notes": self.notes,
            "clips_per_min": self.clips_per_min,
            "p95_latency_ms": self.p95_latency_ms,
            "ttft_ms_p95": self.ttft_ms_p95,
            "tokens_per_sec": self.tokens_per_sec,
            "quality_score": self.quality_score,
            "avg_power_w": self.avg_power_w,
            "cost_per_1000_clips_inr": self.cost_per_1000_clips,
            "tco": self.tco,
            "norm": self.norm,
            "balanced_score": self.balanced_score,
            "synthetic": self.synthetic,
        }


def _make_items(n: int, video_sec: float) -> list[dict[str, Any]]:
    return [{"index": i, "n": n, "video_sec": video_sec} for i in range(n)]


def run_one_runtime(
    candidate: "rt.RuntimeCandidate",
    *,
    task: str,
    n_items: int = DEFAULT_N_ITEMS,
    video_sec_per_item: float = DEFAULT_VIDEO_SEC_PER_ITEM,
    n_cameras: int = DEFAULT_N_CAMERAS,
    gpu_name: str = DEFAULT_GPU,
    rate_inr_per_kwh: float = tco.DEFAULT_RATE_INR_PER_KWH,
) -> RuntimeResult:
    """Run a single candidate through the Phase-11 harness (fake) + TCO."""
    fp = candidate.fake
    runner = FakeRunner(
        task=task,
        base_latency_ms=fp.base_latency_ms,
        jitter_ms=fp.jitter_ms,
        power_w=fp.power_w,
        tokens_per_item=fp.tokens_per_item,
        ttft_ms=fp.ttft_ms,
        stages=("decode_clip", "prefill", "decode", "parse"),
    )
    items = _make_items(n_items, video_sec_per_item)
    res = run_benchmark(
        f"bakeoff_{task}_{candidate.name}", task, items, runner, gpu_name=gpu_name)
    cost = tco.tco_report(res.avg_power_w, res.clips_per_min, n_cameras,
                          rate_inr_per_kwh=rate_inr_per_kwh)
    return RuntimeResult(
        runtime=candidate.name,
        available=rt.is_available(candidate),
        kind=candidate.kind,
        notes=candidate.notes,
        clips_per_min=res.clips_per_min,
        p95_latency_ms=res.p95_latency_ms,
        ttft_ms_p95=res.ttft_ms_p95,
        tokens_per_sec=res.tokens_per_sec,
        quality_score=fp.quality_score,
        avg_power_w=res.avg_power_w,
        cost_per_1000_clips=cost["cost_per_1000_clips_inr"],
        tco=cost,
        synthetic=fp.synthetic,
    )


def run_bakeoff(task: str, **kw: Any) -> list[RuntimeResult]:
    """Run every candidate runtime for `task` through the harness (fake mode)."""
    results = [run_one_runtime(c, task=task, **kw) for c in rt.candidates_for(task)]
    return rank(results)


# ---- Scoring + selection ----------------------------------------------------
def _minmax(values: list[float], *, higher_is_better: bool) -> list[float]:
    """Min-max normalize to 0..1 (1 = best). Degenerate spread -> all 1.0."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    span = hi - lo
    if higher_is_better:
        return [(v - lo) / span for v in values]
    return [(hi - v) / span for v in values]  # invert: lower raw -> higher score


def rank(
    results: list[RuntimeResult],
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> list[RuntimeResult]:
    """Normalize metrics, compute balanced_score, return sorted (best first).

    Pure: mutates each result's `norm`/`balanced_score` and returns a new sorted
    list. Sorting is stable on (-balanced_score, runtime) for determinism.
    """
    if not results:
        return []
    tput = _minmax([r.clips_per_min for r in results], higher_is_better=True)
    lat = _minmax([r.p95_latency_ms for r in results], higher_is_better=False)
    qual = _minmax([r.quality_score for r in results], higher_is_better=True)
    cost = _minmax([r.cost_per_1000_clips for r in results], higher_is_better=False)
    for r, t, l, q, c in zip(results, tput, lat, qual, cost):
        r.norm = {"throughput": t, "latency": l, "quality": q, "cost": c}
        r.balanced_score = (
            weights["throughput"] * t
            + weights["latency"] * l
            + weights["quality"] * q
            + weights["cost"] * c
        )
    return sorted(results, key=lambda r: (-r.balanced_score, r.runtime))


# Robustness ranking for fallback: truly-available first, then portable kinds
# (local/server beat container which needs NGC + docker GPU runtime).
_KIND_ROBUSTNESS = {"local": 3, "server": 2, "container": 1}


def _robustness_key(r: RuntimeResult) -> tuple:
    return (1 if r.available else 0, _KIND_ROBUSTNESS.get(r.kind, 0), r.balanced_score)


@dataclass
class Selection:
    """The bake-off's recommendation: a production + fallback runtime + rationale."""
    task: str
    production: str        # runtime name
    fallback: str          # runtime name
    weights: dict[str, float]
    rationale: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "production": self.production,
            "fallback": self.fallback,
            "weights": self.weights,
            "rationale": self.rationale,
        }


def select_profiles(
    results: list[RuntimeResult],
    *,
    task: str = "",
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> Selection:
    """Pick production (best balanced score) + fallback (most-available/robust).

    Production: highest balanced_score among *available* candidates; if none are
    available (bare box) fall back to the highest overall so we still recommend.
    Fallback: the most-robust candidate that is NOT production — available + most
    portable kind wins, balanced_score breaks ties. Guarantees a degrade target.
    """
    ranked = rank(list(results), weights)
    if not ranked:
        raise ValueError("select_profiles: no results")

    available = [r for r in ranked if r.available]
    prod = (available or ranked)[0]   # ranked is best-first

    # Fallback: most-robust among the rest; if only one candidate, fallback==prod.
    rest = [r for r in ranked if r.runtime != prod.runtime]
    fb = max(rest, key=_robustness_key) if rest else prod

    prod_avail = "available" if prod.available else "UNAVAILABLE on this box (synthetic)"
    rationale = {
        "production": (
            f"{prod.runtime}: top balanced score {prod.balanced_score:.3f} "
            f"(weights {weights}); {prod_avail}. "
            "SYNTHETIC ranking — re-run on GPU with real runtimes to confirm."),
        "fallback": (
            f"{fb.runtime}: most-robust degrade target "
            f"(available={fb.available}, kind={fb.kind}, "
            f"score={fb.balanced_score:.3f})."),
        "note": (
            "All metrics are synthetic (fake timing model). The selection logic, "
            "scoring weights, and rollback are real; numbers are placeholders until "
            "the candidate runtimes run on the GPU."),
    }
    return Selection(task=task, production=prod.runtime, fallback=fb.runtime,
                     weights=dict(weights), rationale=rationale)
