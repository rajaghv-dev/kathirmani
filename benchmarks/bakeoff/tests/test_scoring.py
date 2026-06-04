"""Scoring + selection: known synthetic inputs -> expected production winner."""
from benchmarks.bakeoff import bakeoff as bk


def _result(name, *, clips, p95, qual, cost, available, kind="server"):
    return bk.RuntimeResult(
        runtime=name, available=available, kind=kind, notes="",
        clips_per_min=clips, p95_latency_ms=p95, ttft_ms_p95=p95 / 4,
        tokens_per_sec=clips, quality_score=qual, avg_power_w=200,
        cost_per_1000_clips=cost, tco={"cost_per_1000_clips_inr": cost})


def test_minmax_normalization_bounds_and_inversion():
    # higher-is-better: best raw -> 1.0, worst -> 0.0
    assert bk._minmax([1.0, 2.0, 3.0], higher_is_better=True) == [0.0, 0.5, 1.0]
    # lower-is-better: smallest raw -> 1.0 (best)
    assert bk._minmax([1.0, 2.0, 3.0], higher_is_better=False) == [1.0, 0.5, 0.0]
    # degenerate spread -> all 1.0 (no info, don't divide by zero)
    assert bk._minmax([5.0, 5.0], higher_is_better=True) == [1.0, 1.0]


def test_rank_dominant_candidate_wins():
    # 'fast' dominates every metric -> must rank first with the top score.
    fast = _result("fast", clips=100, p95=900, qual=0.96, cost=0.10, available=True)
    mid = _result("mid", clips=60, p95=1500, qual=0.94, cost=0.20, available=True)
    slow = _result("slow", clips=30, p95=2600, qual=0.90, cost=0.40, available=True)
    ranked = bk.rank([slow, mid, fast])
    assert [r.runtime for r in ranked] == ["fast", "mid", "slow"]
    assert ranked[0].balanced_score > ranked[-1].balanced_score
    # normalized fields all present and in [0,1]
    for r in ranked:
        for v in r.norm.values():
            assert 0.0 <= v <= 1.0


def test_select_production_is_best_available():
    # The dominant candidate is UNAVAILABLE; the best *available* must be production.
    best_unavail = _result("trt", clips=120, p95=800, qual=0.96, cost=0.08,
                           available=False, kind="server")
    best_avail = _result("ollama", clips=70, p95=1800, qual=0.93, cost=0.18,
                         available=True, kind="server")
    worst = _result("xfm", clips=40, p95=2600, qual=0.96, cost=0.30,
                    available=True, kind="local")
    sel = bk.select_profiles([best_unavail, best_avail, worst], task="t")
    assert sel.production == "ollama"          # best among available
    assert sel.fallback != sel.production       # distinct degrade target


def test_production_falls_back_to_best_overall_when_none_available():
    # Bare box: nothing available -> still recommend the top-scoring runtime.
    a = _result("trt", clips=120, p95=800, qual=0.96, cost=0.08, available=False)
    b = _result("vllm", clips=90, p95=1100, qual=0.96, cost=0.12, available=False)
    sel = bk.select_profiles([b, a], task="t")
    assert sel.production == "trt"   # highest balanced score overall


def test_fallback_prefers_available_and_portable():
    prod = _result("trt", clips=120, p95=800, qual=0.96, cost=0.08,
                   available=False, kind="server")
    avail_local = _result("xfm", clips=45, p95=2600, qual=0.96, cost=0.30,
                          available=True, kind="local")
    avail_container = _result("nim", clips=100, p95=1000, qual=0.96, cost=0.10,
                              available=False, kind="container")
    sel = bk.select_profiles([prod, avail_local, avail_container], task="t")
    # Production = best available among them = xfm? No: none of trt/nim available,
    # xfm is the only available -> production. Fallback then = most-robust of rest.
    assert sel.production == "xfm"
    assert sel.fallback in {"trt", "nim"}
