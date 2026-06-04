"""Harness — p95/throughput correctness on known synthetic timings (no GPU)."""
import math

from benchmarks.harness import (
    BenchmarkResult,
    FakeRunner,
    RunnerResult,
    percentile,
    run_benchmark,
)


def test_percentile_matches_numpy_linear_interpolation():
    xs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert math.isclose(percentile(xs, 50), 55.0)
    assert math.isclose(percentile(xs, 95), 95.5)
    assert percentile([], 95) == 0.0
    assert percentile([42], 95) == 42.0


def _fixed_runner(latencies, video_sec=5.0, power=80.0):
    """Runner that replays a fixed latency per item (deterministic)."""
    seq = iter(latencies)

    def run(item):
        return RunnerResult(latency_ms=next(seq), video_sec=video_sec, power_w=power)
    return run


def test_run_benchmark_computes_p50_p95_and_mean():
    lats = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    items = [{"index": i, "n": len(lats)} for i in range(len(lats))]
    res = run_benchmark("t", "detection", items, _fixed_runner(lats))
    assert math.isclose(res.p50_latency_ms, 55.0)
    assert math.isclose(res.p95_latency_ms, 95.5)
    assert math.isclose(res.mean_latency_ms, 55.0)
    assert res.n_items == 10


def test_throughput_uses_injected_clock():
    # Inject a clock that advances exactly 2 seconds over the whole run.
    ticks = iter([100.0, 102.0])
    lats = [50.0] * 4
    items = [{"index": i, "n": 4, "video_sec": 5.0} for i in range(4)]
    res = run_benchmark("t", "detection", items, _fixed_runner(lats),
                        clock=lambda: next(ticks))
    assert math.isclose(res.wall_time_sec, 2.0)
    # 4 clips in 2s -> 120 clips/min
    assert math.isclose(res.clips_per_min, 120.0)
    # 4 * 5s video in 2s -> 10 video-sec/sec
    assert math.isclose(res.video_sec_per_sec, 10.0)


def test_throughput_falls_back_to_summed_latency_when_clock_flat():
    # Flat clock (no elapsed wall time) -> use summed per-item latency (4*0.25s=1s).
    lats = [250.0] * 4   # ms -> 1.0s total
    items = [{"index": i, "n": 4} for i in range(4)]
    res = run_benchmark("t", "detection", items, _fixed_runner(lats),
                        clock=lambda: 7.0)
    assert math.isclose(res.wall_time_sec, 1.0)
    assert math.isclose(res.clips_per_min, 240.0)   # 4 clips / 1s * 60


def test_avg_power_is_mean_of_samples():
    powers = iter([60.0, 80.0, 100.0])

    def run(item):
        return RunnerResult(latency_ms=10.0, power_w=next(powers))
    items = [{"index": i, "n": 3} for i in range(3)]
    res = run_benchmark("t", "detection", items, run)
    assert math.isclose(res.avg_power_w, 80.0)


def test_vlm_metrics_aggregated():
    def run(item):
        return RunnerResult(latency_ms=2000.0, ttft_ms=500.0,
                            tokens_per_sec=100.0, output_tokens=180,
                            parse_success=(item["index"] % 2 == 0), power_w=190.0)
    items = [{"index": i, "n": 4} for i in range(4)]
    res = run_benchmark("v", "clip_reasoning", items, run)
    assert math.isclose(res.tokens_per_sec, 100.0)
    assert math.isclose(res.ttft_ms_p95, 500.0)
    assert math.isclose(res.parse_success_rate, 0.5)   # 2/4
    assert res.total_output_tokens == 720


def test_detection_task_has_no_vlm_metrics():
    items = [{"index": i, "n": 3} for i in range(3)]
    res = run_benchmark("d", "detection", items, _fixed_runner([10, 20, 30]))
    assert res.tokens_per_sec is None
    assert res.ttft_ms_p95 is None
    assert res.parse_success_rate is None


def test_fake_runner_is_deterministic_and_splits_stages():
    fk = FakeRunner(task="detection", base_latency_ms=40.0, jitter_ms=10.0,
                    stages=("preprocess", "infer", "postprocess"))
    r1 = fk({"index": 0, "n": 10, "video_sec": 5.0})
    r2 = fk({"index": 0, "n": 10, "video_sec": 5.0})
    assert r1.latency_ms == r2.latency_ms        # deterministic
    # stage_ms sums to total latency; infer is the largest stage.
    assert math.isclose(sum(r1.stage_ms.values()), r1.latency_ms)
    assert r1.stage_ms["infer"] > r1.stage_ms["preprocess"]


def test_fake_runner_vlm_reports_tokens_per_sec():
    fk = FakeRunner(task="clip_reasoning", base_latency_ms=2000.0, jitter_ms=0.0,
                    tokens_per_item=200, ttft_ms=500.0, power_w=190.0)
    r = fk({"index": 0, "n": 1})
    # decode_ms = 2000-500 = 1500ms -> 200 tokens / 1.5s = 133.33 tok/s
    assert math.isclose(r.tokens_per_sec, 200.0 / 1.5)
    assert r.parse_success is True
