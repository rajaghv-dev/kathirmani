"""TCO math — exact-number assertions (pure functions, no GPU/DB)."""
import math

from benchmarks import tco


def test_energy_and_cost_formula_matches_economy_model():
    # Same formula as src/marlin/metrics.compute_economy:
    #   energy_wh = power * sec/3600 ; cost_inr = energy/1000 * rate
    e = tco.energy_wh(100.0, 3600.0)          # 100W for 1h
    assert e == 100.0
    assert tco.cost_inr(100.0, 10.0) == 1.0   # 100Wh @ ₹10/kWh = ₹1


def test_default_rate_is_economy_json_tariff():
    # results/economy.json: cost_inr/energy_wh = 0.081905/8.1905 = 0.01 ₹/Wh = ₹10/kWh
    assert tco.DEFAULT_RATE_INR_PER_KWH == 10.0


def test_cost_per_hour():
    assert tco.cost_per_hour(100.0, 10.0) == 1.0
    assert tco.cost_per_hour(250.0, 8.0) == 2.0     # 250Wh @ ₹8/kWh


def test_cost_per_1000_clips_exact():
    # 60W, 60 clips/min -> 1000 clips take 1000s -> 16.6667Wh -> ₹0.16667
    c = tco.cost_per_1000_clips(60.0, 60.0, rate_inr_per_kwh=10.0)
    assert math.isclose(c, 60.0 * (1000.0 / 3600.0) / 1000.0 * 10.0)
    assert math.isclose(c, 0.16666666666666666)


def test_cost_per_1000_clips_zero_throughput_is_zero():
    assert tco.cost_per_1000_clips(60.0, 0.0) == 0.0


def test_cost_per_store_month_exact():
    # 100W continuous, 24*30h = 720h -> 72000Wh -> @₹10/kWh = ₹720
    assert tco.cost_per_store_month(100.0, rate_inr_per_kwh=10.0) == 720.0


def test_cost_per_camera_month_splits_box_evenly():
    # Same box (₹720/mo) shared across 4 cameras -> ₹180 each.
    assert tco.cost_per_camera_month(100.0, 4, rate_inr_per_kwh=10.0) == 180.0
    # Store month == camera_month * n_cameras (consistency).
    store = tco.cost_per_store_month(100.0, rate_inr_per_kwh=10.0)
    cam = tco.cost_per_camera_month(100.0, 4, rate_inr_per_kwh=10.0)
    assert math.isclose(store, cam * 4)


def test_cost_per_camera_month_zero_cameras_is_zero():
    assert tco.cost_per_camera_month(100.0, 0) == 0.0


def test_tco_report_bundles_all_three_headline_metrics():
    r = tco.tco_report(100.0, 60.0, 4, rate_inr_per_kwh=10.0)
    assert r["cost_per_1000_clips_inr"] == tco.cost_per_1000_clips(100.0, 60.0, rate_inr_per_kwh=10.0)
    assert r["cost_per_camera_month_inr"] == 180.0
    assert r["cost_per_store_month_inr"] == 720.0
    assert r["avg_power_w"] == 100.0
    assert r["n_cameras"] == 4
