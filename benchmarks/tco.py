"""TCO cost model — pure functions, extending the existing energy(Wh)→₹ model.

Reuses the repo's electricity-cost concept from `results/economy.json` /
`src/marlin/metrics.compute_economy`:

    energy_wh = avg_power_w * (seconds / 3600)
    cost_inr  = energy_wh / 1000 * electricity_rate_inr_per_kwh

The economy.json default rate is ₹10/kWh (i.e. ₹0.01/Wh — verifiable as
cost_inr/energy_wh = 0.081905/8.1905 = 0.01). We keep that as the default here.

These functions take avg GPU power (W) + throughput and derive the three Phase 11
TCO headline metrics:
  - cost_per_1000_clips
  - cost_per_camera_month
  - cost_per_store_month

All pure + deterministic so the unit tests can assert exact numbers.
"""
from __future__ import annotations

# Default electricity tariff, ₹ per kWh — matches results/economy.json (₹0.01/Wh).
DEFAULT_RATE_INR_PER_KWH: float = 10.0

# Calendar assumption for "/month" metrics: 30 days. Cameras/stores run 24/7.
HOURS_PER_DAY: float = 24.0
DAYS_PER_MONTH: float = 30.0
SECONDS_PER_HOUR: float = 3600.0


def energy_wh(avg_power_w: float, seconds: float) -> float:
    """Energy (watt-hours) drawn at `avg_power_w` over `seconds`.

    Same formula as src/marlin/metrics.compute_economy.
    """
    return avg_power_w * (seconds / SECONDS_PER_HOUR)


def cost_inr(energy_watt_hours: float, rate_inr_per_kwh: float = DEFAULT_RATE_INR_PER_KWH) -> float:
    """Convert energy (Wh) to ₹ at the given tariff (Wh→kWh then × rate)."""
    return energy_watt_hours / 1000.0 * rate_inr_per_kwh


def cost_per_hour(avg_power_w: float, rate_inr_per_kwh: float = DEFAULT_RATE_INR_PER_KWH) -> float:
    """₹ to run one GPU for one hour at `avg_power_w`."""
    return cost_inr(energy_wh(avg_power_w, SECONDS_PER_HOUR), rate_inr_per_kwh)


def cost_per_1000_clips(
    avg_power_w: float,
    clips_per_min: float,
    *,
    n: int = 1000,
    rate_inr_per_kwh: float = DEFAULT_RATE_INR_PER_KWH,
) -> float:
    """₹ of GPU energy to process `n` (default 1000) clips at `clips_per_min`.

    seconds = n / (clips_per_min / 60); cost = energy(power, seconds) → ₹.
    """
    if clips_per_min <= 0:
        return 0.0
    seconds = n / (clips_per_min / 60.0)
    return cost_inr(energy_wh(avg_power_w, seconds), rate_inr_per_kwh)


def cost_per_camera_month(
    avg_power_w: float,
    n_cameras: int,
    *,
    rate_inr_per_kwh: float = DEFAULT_RATE_INR_PER_KWH,
    hours_per_day: float = HOURS_PER_DAY,
    days_per_month: float = DAYS_PER_MONTH,
) -> float:
    """₹/camera/month: amortize the box's 24/7 GPU power across `n_cameras`.

    The GPU draws `avg_power_w` continuously while serving `n_cameras` streams;
    monthly box energy ÷ cameras = per-camera share.
    """
    if n_cameras <= 0:
        return 0.0
    seconds = hours_per_day * days_per_month * SECONDS_PER_HOUR
    box_month_inr = cost_inr(energy_wh(avg_power_w, seconds), rate_inr_per_kwh)
    return box_month_inr / n_cameras


def cost_per_store_month(
    avg_power_w: float,
    *,
    rate_inr_per_kwh: float = DEFAULT_RATE_INR_PER_KWH,
    hours_per_day: float = HOURS_PER_DAY,
    days_per_month: float = DAYS_PER_MONTH,
) -> float:
    """₹/store/month: one box runs one store 24/7 (= whole-box monthly energy ₹)."""
    seconds = hours_per_day * days_per_month * SECONDS_PER_HOUR
    return cost_inr(energy_wh(avg_power_w, seconds), rate_inr_per_kwh)


def tco_report(
    avg_power_w: float,
    clips_per_min: float,
    n_cameras: int,
    *,
    rate_inr_per_kwh: float = DEFAULT_RATE_INR_PER_KWH,
) -> dict[str, float]:
    """Bundle the three Phase 11 TCO headline metrics + inputs into one dict."""
    return {
        "avg_power_w": avg_power_w,
        "electricity_rate_inr_per_kwh": rate_inr_per_kwh,
        "n_cameras": n_cameras,
        "cost_per_hour_inr": cost_per_hour(avg_power_w, rate_inr_per_kwh),
        "cost_per_1000_clips_inr": cost_per_1000_clips(
            avg_power_w, clips_per_min, rate_inr_per_kwh=rate_inr_per_kwh),
        "cost_per_camera_month_inr": cost_per_camera_month(
            avg_power_w, n_cameras, rate_inr_per_kwh=rate_inr_per_kwh),
        "cost_per_store_month_inr": cost_per_store_month(
            avg_power_w, rate_inr_per_kwh=rate_inr_per_kwh),
    }
