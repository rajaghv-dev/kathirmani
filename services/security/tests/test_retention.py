"""retention: selection + evidence-lock logic on synthetic data. No DB needed."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from services.security import retention

NOW = datetime(2026, 6, 4, tzinfo=timezone.utc)


def _row(rid, age_days, tcol="created_at"):
    return {"id": rid, tcol: NOW - timedelta(days=age_days)}


def test_loads_real_policy():
    pol = retention.load_policy()
    assert pol["retention"]["raw_clips_days"] == 14
    assert pol["cleanup"]["dry_run_default"] is True


def test_select_expired_by_age():
    rows = [_row("a", 20), _row("b", 5), _row("c", 14), _row("d", 15)]
    # >14 days → strictly older than the cutoff (cutoff = NOW-14d; <= edge expires)
    expired = retention.select_expired(rows, days=14, now=NOW)
    ids = {r["id"] for r in expired}
    # 'c' is exactly 14d old (== edge) → expired; 'b' (5d) retained
    assert ids == {"a", "c", "d"}


def test_evidence_lock_skips_open_incident_clips():
    rows = [_row("clip1", 30), _row("clip2", 30), _row("clip3", 30)]
    locked = {"clip2"}  # tied to an OPEN incident
    expired = retention.select_expired(rows, days=14, now=NOW, locked_ids=locked)
    ids = {r["id"] for r in expired}
    assert ids == {"clip1", "clip3"}
    assert "clip2" not in ids  # chain-of-custody: locked, not deleted


def test_no_lock_set_deletes_all_expired():
    rows = [_row("x", 100), _row("y", 100)]
    expired = retention.select_expired(rows, days=14, now=NOW)
    assert {r["id"] for r in expired} == {"x", "y"}


def test_custom_time_field_events():
    rows = [_row("e1", 200, tcol="event_time"), _row("e2", 10, tcol="event_time")]
    expired = retention.select_expired(rows, days=180, now=NOW, time_field="event_time")
    assert {r["id"] for r in expired} == {"e1"}


def test_missing_timestamp_skipped():
    rows = [{"id": "n"}, _row("o", 100)]
    expired = retention.select_expired(rows, days=14, now=NOW)
    assert {r["id"] for r in expired} == {"o"}


def test_cli_dry_run_is_default_and_safe(capsys):
    # No DB in unit env → plan() returns [] (guarded). Must not raise, must not apply.
    rc = retention.main([])
    assert rc == 0
    assert "DRY-RUN" in capsys.readouterr().out


def test_table_map_covers_policy_keys():
    pol = retention.load_policy()
    # thumbnails has no table mapping (FS-only) — every *other* key must map.
    unmapped = {k for k in pol["retention"] if k not in retention.TABLE_FOR}
    assert unmapped == {"thumbnails_days"}
