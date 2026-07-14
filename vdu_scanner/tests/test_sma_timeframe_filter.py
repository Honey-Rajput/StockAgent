"""Tests for SMA timeframe tab filtering (DB legacy + new passes_* flags)."""

from ui_components import matches_sma_timeframe_filter


def test_legacy_record_without_flags_shows_in_all_timeframes():
    record = {"symbol": "TEST", "cmp": 300.0}
    assert matches_sma_timeframe_filter(record, "Daily") is True
    assert matches_sma_timeframe_filter(record, "Weekly") is True
    assert matches_sma_timeframe_filter(record, "All (Daily + Weekly Convergence)") is True


def test_daily_only_record():
    record = {"passes_daily": True, "passes_weekly": False}
    assert matches_sma_timeframe_filter(record, "Daily") is True
    assert matches_sma_timeframe_filter(record, "Weekly") is False
    assert matches_sma_timeframe_filter(record, "All (Daily + Weekly Convergence)") is False


def test_weekly_only_record():
    record = {"passes_daily": False, "passes_weekly": True}
    assert matches_sma_timeframe_filter(record, "Daily") is False
    assert matches_sma_timeframe_filter(record, "Weekly") is True
