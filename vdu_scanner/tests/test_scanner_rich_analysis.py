"""Tests for compute_rich_analysis slow path and rich JSON output."""

import json

import pandas as pd

from scanner import compute_rich_analysis
from tests.conftest import make_ohlcv_df


def test_compute_rich_analysis_returns_valid_json_without_indicators():
    df = make_ohlcv_df(n=60, base_close=220.0)
    rec = compute_rich_analysis(
        df, "TEST", "Unit Test",
        "Set stop loss at 200.00 with target 260.00.",
        indicators=None,
    )
    payload = json.loads(rec)
    assert payload["is_rich"] is True
    assert isinstance(payload["rsi"], (int, float))
    assert isinstance(payload["cci"], (int, float))
    assert payload["oi_status"] is None
    assert payload["oi_interp"] is None
    assert "Set stop loss" in payload["text"]


def test_compute_rich_analysis_includes_oi_when_column_present():
    n = 20
    dates = pd.bdate_range("2024-01-01", periods=n)
    # Monotonic rising close => last bar is price up vs previous
    close = [300.0 + i * 2.0 for i in range(n)]
    df = pd.DataFrame({
        "Date": dates,
        "Open": close,
        "High": [c + 1 for c in close],
        "Low": [c - 1 for c in close],
        "Close": close,
        "Volume": [100_000] * n,
        "Open Interest": [1_000_000 + i * 10_000 for i in range(n)],
    })
    rec = compute_rich_analysis(df, "TEST", "OI Test", "Set stop loss at 290.", indicators=None)
    payload = json.loads(rec)
    assert payload["oi_status"] == "Long Build-up"
    assert payload["oi_interp"] is not None
    assert "OI Analysis" in payload["text"]


def test_compute_rich_analysis_short_data_returns_plain_text():
    df = make_ohlcv_df(n=5, base_close=200.0)
    base = "Simple recommendation text."
    assert compute_rich_analysis(df, "TEST", "Short", base, indicators=None) == base
