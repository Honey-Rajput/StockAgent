"""Tests for process_single_symbol orchestration."""

from scan_orchestrator import process_single_symbol
from tests.conftest import make_ohlcv_df


def _run(sym, df, defaults):
    return process_single_symbol(sym, df, **defaults)


def test_process_single_symbol_rejects_short_history(process_symbol_defaults):
    res = _run("TEST", make_ohlcv_df(n=3), process_symbol_defaults)
    assert res["failed"] is True


def test_process_single_symbol_rejects_low_price(process_symbol_defaults):
    res = _run("TEST", make_ohlcv_df(n=60, base_close=150.0), process_symbol_defaults)
    assert res["failed"] is True


def test_process_single_symbol_returns_expected_shape(process_symbol_defaults, ohlcv_260):
    res = _run("TEST", ohlcv_260, process_symbol_defaults)
    assert res["failed"] is False
    for key in (
        "gapup", "above_ma", "support_ma", "crossover_ma",
        "minervini", "flagged", "wt", "vcs", "structural_vcp", "vpa",
    ):
        assert key in res


def test_process_single_symbol_handles_none_df(process_symbol_defaults):
    res = _run("TEST", None, process_symbol_defaults)
    assert res["failed"] is True
