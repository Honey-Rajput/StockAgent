"""Tests for WaveTrend scan — especially CMP vs SMA flags."""

import pandas as pd

from scanner import scan_wt_cross
from tests.conftest import make_ohlcv_df


def _wt_indicators_df(base_df: pd.DataFrame, sma20: float, sma50: float) -> dict:
    """Pre-computed indicator frame so WT path is deterministic without tuning prices."""
    df = base_df.copy()
    df["WT1"] = -55.0
    df["WT2"] = -50.0
    df["SMA20"] = sma20
    df["SMA50"] = sma50
    return {"df": df}


def test_above_sma_flags_use_cmp_not_buy_price():
    """
    CMP=250, SMA20=240 => above_20sma True.
    Buy price (from support) is typically below CMP — old bug compared buy_price to SMA.
    """
    df = make_ohlcv_df(n=50, base_close=250.0, seed=1)
    cmp = float(df["Close"].iloc[-1])
    assert cmp >= 100.0

    # SMA20 below CMP -> should be above 20 SMA
    ind = _wt_indicators_df(df, sma20=cmp - 15.0, sma50=cmp - 30.0)
    result = scan_wt_cross("TEST", df, wt_oversold_threshold=-40.0, indicators=ind)
    assert result is not None
    assert result["above_20sma"] is True
    assert result["above_50sma"] is True

    # SMA20 above CMP -> should NOT be above 20 SMA
    ind2 = _wt_indicators_df(df, sma20=cmp + 20.0, sma50=cmp + 40.0)
    result2 = scan_wt_cross("TEST", df, wt_oversold_threshold=-40.0, indicators=ind2)
    assert result2 is not None
    assert result2["above_20sma"] is False
    assert result2["above_50sma"] is False


def test_scan_wt_cross_rejects_penny_stocks():
    df = make_ohlcv_df(n=50, base_close=80.0, seed=2)
    ind = _wt_indicators_df(df, sma20=70.0, sma50=65.0)
    assert scan_wt_cross("TEST", df, indicators=ind) is None
