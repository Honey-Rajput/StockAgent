import pandas as pd
import numpy as np
import pytest
from scanner import scan_hh_hl_uptrend

def create_synthetic_uptrend_df(n_bars: int = 60) -> pd.DataFrame:
    """Creates a synthetic OHLCV dataframe with ascending swing highs and swing lows."""
    dates = pd.date_range("2026-01-01", periods=n_bars, freq="D")
    
    # Base upward drift + sinusoidal swings
    t = np.linspace(0, 4 * np.pi, n_bars)
    drift = np.linspace(100, 160, n_bars)
    swing = 8.0 * np.sin(t)
    close = drift + swing
    high = close + np.random.uniform(1.0, 2.5, n_bars)
    low = close - np.random.uniform(1.0, 2.5, n_bars)
    open_ = close + np.random.uniform(-0.5, 0.5, n_bars)
    volume = np.random.randint(500000, 2000000, n_bars)

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    return df

def test_scan_hh_hl_detects_uptrend():
    df = create_synthetic_uptrend_df(70)
    res = scan_hh_hl_uptrend("TESTSYM", df, window=3)
    assert res is not None
    assert res['symbol'] == "TESTSYM"
    assert res['hh_count'] >= 1
    assert res['hl_count'] >= 1
    assert res['trend_score'] > 0
    assert 'setup_tag' in res
    assert res['cmp'] > 0

def test_scan_hh_hl_rejects_insufficient_bars():
    df = create_synthetic_uptrend_df(15)
    res = scan_hh_hl_uptrend("SHORT", df)
    assert res is None

def test_scan_hh_hl_rejects_downtrend():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    # Declining price
    close = np.linspace(200, 100, 60)
    df = pd.DataFrame({
        'Open': close + 1,
        'High': close + 2,
        'Low': close - 2,
        'Close': close,
        'Volume': [100000] * 60
    }, index=dates)
    res = scan_hh_hl_uptrend("DOWN", df)
    assert res is None
