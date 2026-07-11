"""Shared fixtures for vdu_scanner unit tests (no network)."""

import numpy as np
import pandas as pd
import pytest


def make_ohlcv_df(
    n: int = 60,
    base_close: float = 250.0,
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Synthetic daily OHLCV suitable for scanner tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n)
    noise = rng.normal(0, 1.5, n).cumsum()
    close = base_close + noise
    close = np.maximum(close, 50.0)

    open_ = close - rng.uniform(0, 2, n)
    high = np.maximum(open_, close) + rng.uniform(0, 3, n)
    low = np.minimum(open_, close) - rng.uniform(0, 3, n)
    volume = rng.integers(80_000, 400_000, n)

    return pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


@pytest.fixture
def ohlcv_60():
    return make_ohlcv_df(n=60, base_close=250.0)


@pytest.fixture
def ohlcv_260():
    return make_ohlcv_df(n=260, base_close=320.0, seed=7)


@pytest.fixture
def process_symbol_defaults():
    """Default scan parameters matching typical app slider values."""
    return {
        "open_price_map": {},
        "close_price_map": {},
        "high_price_map": {},
        "low_price_map": {},
        "volume_map": {},
        "min_dry": 30,
        "max_dry": 60,
        "min_vol_ratio": 2.0,
        "min_price_chg": 1.5,
        "min_dry_spikes": 2,
        "min_signal_str": 0.0,
        "above_50dma_only": False,
        "above_200dma_only": False,
        "vcp_max_tightness": 10.0,
        "sma20_lower_bound": 0.94,
        "sma20_upper_bound": 1.06,
        "sma50_lower_bound": 0.92,
        "sma50_upper_bound": 1.08,
        "sma20_min_volume": 0,
        "sma_timeframe": "All (Multi-Timeframe Convergence)",
    }
