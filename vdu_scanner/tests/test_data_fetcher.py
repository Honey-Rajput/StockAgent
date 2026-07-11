"""Tests for yfinance DataFrame normalization."""

import pandas as pd

from data_fetcher import _flatten_yf_dataframe


def _sample_ohlcv_indexed(n=5):
    dates = pd.bdate_range("2024-06-01", periods=n)
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [102.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [101.0 + i for i in range(n)],
            "Volume": [50_000 + i * 1000 for i in range(n)],
        },
        index=dates,
    )


def test_flatten_flat_single_ticker():
    raw = _sample_ohlcv_indexed()
    out = _flatten_yf_dataframe(raw)
    assert out is not None
    assert list(out.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 5
    assert out["Date"].dt.tz is None


def _yfinance_multiindex(dfs: dict) -> pd.DataFrame:
    """Build column MultiIndex like yfinance: (Field, Ticker)."""
    parts = []
    for ticker, frame in dfs.items():
        sub = frame.copy()
        sub.columns = pd.MultiIndex.from_arrays([sub.columns, [ticker] * len(sub.columns)])
        parts.append(sub)
    return pd.concat(parts, axis=1)


def test_flatten_multiindex_single_ticker_wrapped():
    raw = _sample_ohlcv_indexed()
    wrapped = _yfinance_multiindex({"RELIANCE.NS": raw})
    out = _flatten_yf_dataframe(wrapped)
    assert out is not None
    assert "Close" in out.columns


def test_flatten_multiindex_multi_ticker_extracts_symbol():
    raw_a = _sample_ohlcv_indexed()
    raw_b = _sample_ohlcv_indexed()
    raw_b["Close"] = raw_b["Close"] + 50
    multi = _yfinance_multiindex({"RELIANCE.NS": raw_a, "TCS.NS": raw_b})
    out = _flatten_yf_dataframe(multi, symbol_ns="TCS.NS")
    assert out is not None
    assert float(out["Close"].iloc[-1]) == float(raw_b["Close"].iloc[-1])


def test_flatten_renames_price_to_close():
    raw = _sample_ohlcv_indexed().rename(columns={"Close": "Price"})
    out = _flatten_yf_dataframe(raw)
    assert out is not None
    assert "Close" in out.columns


def test_flatten_returns_none_for_empty():
    assert _flatten_yf_dataframe(pd.DataFrame()) is None
    assert _flatten_yf_dataframe(None) is None
