from dataclasses import dataclass
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config knobs - tune these to taste
# ---------------------------------------------------------------------------

@dataclass
class ZangerConfig:
    ma_fast: int = 50
    ma_slow: int = 150
    ma_slowest: int = 200

    lookback_run: int = 40          # bars to look back for the "prior move"
    min_run_pct: float = 25.0       # min % move off the low to qualify as a run
    hft_run_pct: float = 90.0       # high-tight-flag threshold (Zanger's favorite)

    base_lookback: int = 15         # bars considered for the base/consolidation
    max_base_depth_pct: float = 25.0   # base shouldn't retrace more than this
    hft_max_base_depth_pct: float = 25.0  # HTF bases are shallow (10-25%)

    breakout_vol_mult: float = 2.0  # breakout volume vs avg volume
    avg_vol_window: int = 50

    min_close_above_base_pct: float = 0.0  # 0 = just needs to close above resistance


# ---------------------------------------------------------------------------
# Core indicators
# ---------------------------------------------------------------------------

def _add_moving_averages(df: pd.DataFrame, cfg: ZangerConfig) -> pd.DataFrame:
    df = df.copy()
    df["ma_fast"] = df["Close"].rolling(cfg.ma_fast).mean()
    df["ma_slow"] = df["Close"].rolling(cfg.ma_slow).mean()
    df["ma_slowest"] = df["Close"].rolling(cfg.ma_slowest).mean()
    df["avg_vol"] = df["Volume"].rolling(cfg.avg_vol_window).mean()
    return df


def _uptrend_stack(row) -> bool:
    """Price > 50MA > 150MA > 200MA, the classic Zanger/CANSLIM stack."""
    try:
        return (
            row["Close"] > row["ma_fast"] > row["ma_slow"] > row["ma_slowest"]
        )
    except TypeError:
        return False


def _ma_sloping_up(df: pd.DataFrame, col: str, bars: int = 10) -> pd.Series:
    return df[col] > df[col].shift(bars)


# ---------------------------------------------------------------------------
# Prior run detection (the fuel behind the base)
# ---------------------------------------------------------------------------

def _prior_run_pct(df: pd.DataFrame, cfg: ZangerConfig) -> pd.Series:
    """
    % move from the lowest low to the highest high within lookback_run bars
    *before* the current base window. Approximates "how big was the move
    that built up the excitement before this base."
    """
    window = df["High"].rolling(cfg.lookback_run)
    low_window = df["Low"].rolling(cfg.lookback_run)
    run_high = window.max()
    run_low = low_window.min()
    return (run_high - run_low) / run_low * 100


# ---------------------------------------------------------------------------
# Base / consolidation detection
# ---------------------------------------------------------------------------

def _base_stats(df: pd.DataFrame, cfg: ZangerConfig):
    """
    Returns rolling base_high (resistance), base_low, and base_depth_pct
    over the most recent `base_lookback` bars (excluding today, so today
    can be tested as the breakout bar).
    """
    base_high = df["High"].shift(1).rolling(cfg.base_lookback).max()
    base_low = df["Low"].shift(1).rolling(cfg.base_lookback).min()
    base_depth_pct = (base_high - base_low) / base_high * 100
    return base_high, base_low, base_depth_pct


# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------

def _breakout_signal(df: pd.DataFrame, base_high: pd.Series, cfg: ZangerConfig) -> pd.Series:
    price_break = df["Close"] > base_high * (1 + cfg.min_close_above_base_pct / 100)
    vol_break = df["Volume"] > df["avg_vol"] * cfg.breakout_vol_mult
    return price_break & vol_break


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scan_zanger(df: pd.DataFrame, cfg: ZangerConfig = ZangerConfig()) -> pd.DataFrame:
    """
    df must have columns: Open, High, Low, Close, Volume (yfinance format).
    """
    out = _add_moving_averages(df, cfg)
    out["trend_ok"] = out.apply(_uptrend_stack, axis=1) & _ma_sloping_up(out, "ma_fast")
    out["prior_run_pct"] = _prior_run_pct(out, cfg)
    
    base_high, base_low, base_depth_pct = _base_stats(out, cfg)
    out["base_high"] = base_high
    out["base_low"] = base_low
    out["base_depth_pct"] = base_depth_pct

    valid_base = out["base_depth_pct"] <= cfg.max_base_depth_pct
    is_hft = (out["prior_run_pct"] >= cfg.hft_run_pct) & (
        out["base_depth_pct"] <= cfg.hft_max_base_depth_pct
    )
    is_flat_base = (out["prior_run_pct"] >= cfg.min_run_pct) & valid_base & ~is_hft

    out["is_high_tight_flag"] = is_hft
    out["is_flat_base"] = is_flat_base

    out["breakout"] = _breakout_signal(out, base_high, cfg)

    out["zanger_signal"] = out["trend_ok"] & (is_hft | is_flat_base) & out["breakout"]

    out["setup_type"] = np.select(
        [out["zanger_signal"] & is_hft, out["zanger_signal"] & is_flat_base],
        ["high_tight_flag", "flat_base"],
        default=None,
    )

    out["suggested_stop"] = out[["base_low", "Low"]].max(axis=1)
    return out

def get_latest_signal(df_result: pd.DataFrame) -> dict:
    row = df_result.iloc[-1]
    return {
        "date": df_result.index[-1],
        "close": row["Close"],
        "zanger_signal": bool(row["zanger_signal"]),
        "setup_type": row["setup_type"],
        "prior_run_pct": round(row["prior_run_pct"], 1) if pd.notna(row["prior_run_pct"]) else None,
        "base_depth_pct": round(row["base_depth_pct"], 1) if pd.notna(row["base_depth_pct"]) else None,
        "breakout_volume_ratio": round(row["Volume"] / row["avg_vol"], 2) if row["avg_vol"] else None,
        "suggested_stop": round(row["suggested_stop"], 2) if pd.notna(row["suggested_stop"]) else None,
    }
