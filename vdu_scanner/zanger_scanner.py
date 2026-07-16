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
    require_uptrend: bool = True # Toggle to turn off the strict 50>150>200 MA filter

    # --- ranking weights (used by rank_signals / bulk scan) ---
    max_acceptable_risk_pct: float = 15.0  # stop distances above this get penalized hard
    volume_weight: float = 0.6             # how much conviction (volume) matters vs risk
    risk_weight: float = 0.4


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

def _early_entry_signal(df: pd.DataFrame, base_high: pd.Series, base_low: pd.Series) -> pd.Series:
    # Price is within the base
    in_base = (df["Close"] <= base_high) & (df["Close"] >= base_low)
    # Price is in the upper half of the base or within 10% of high
    near_high = (base_high - df["Close"]) / df["Close"] <= 0.10
    # Volume is drying up (<= 1.2x average)
    vol_dry = df["Volume"] <= df["avg_vol"] * 1.2
    return in_base & near_high & vol_dry


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
    out["early_entry"] = _early_entry_signal(out, base_high, base_low)

    if getattr(cfg, 'require_uptrend', True):
        out["zanger_signal"] = out["trend_ok"] & (is_hft | is_flat_base) & (out["breakout"] | out["early_entry"])
    else:
        out["zanger_signal"] = (is_hft | is_flat_base) & (out["breakout"] | out["early_entry"])

    setup_conds = [
        out["breakout"] & is_hft,
        out["breakout"] & is_flat_base,
        out["early_entry"] & is_hft,
        out["early_entry"] & is_flat_base
    ]
    setup_labels = [
        "Breakout (High Tight Flag)",
        "Breakout (Flat Base)",
        "Early Entry (High Tight Flag)",
        "Early Entry (Flat Base)"
    ]
    out["setup_type"] = np.select(setup_conds, setup_labels, default=None)

    out["suggested_stop"] = out[["base_low", "Low"]].max(axis=1)
    out["risk_pct"] = (out["Close"] - out["suggested_stop"]) / out["Close"] * 100
    out["target_price"] = out["Close"] + 3 * (out["Close"] - out["suggested_stop"])

    return out

def get_latest_signal(df_result: pd.DataFrame) -> dict:
    row = df_result.iloc[-1]
    vol_ratio = row["Volume"] / row["avg_vol"] if row["avg_vol"] else None
    risk_pct = row["risk_pct"] if pd.notna(row["risk_pct"]) else None
    return {
        "date": df_result.index[-1].strftime('%Y-%m-%d') if hasattr(df_result.index[-1], 'strftime') else str(df_result.index[-1]).split(' ')[0],
        "close": row["Close"],
        "zanger_signal": bool(row["zanger_signal"]),
        "setup_type": row["setup_type"],
        "prior_run_pct": round(row["prior_run_pct"], 1) if pd.notna(row["prior_run_pct"]) else None,
        "base_depth_pct": round(row["base_depth_pct"], 1) if pd.notna(row["base_depth_pct"]) else None,
        "breakout_volume_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "suggested_stop": round(row["suggested_stop"], 2) if pd.notna(row["suggested_stop"]) else None,
        "risk_pct": round(risk_pct, 2) if risk_pct is not None else None,
        "target_price": round(row["target_price"], 2) if pd.notna(row.get("target_price")) else None,
        "score": round(row.get("score"), 1) if "score" in row and pd.notna(row.get("score")) else None,
        "confidence_level": row.get("confidence_level", None),
        "breakout_status": row.get("breakout_status", None),
    }


# ---------------------------------------------------------------------------
# Ranking - conviction (volume) vs. risk (stop distance)
# ---------------------------------------------------------------------------

def rank_signals(hits: pd.DataFrame, cfg: ZangerConfig = ZangerConfig()) -> pd.DataFrame:
    df = hits.copy()
    if df.empty:
        return df

    max_vol_ratio = min(df["breakout_volume_ratio"].max(), 3.0) or 1
    df["volume_score"] = (df["breakout_volume_ratio"].clip(upper=3.0) / max_vol_ratio * 100).clip(0, 100)

    def risk_score(risk_pct):
        if pd.isna(risk_pct):
            return 0
        if risk_pct <= cfg.max_acceptable_risk_pct:
            return max(0, 100 - (risk_pct / cfg.max_acceptable_risk_pct) * 60)
        overshoot = risk_pct - cfg.max_acceptable_risk_pct
        return max(0, 40 - overshoot * 4)

    df["risk_score"] = df["risk_pct"].apply(risk_score)

    vol_weight = getattr(cfg, "volume_weight", 0.6)
    risk_weight = getattr(cfg, "risk_weight", 0.4)

    df["score"] = (df["volume_score"] * vol_weight + df["risk_score"] * risk_weight).round(1)

    # Calculate confidence level based on score
    conditions = [
        (df["score"] >= 80),
        (df["score"] >= 50)
    ]
    choices = ["High", "Medium"]
    df["confidence_level"] = np.select(conditions, choices, default="Low")

    # Determine Breakout Status from setup_type
    df["breakout_status"] = df["setup_type"].apply(lambda x: "Early Breakout" if isinstance(x, str) and "Early Entry" in x else "Already Breakout")

    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    return df
