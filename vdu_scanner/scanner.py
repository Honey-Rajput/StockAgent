# scanner.py
import pandas as pd
import numpy as np
from config import DRY_VOLUME_THRESHOLD, MIN_VOLUME_RATIO, MIN_PRICE_CHANGE, DRY_ZONE_MIN_DAYS, DRY_ZONE_MAX_DAYS

def scan_stock(
    symbol: str, 
    df: pd.DataFrame, 
    min_dry_days: int = DRY_ZONE_MIN_DAYS, 
    max_dry_days: int = DRY_ZONE_MAX_DAYS,
    min_volume_ratio: float = MIN_VOLUME_RATIO,
    min_price_change: float = MIN_PRICE_CHANGE,
    max_dry_spikes: int = 2
) -> dict | None:
    """
    Evaluates a stock's OHLCV history for the Volume Dry-Up (VDU) Breakout pattern.
    
    STEP 1: Calculate baseline average volume over available history (up to last 90 days).
    STEP 2: Exclude today's breakout day and search backward for the most recent 
            Dry Zone of length between min_dry_days and max_dry_days, allowing up to 
            max_dry_spikes where volume exceeds 40% of baseline volume.
    STEP 3: Evaluate today's candle for the breakout pattern (Volume surge, bullish candle, minimum price change).
    STEP 4: Calculate the Signal Strength Score (0 to 100).
    
    Returns a dictionary of scan metrics and modified DataFrame if matched, otherwise None.
    """
    if df is None or len(df) < 50:
        # We need at least 50 trading days to compute standard indicators (like 50 DMA) and scan for VDU
        return None
        
    # --- STEP 1: Baseline Volume ---
    # Take the last 90 trading days to compute the baseline average volume
    baseline_subset = df.iloc[-90:] if len(df) >= 90 else df
    baseline_avg_vol = baseline_subset['Volume'].mean()
    
    if baseline_avg_vol <= 0:
        return None
        
    # Enforce at least 1 day of dry consolidation to prevent division by zero or NaN average volumes
    min_dry_days = max(1, min_dry_days)
    
    # --- STEP 2: Find Dry Zone (Exclude Today) ---
    # Exclude the latest day (today's candle) to scan historical dry period
    history_df = df.iloc[:-1].reset_index(drop=True)
    
    # Identify indices that DO NOT qualify as dry days (Volume > 40% of baseline volume)
    is_not_dry_mask = history_df['Volume'] > (DRY_VOLUME_THRESHOLD * baseline_avg_vol)
    
    best_window = None  # Format: (start_idx, end_idx, length, not_dry_count)
    
    # Scan only recent days (at most 10 trading days back) to guarantee the consolidation ended recently
    search_start_idx = len(history_df) - 1
    search_end_idx = max(0, len(history_df) - 10)
    
    for idx in range(search_start_idx, search_end_idx - 1, -1):
        # Check window lengths from max_dry_days down to min_dry_days
        for L in range(max_dry_days, min_dry_days - 1, -1):
            start_idx = idx - L + 1
            if start_idx < 0:
                continue
                
            # Count spikes (non-dry volume days) in this window
            not_dry_count = is_not_dry_mask.iloc[start_idx : idx + 1].sum()
            if not_dry_count <= max_dry_spikes:
                best_window = (start_idx, idx, L, int(not_dry_count))
                break
        if best_window:
            break
            
    if best_window is None:
        return None
        
    start_idx, end_idx, dry_days_count, dry_spikes = best_window
    
    # Calculate statistics inside the dry zone
    dry_zone_df = history_df.iloc[start_idx : end_idx + 1]
    dry_avg_vol = dry_zone_df['Volume'].mean()
    
    if dry_avg_vol <= 0:
        return None
        
    # --- STEP 3: Today's Breakout Check ---
    today = df.iloc[-1]
    volume_ratio = today['Volume'] / dry_avg_vol
    
    # Condition a) Volume surge
    volume_surge_ok = volume_ratio >= min_volume_ratio
    # Condition b) Bullish candle (Close > Open)
    bullish_candle_ok = today['Close'] > today['Open']
    # Condition c) Minimum price move percentage from Open (intraday price move)
    pct_change_today = (today['Close'] - today['Open']) / today['Open'] * 100
    price_change_ok = pct_change_today >= min_price_change
    
    if not (volume_surge_ok and bullish_candle_ok and price_change_ok):
        return None
        
    # --- STEP 4: Signal Strength Score (0 to 100) & Indicators ---
    df_indicators = df.copy()
    df_indicators['MA50'] = df_indicators['Close'].rolling(window=50).mean()
    
    today_ma = df_indicators['MA50'].iloc[-1]
    above_50dma = False
    if not pd.isna(today_ma):
        above_50dma = today['Close'] > today_ma
        
    # Calculate score weights
    score = 0.0
    # 1. Volume ratio: Up to 40 points
    score += min(volume_ratio / 10.0 * 40.0, 40.0)
    # 2. Price move: Up to 30 points
    score += min(pct_change_today / 5.0 * 30.0, 30.0)
    # 3. Dry duration: Up to 20 points
    score += min(dry_days_count / 60.0 * 20.0, 20.0)
    # 4. Moving Average filter: 10 points
    if above_50dma:
        score += 10.0
        
    score = round(min(score, 100.0), 1)
    
    # Calculate day-over-day price change (standard Close-to-Close change)
    yesterday = df.iloc[-2] if len(df) >= 2 else today
    day_change_pct = ((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if len(df) >= 2 else 0.0
    
    # Calculate available 120-day high and low
    high_120d = float(df['High'].max())
    low_120d = float(df['Low'].min())
    
    from config import get_company_name
    company_name = get_company_name(symbol)
    
    return {
        "symbol": symbol.strip().upper(),
        "company_name": company_name,
        "cmp": float(today['Close']),
        "day_change_pct": round(day_change_pct, 2),
        "today_volume": int(today['Volume']),
        "dry_avg_vol": round(dry_avg_vol, 1),
        "volume_ratio": round(volume_ratio, 2),
        "dry_days_count": int(dry_days_count),
        "dry_spikes": dry_spikes,
        "dry_start_date": pd.to_datetime(history_df['Date'].iloc[start_idx]),
        "dry_end_date": pd.to_datetime(history_df['Date'].iloc[end_idx]),
        "signal_strength": score,
        "pct_change_today": round(pct_change_today, 2),
        "above_50dma": above_50dma,
        "df": df_indicators,
        "high_52w": high_120d,      # Let's label high_52w as the highest in our fetched window
        "low_52w": low_120d        # Let's label low_52w as the lowest in our fetched window
    }

def scan_coiled_spring(symbol: str, df: pd.DataFrame) -> dict | None:
    """
    Scans a stock's history to identify if it is in a "Final Contraction" (Coiled Spring / VCP) setup.
    This represents an extremely tight price consolidation with dried volume prior to breakout.
    
    1. Uptrend Filter: Close > 50 SMA, 50 SMA > 100 SMA.
    2. Volatility Squeeze: 5-Day High-Low range <= 4.0% and tighter than preceding 5-day range.
    3. Volume Dry-Up: 5-Day average volume <= 50% of 50-day average volume baseline, and declining.
    """
    if df is None or len(df) < 50:
        return None
        
    # --- STEP 1: Uptrend Filter ---
    df_copy = df.copy()
    df_copy['MA50'] = df_copy['Close'].rolling(window=50).mean()
    df_copy['MA100'] = df_copy['Close'].rolling(window=100).mean()
    df_copy['EMA20'] = df_copy['Close'].ewm(span=20, adjust=False).mean()
    
    today = df_copy.iloc[-1]
    
    # 50 SMA and 100 SMA calculations
    today_ma50 = today['MA50']
    today_ma100 = today['MA100']
    
    if pd.isna(today_ma50) or pd.isna(today_ma100):
        return None
        
    # Standard VCP uptrend check: Close is above 50 SMA, and 50 SMA is above 100 SMA
    uptrend_ok = today['Close'] > today_ma50 and today_ma50 > today_ma100
    if not uptrend_ok:
        return None
        
    # --- STEP 2: Volatility Contraction Squeeze ---
    # We inspect Segment A (last 5 trading days) and Segment B (preceding 5 trading days, days -10 to -5)
    segment_a = df_copy.iloc[-5:]
    segment_b = df_copy.iloc[-10:-5]
    
    if len(segment_a) < 5 or len(segment_b) < 5:
        return None
        
    # High-Low Range for Segment A (%)
    max_h_a = segment_a['High'].max()
    min_l_a = segment_a['Low'].min()
    range_a = ((max_h_a - min_l_a) / min_l_a) * 100
    
    # High-Low Range for Segment B (%)
    max_h_b = segment_b['High'].max()
    min_l_b = segment_b['Low'].min()
    range_b = ((max_h_b - min_l_b) / min_l_b) * 100
    
    # Check 1: Extreme Price Tightness (Range_5d <= 4.0%)
    tightness_ok = range_a <= 4.0
    # Check 2: Volatility is actively contracting (Range_5d < Range_10d_5d)
    contraction_ok = range_a < range_b
    
    if not (tightness_ok and contraction_ok):
        return None
        
    # --- STEP 3: Volume Dry-Up (VDU) ---
    # 50-day average baseline volume
    baseline_subset = df_copy.iloc[-50:]
    baseline_avg_vol = baseline_subset['Volume'].mean()
    
    if baseline_avg_vol <= 0:
        return None
        
    # Average volume for Segment A and Segment B
    avg_vol_a = segment_a['Volume'].mean()
    avg_vol_b = segment_b['Volume'].mean()
    
    # Check 1: Volume is severely dried up (AvgVolume_5d <= 50% of 50-day baseline)
    vol_dry_ok = avg_vol_a <= (0.50 * baseline_avg_vol)
    # Check 2: Volume is actively declining (AvgVolume_5d < AvgVolume_10d_5d)
    vol_declining_ok = avg_vol_a < avg_vol_b
    
    if not (vol_dry_ok and vol_declining_ok):
        return None
        
    # --- STEP 4: Squeeze Score Calculation ---
    # 1. Tightness factor: Up to 50 pts
    score_tightness = max(0.0, 100.0 - (range_a / 4.0 * 100.0)) / 2.0
    # 2. Volume dry-up factor: Up to 30 pts
    score_volume = max(0.0, 1.0 - (avg_vol_a / (0.50 * baseline_avg_vol))) * 30.0
    # 3. Support trend factor: 20 pts if closing above 20 EMA
    score_trend = 20.0 if today['Close'] > today['EMA20'] else 0.0
    
    squeeze_score = round(min(score_tightness + score_volume + score_trend, 100.0), 1)
    
    # Day change pct for reporting
    yesterday = df_copy.iloc[-2] if len(df_copy) >= 2 else today
    day_change_pct = ((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if len(df_copy) >= 2 else 0.0
    
    from config import get_company_name
    company_name = get_company_name(symbol)
    
    return {
        "symbol": symbol.strip().upper(),
        "company_name": company_name,
        "cmp": float(today['Close']),
        "day_change_pct": round(day_change_pct, 2),
        "range_5d": round(range_a, 2),
        "range_prev": round(range_b, 2),
        "avg_vol_5d": round(avg_vol_a, 1),
        "baseline_avg_vol": round(baseline_avg_vol, 1),
        "vol_ratio": round(avg_vol_a / baseline_avg_vol, 2),
        "squeeze_score": squeeze_score,
        "above_20ema": today['Close'] > today['EMA20'],
        "df": df_copy
    }

