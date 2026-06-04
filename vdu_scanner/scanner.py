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
    min_dry_spikes: int = 2
) -> dict | None:
    """
    Evaluates a stock's OHLCV history for the Volume Dry-Up (VDU) Breakout pattern.
    
    STEP 1: Calculate baseline average volume over available history (up to last 90 days).
    STEP 2: Exclude today's breakout day and search backward for the most recent 
            Dry Zone of length between min_dry_days and max_dry_days, requiring at least 
            min_dry_spikes where volume exceeds 40% of baseline volume (accumulation spikes).
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
    
    best_window = None  # Format: (start_idx, end_idx, length, not_dry_count, dry_avg_vol)
    min_found_avg_vol = float('inf')
    
    # Scan only recent days (at most 10 trading days back) to guarantee the consolidation ended recently
    search_start_idx = len(history_df) - 1
    search_end_idx = max(0, len(history_df) - 10)
    
    for idx in range(search_start_idx, search_end_idx - 1, -1):
        # Check window lengths from min_dry_days up to max_dry_days to find the best tight dry zone
        for L in range(min_dry_days, max_dry_days + 1):
            start_idx = idx - L + 1
            if start_idx < 0:
                continue
                
            # Calculate average volume of this candidate dry zone window
            dry_zone_df = history_df.iloc[start_idx : idx + 1]
            dry_avg_vol = dry_zone_df['Volume'].mean()
            
            # Enforce quality filter: consolidation zone average volume MUST be dry (<= 60% of baseline average)
            if dry_avg_vol > 0 and dry_avg_vol <= (0.60 * baseline_avg_vol):
                # Count spikes (non-dry volume days where Volume > 40% of baseline) inside the window
                not_dry_count = is_not_dry_mask.iloc[start_idx : idx + 1].sum()
                if not_dry_count >= min_dry_spikes:
                    # We found a valid dry window! Select the driest one to represent the highest quality contraction
                    if dry_avg_vol < min_found_avg_vol:
                        min_found_avg_vol = dry_avg_vol
                        best_window = (start_idx, idx, L, int(not_dry_count), dry_avg_vol)
                        
    if best_window is None:
        return None
        
    start_idx, end_idx, dry_days_count, dry_spikes, dry_avg_vol = best_window
        
    # --- STEP 3: Today's Breakout Check ---
    today = df.iloc[-1]
    yesterday = df.iloc[-2] if len(df) >= 2 else today
    volume_ratio = today['Volume'] / dry_avg_vol
    
    # Condition a) Volume surge
    volume_surge_ok = volume_ratio >= min_volume_ratio
    # Condition b) Bullish candle close is higher than open OR it is a massive gap-up continuation
    bullish_candle_ok = today['Close'] > today['Open'] or today['Close'] > yesterday['Close']
    # Condition c) Price breakout of at least min_price_change (either Close-to-Close or Intraday Close-to-Open)
    pct_change_intraday = (today['Close'] - today['Open']) / today['Open'] * 100
    pct_change_close = ((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if len(df) >= 2 else pct_change_intraday
    price_change_ok = (pct_change_intraday >= min_price_change) or (pct_change_close >= min_price_change)
    
    if not (volume_surge_ok and bullish_candle_ok and price_change_ok):
        return None
        
    # --- STEP 4: Signal Strength Score (0 to 100) & Indicators ---
    df_indicators = df.copy()
    df_indicators['MA50'] = df_indicators['Close'].rolling(window=50).mean()
    
    today_ma = df_indicators['MA50'].iloc[-1]
    above_50dma = False
    if not pd.isna(today_ma):
        above_50dma = today['Close'] > today_ma
        
    df_indicators['MA200'] = df_indicators['Close'].rolling(window=200).mean()
    today_ma200 = df_indicators['MA200'].iloc[-1]
    above_200dma = False
    if not pd.isna(today_ma200):
        above_200dma = today['Close'] > today_ma200
        
    # Calculate score weights
    score = 0.0
    # 1. Volume ratio: Up to 40 points
    score += min(volume_ratio / 10.0 * 40.0, 40.0)
    # 2. Price move: Up to 30 points
    score += min(max(pct_change_intraday, pct_change_close) / 5.0 * 30.0, 30.0)
    # 3. Dry duration: Up to 20 points
    score += min(dry_days_count / 60.0 * 20.0, 20.0)
    # 4. Moving Average filter: 10 points
    if above_50dma:
        score += 10.0
        
    score = round(min(score, 100.0), 1)
    
    # Calculate day-over-day price change (standard Close-to-Close change)
    day_change_pct = pct_change_close
    
    # Calculate available 120-day high and low
    high_120d = float(df['High'].max())
    low_120d = float(df['Low'].min())
    
    # Advanced Trading Setup Calculations:
    buy_price = round(float(today['Close']), 2)
    # Swing low stop loss: lowest price of the last 5 days (pullback anchor) minus 2% buffer
    min_5d_low = float(df['Low'].iloc[-5:].min())
    exit_price = round(min(buy_price * 0.95, min_5d_low * 0.98), 2)
    # Target: 15% swing trade objective
    target_price = round(buy_price * 1.15, 2)
    
    # Confidence text based on algorithmic signal strength
    if score >= 75:
        confidence = f"High Conviction ({score}%)"
    elif score >= 50:
        confidence = f"Medium-High ({score}%)"
    else:
        confidence = f"Medium ({score}%)"
        
    base_rec = (
        f"Strong institutional VDU breakout! Volume ratio is {volume_ratio:.1f}x with signal score {score}%. "
        f"Buy around CMP ₹{buy_price:.2f}. Set stop loss at swing low ₹{exit_price:.2f} (risk {(buy_price-exit_price)/buy_price*100:.1f}%) "
        f"with a target of ₹{target_price:.2f} (potential +15.0%)."
    )
    recommendation = compute_rich_analysis(df_indicators, symbol, "VDU Breakout", base_rec)
    
    from config import get_company_name
    company_name = get_company_name(symbol)
    
    return {
        "symbol": symbol.strip().upper(),
        "company_name": company_name,
        "cmp": buy_price,
        "day_change_pct": round(day_change_pct, 2),
        "today_volume": int(today['Volume']),
        "dry_avg_vol": round(dry_avg_vol, 1),
        "volume_ratio": round(volume_ratio, 2),
        "dry_days_count": int(dry_days_count),
        "dry_spikes": dry_spikes,
        "dry_start_date": pd.to_datetime(history_df['Date'].iloc[start_idx]),
        "dry_end_date": pd.to_datetime(history_df['Date'].iloc[end_idx]),
        "signal_strength": score,
        "pct_change_today": round(day_change_pct, 2),
        "above_50dma": above_50dma,
        "above_200dma": above_200dma,
        "df": df_indicators,
        "high_52w": high_120d,      
        "low_52w": low_120d,
        "buy_price": buy_price,
        "exit_price": exit_price,
        "target_price": target_price,
        "confidence": confidence,
        "recommendation": recommendation
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
    
    # Advanced Trading Setup Calculations for Coiled Spring VCP Squeeze
    buy_price = round(float(today['Close']), 2)
    # Tight VCP stop loss at 4%
    exit_price = round(buy_price * 0.96, 2)
    # 15% target
    target_price = round(buy_price * 1.15, 2)
    
    if squeeze_score >= 75:
        confidence = f"High ({squeeze_score}%)"
    else:
        confidence = f"Medium-High ({squeeze_score}%)"
        
    base_rec = (
        f"Extremely tight volatility contraction (VCP) squeeze score of {squeeze_score}%. "
        f"Buy at CMP ₹{buy_price:.2f} with a tight stop loss just below contraction support at ₹{exit_price:.2f} "
        f"(risk 4.0%) and look for a strong breakout toward target ₹{target_price:.2f} (+15.0%)."
    )
    recommendation = compute_rich_analysis(df_copy, symbol, "VCP Squeeze", base_rec)

    from config import get_company_name
    company_name = get_company_name(symbol)
    
    return {
        "symbol": symbol.strip().upper(),
        "company_name": company_name,
        "cmp": buy_price,
        "day_change_pct": round(day_change_pct, 2),
        "range_5d": round(range_a, 2),
        "range_prev": round(range_b, 2),
        "avg_vol_5d": round(avg_vol_a, 1),
        "baseline_avg_vol": round(baseline_avg_vol, 1),
        "vol_ratio": round(avg_vol_a / baseline_avg_vol, 2),
        "squeeze_score": squeeze_score,
        "above_20ema": today['Close'] > today['EMA20'],
        "df": df_copy,
        "buy_price": buy_price,
        "exit_price": exit_price,
        "target_price": target_price,
        "confidence": confidence,
        "recommendation": recommendation
    }

def scan_wt_cross(symbol: str, df: pd.DataFrame, wt_oversold_threshold: float = -40.0) -> dict | None:
    """
    Scans a stock's history to calculate WaveTrend with Crosses [LazyBear] (WT_CROSS_LB).
    Returns WT details if wt1 <= wt_oversold_threshold (oversold zone).
    Also detects bullish buy signal (green dot): wt1 crosses above wt2 from oversold zone.
    
    Mathematical details:
    ap = Typical Price = (High + Low + Close) / 3
    esa = EMA(ap, n1=10)
    d = EMA(abs(ap - esa), n1=10)
    ci = (ap - esa) / (0.015 * d)
    wt1 = EMA(ci, n2=21)
    wt2 = SMA(wt1, 4)
    
    Buy signal (green dot): wt1 crosses above wt2 while in oversold zone (wt2 <= wt_oversold_threshold)
    """
    if df is None or len(df) < 40:
        return None
        
    df_copy = df.copy()
    
    # Typical price: hlc3
    ap = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3.0
    
    # esa = ema(ap, 10)
    esa = ap.ewm(span=10, adjust=False).mean()
    
    # d = ema(abs(ap - esa), 10)
    d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
    
    # ci = (ap - esa) / (0.015 * d)
    ci = (ap - esa) / (0.015 * d + 1e-10)
    
    # wt1 = tci = ema(ci, 21)
    wt1 = ci.ewm(span=21, adjust=False).mean()
    
    # wt2 = sma(wt1, 4)
    wt2 = wt1.rolling(window=4).mean()
    
    today_wt1 = wt1.iloc[-1]
    today_wt2 = wt2.iloc[-1]
    
    if pd.isna(today_wt1) or pd.isna(today_wt2):
        return None
        
    # Check if wt1 is in oversold zone (below threshold)
    if today_wt1 > wt_oversold_threshold:
        return None
    
    today = df_copy.iloc[-1]
    yesterday = df_copy.iloc[-2] if len(df_copy) >= 2 else today
    day_change_pct = ((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if len(df_copy) >= 2 else 0.0
    
    # Detect buy signal (green dot): wt1 crosses above wt2
    # We check if a bullish crossover (green dot) occurred recently (today or within the last 3 days/bars)
    # to highlight stocks that are in active buying stages.
    buy_signal = False
    for offset in range(1, 4):
        if len(wt1) > offset and len(wt2) > offset:
            t_wt1 = wt1.iloc[-offset]
            t_wt2 = wt2.iloc[-offset]
            p_wt1 = wt1.iloc[-offset - 1]
            p_wt2 = wt2.iloc[-offset - 1]
            if not pd.isna(t_wt1) and not pd.isna(t_wt2) and not pd.isna(p_wt1) and not pd.isna(p_wt2):
                if (p_wt1 <= p_wt2) and (t_wt1 > t_wt2):
                    buy_signal = True
                    break
    
    # Advanced Trading Setup Calculations for WaveTrend
    buy_price = round(float(today['Close']), 2)
    exit_price = round(buy_price * 0.95, 2)
    target_price = round(buy_price * 1.12, 2)
    
    # Calculate SMAs to see if price is trading above 20 SMA & 50 SMA
    df_copy['SMA20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA50'] = df_copy['Close'].rolling(window=50).mean()
    today_sma20 = df_copy['SMA20'].iloc[-1]
    today_sma50 = df_copy['SMA50'].iloc[-1]
    above_20sma = bool(buy_price > today_sma20) if not pd.isna(today_sma20) else False
    above_50sma = bool(buy_price > today_sma50) if not pd.isna(today_sma50) else False

    if buy_signal:
        confidence = "High (WT Buy Signal)"
        base_rec = (
            f"Bullish mean-reversion buy signal (LazyBear Green Dot) triggered in oversold zone! "
            f"WT1 is {today_wt1:.1f} and crossed above WT2. Buy around CMP ₹{buy_price:.2f}. "
            f"Place stop loss at ₹{exit_price:.2f} (risk 5.0%) with a target bounce at ₹{target_price:.2f} (+12.0%)."
        )
    else:
        confidence = "Medium (WT Oversold)"
        base_rec = (
            f"Stock is in a deep WaveTrend oversold zone (WT1 = {today_wt1:.1f} is below {wt_oversold_threshold}). "
            f"No green dot cross yet, but prime for accumulation. Buy on pullbacks near ₹{buy_price:.2f} "
            f"with stop loss at ₹{exit_price:.2f} and target bounce of ₹{target_price:.2f}."
        )
    recommendation = compute_rich_analysis(df_copy, symbol, "WaveTrend Cross", base_rec)

    from config import get_company_name
    company_name = get_company_name(symbol)
    
    return {
        "symbol": symbol.strip().upper(),
        "company_name": company_name,
        "cmp": buy_price,
        "day_change_pct": round(day_change_pct, 2),
        "wt_value": round(today_wt1, 2),
        "wt2_value": round(today_wt2, 2),
        "buy_signal": buy_signal,
        "wt_diff": round(today_wt1 - today_wt2, 2),
        "volume": int(today['Volume']),
        "buy_price": buy_price,
        "exit_price": exit_price,
        "target_price": target_price,
        "confidence": confidence,
        "recommendation": recommendation,
        "above_20sma": above_20sma,
        "above_50sma": above_50sma
    }

def compute_rich_analysis(df, symbol, strategy_name, base_rec_text):
    """
    Computes CCI, RSI, EMA, SMA and checklist triggers, 
    and returns a structured JSON recommendation string.
    """
    if df is None or len(df) < 14:
        return base_rec_text
        
    try:
        import pandas as pd
        import json
        
        # Standard indicators
        close_series = df['Close']
        high_series = df['High']
        low_series = df['Low']
        
        # 1. RSI (14)
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / (ema_loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = float(rsi_series.iloc[-1])
        
        # 2. CCI (14)
        tp = (high_series + low_series + close_series) / 3
        sma_tp = tp.rolling(window=14).mean()
        mad_manual = tp.rolling(window=14).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        cci_series = (tp - sma_tp) / (0.015 * mad_manual + 1e-9)
        cci_val = float(cci_series.iloc[-1])
        
        # 3. Moving Averages
        ema20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
        sma50 = float(close_series.rolling(window=50).mean().iloc[-1]) if len(df) >= 50 else float(close_series.iloc[-1])
        sma200 = float(close_series.rolling(window=200).mean().iloc[-1]) if len(df) >= 200 else float(close_series.iloc[-1])
        cmp = float(close_series.iloc[-1])
        
        # 4. Interpretations & Status
        if rsi_val >= 70:
            rsi_status = "Overbought"
            rsi_interp = "Extremely strong bullish momentum; check for near-term exhaustion."
        elif rsi_val >= 50:
            rsi_status = "Bullish Momentum"
            rsi_interp = "Solid buying interest with strong upward price power."
        elif rsi_val >= 35:
            rsi_status = "Neutral Consolidation"
            rsi_interp = "Price is stable, hovering within a sideways range."
        else:
            rsi_status = "Oversold Bounce"
            rsi_interp = "Deep oversold territory; highly prime for a technical recovery."
            
        if cci_val >= 100:
            cci_status = "Bullish Breakout"
            cci_interp = "Price velocity is in a strong upward breakout phase."
        elif cci_val >= 0:
            cci_status = "Positive Territory"
            cci_interp = "Short-term momentum is supportive of higher prices."
        elif cci_val >= -100:
            cci_status = "Weak Momentum"
            cci_interp = "Trading in bearish territory, under consolidation."
        else:
            cci_status = "Extremely Oversold"
            cci_interp = "Deep oversold condition; expect a sharp mean-reversion move."
            
        # Triggers Checklist
        triggers = []
        if cmp > sma50 and sma50 > sma200:
            triggers.append("✔️ Golden Trend: Price is supported by long-term institutional SMA structure.")
        elif cmp > sma50:
            triggers.append("✔️ Mid-Term Bullish: Trading comfortably above the 50-day moving average.")
            
        if rsi_val >= 50 and rsi_val <= 68:
            triggers.append("✔️ RSI Sweet Spot: Healthy buying momentum without being overextended.")
        elif rsi_val < 35:
            triggers.append("✔️ Reversion Alert: Deeply discounted oversold price levels ready to bounce.")
            
        if cci_val > 100:
            triggers.append("✔️ Momentum Surge: CCI confirms active breakout velocity is backing the move.")
        elif cci_val < -100:
            triggers.append("✔️ Oversold Stretch: CCI indicates institutional selling is exhausted.")
            
        if cmp >= ema20 * 0.98 and cmp <= ema20 * 1.02:
            triggers.append("✔️ Dynamic Pullback: Price is pulling back perfectly into the 20-day EMA support anchor.")
        elif cmp > ema20:
            triggers.append("✔️ Fast Trend: Sustained buying velocity holding above the short-term 20 EMA.")
            
        if not triggers:
            triggers.append("✔️ Value Consolidation: Standard technical entry on strategy parameters.")
            
        # Determine the price position relative to MAs dynamically
        ma_reasoning = ""
        if cmp > ema20 and cmp > sma50:
            ma_reasoning = f"Price holds strong above short-term 20 EMA (₹{ema20:,.2f}) and medium-term 50 SMA (₹{sma50:,.2f}), confirming dynamic uptrend support."
        elif cmp > ema20:
            ma_reasoning = f"Price is trading above the fast-moving 20 EMA (₹{ema20:,.2f}), while testing major resistance/consolidation zones."
        elif cmp > sma50:
            ma_reasoning = f"Price holds above structural 50 SMA (₹{sma50:,.2f}) institutional support, despite a short-term dip below 20 EMA."
        else:
            ma_reasoning = f"Price is testing critical structural zones near the 50 SMA (₹{sma50:,.2f}) and 200 SMA (₹{sma200:,.2f}) floors."

        # RSI Momentum reasoning
        rsi_reasoning = ""
        if rsi_val >= 70:
            rsi_reasoning = f"RSI is extremely strong at {rsi_val:.1f} ({rsi_status}), showing high bullish momentum."
        elif rsi_val >= 50:
            rsi_reasoning = f"RSI at {rsi_val:.1f} ({rsi_status}) indicates active buying interest and healthy momentum."
        elif rsi_val >= 35:
            rsi_reasoning = f"RSI at {rsi_val:.1f} ({rsi_status}) indicates a healthy cooling consolidation zone."
        else:
            rsi_reasoning = f"RSI at {rsi_val:.1f} ({rsi_status}) indicates deeply oversold levels, highly prime for a technical bounce."

        # CCI momentum velocity reasoning
        cci_reasoning = ""
        if cci_val >= 100:
            cci_reasoning = f"CCI is at {cci_val:.1f} ({cci_status}), confirming an active breakout phase."
        elif cci_val >= 0:
            cci_reasoning = f"CCI at {cci_val:.1f} ({cci_status}) indicates supportive short-term momentum."
        elif cci_val >= -100:
            cci_reasoning = f"CCI is at {cci_val:.1f} ({cci_status}), reflecting temporary sideways consolidation."
        else:
            cci_reasoning = f"CCI at {cci_val:.1f} ({cci_status}) shows extreme selling exhaustion, signaling an upward pivot."

        # Extract execution parameters (Stop Loss and Target) from base_rec_text
        sl_target_part = ""
        if "Set stop loss" in base_rec_text:
            sl_target_part = base_rec_text[base_rec_text.index("Set stop loss"):]
        elif "Place stop loss" in base_rec_text:
            sl_target_part = base_rec_text[base_rec_text.index("Place stop loss"):]
        elif "stop loss" in base_rec_text.lower():
            try:
                idx = base_rec_text.lower().index("stop loss")
                sl_target_part = base_rec_text[idx-4:]
            except ValueError:
                sl_target_part = base_rec_text
        else:
            sl_target_part = base_rec_text

        final_text = f"{rsi_reasoning} {cci_reasoning} {ma_reasoning} {sl_target_part}"

        analysis_payload = {
            "is_rich": True,
            "text": final_text,
            "rsi": round(rsi_val, 1),
            "rsi_status": rsi_status,
            "rsi_interp": rsi_interp,
            "cci": round(cci_val, 1),
            "cci_status": cci_status,
            "cci_interp": cci_interp,
            "ema20": round(ema20, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "triggers": triggers
        }
        return json.dumps(analysis_payload)
    except Exception as e:
        print(f"Error computing rich technical analysis for {symbol}: {e}")
        return base_rec_text


def scan_monthly_momentum(symbol: str, df_monthly: pd.DataFrame, market_cap_cr: float = 0.0) -> dict | None:
    """
    Scans monthly OHLCV data for the Chartink-style Monthly Momentum Filter:

    Conditions (all on Monthly timeframe):
    1. Close > EMA(Close, 8)
    2. EMA(Close, 8) > EMA(Close, 12)
    3. EMA(Close, 12) > EMA(Close, 20)
    4. ROC(6) > 20   [Rate of Change over 6 months > 20%]
    5. ROC(6) <= 80  [Not overextended — <= 80%]
    6. RSI(14) > 55
    7. RSI(14) < 85
    8. Volume > SMA(Volume, 12)
    9. Market Cap >= 3000 Crore
    10. Close >= 100 (price filter)

    Returns a result dict if ALL conditions are met, else None.
    """
    if df_monthly is None or len(df_monthly) < 22:
        # Need at least 22 monthly bars for EMA(20) + RSI(14) to be stable
        return None

    try:
        close = df_monthly['Close']
        volume = df_monthly['Volume']

        # ---- Price filter ----
        cmp = float(close.iloc[-1])
        if cmp < 100.0:
            return None

        # Condition 0: CMP must be breaking upward from previous month's closing price
        prev_month_close = float(close.iloc[-2]) if len(close) >= 2 else cmp
        if not (cmp > prev_month_close):
            return None

        # Condition 0.5: Current monthly candle must be green (CMP > Monthly Open)
        monthly_open = float(df_monthly['Open'].iloc[-1])
        if not (cmp > monthly_open):
            return None

        # ---- Market Cap filter ----
        if market_cap_cr > 0 and market_cap_cr < 3000.0:
            return None

        # ---- EMA calculations ----
        ema8  = close.ewm(span=8,  adjust=False).mean()
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()

        ema8_val  = float(ema8.iloc[-1])
        ema12_val = float(ema12.iloc[-1])
        ema20_val = float(ema20.iloc[-1])

        # Condition 1: Close > EMA8
        if not (cmp > ema8_val):
            return None
        # Condition 2: EMA8 > EMA12
        if not (ema8_val > ema12_val):
            return None
        # Condition 3: EMA12 > EMA20
        if not (ema12_val > ema20_val):
            return None

        # ---- ROC (6 months) ----
        if len(close) < 7:
            return None
        close_6m_ago = float(close.iloc[-7])
        if close_6m_ago <= 0:
            return None
        roc6 = (cmp - close_6m_ago) / close_6m_ago * 100.0

        # Condition 4: ROC > 20
        if not (roc6 > 20.0):
            return None
        # Condition 5: ROC <= 80
        if not (roc6 <= 80.0):
            return None

        # ---- RSI (14 monthly bars) ----
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = float(rsi_series.iloc[-1])

        # Condition 6: RSI > 55
        if not (rsi_val > 55.0):
            return None
        # Condition 7: RSI < 85
        if not (rsi_val < 85.0):
            return None

        # ---- Volume > SMA(Volume, 12) ----
        if len(volume) < 12:
            return None
        vol_sma12 = float(volume.rolling(window=12).mean().iloc[-1])
        curr_vol  = float(volume.iloc[-1])

        # Condition 8: Volume > SMA(Volume, 12)
        if not (curr_vol > vol_sma12):
            return None

        # ---- All conditions passed — compute trade setup ----
        # Monthly ATR-based stop: use last candle's Low as stop anchor
        last_low = float(df_monthly['Low'].iloc[-1])
        buy_price  = round(cmp, 2)
        exit_price = round(last_low * 0.97, 2)   # 3% below monthly low
        target_price = round(cmp * 1.20, 2)       # 20% swing target (1-2 months)

        # Score based on quality of alignment
        score = 0.0
        # EMA stacking tightness (the tighter, the earlier the entry)
        ema_gap_pct = (ema8_val - ema20_val) / ema20_val * 100
        score += min(ema_gap_pct / 5.0 * 30.0, 30.0)  # up to 30 pts
        # ROC momentum (mid-range is best)
        roc_score = 30.0 - abs(roc6 - 35.0) / 35.0 * 30.0
        score += max(0.0, roc_score)                    # up to 30 pts
        # RSI in sweet spot
        if 60.0 <= rsi_val <= 75.0:
            score += 25.0
        elif 55.0 < rsi_val < 85.0:
            score += 15.0
        # Volume surge bonus
        if curr_vol > vol_sma12 * 1.5:
            score += 15.0
        else:
            score += 8.0
        score = round(min(score, 100.0), 1)

        if score >= 70:
            confidence = f"High ({score}%)"
        elif score >= 50:
            confidence = f"Medium-High ({score}%)"
        else:
            confidence = f"Medium ({score}%)"

        prev_close = float(close.iloc[-2]) if len(close) >= 2 else cmp
        day_change_pct = round((cmp - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0.0

        base_rec = (
            f"Monthly EMA Stack (EMA8 > EMA12 > EMA20) confirmed bullish alignment. "
            f"ROC(6M) = {roc6:.1f}% (healthy momentum), RSI(14M) = {rsi_val:.1f} (non-overbought). "
            f"Volume breakout above 12M SMA confirms institutional participation. "
            f"Buy near ₹{buy_price:.2f}, stop at monthly low ₹{exit_price:.2f}, target ₹{target_price:.2f} (+20%)."
        )
        recommendation = compute_rich_analysis(df_monthly, symbol, "Monthly EMA Momentum", base_rec)

        from config import get_company_name
        company_name = get_company_name(symbol)

        return {
            "symbol": symbol.strip().upper(),
            "company_name": company_name,
            "cmp": buy_price,
            "day_change_pct": day_change_pct,
            "ema8": round(ema8_val, 2),
            "ema12": round(ema12_val, 2),
            "ema20": round(ema20_val, 2),
            "roc6": round(roc6, 2),
            "rsi_monthly": round(rsi_val, 2),
            "volume": int(curr_vol),
            "vol_sma12": round(vol_sma12, 0),
            "market_cap_cr": round(market_cap_cr, 1),
            "momentum_score": score,
            "buy_price": buy_price,
            "exit_price": exit_price,
            "target_price": target_price,
            "confidence": confidence,
            "recommendation": recommendation,
            "return_1m": day_change_pct,
        }

    except Exception as e:
        print(f"Monthly momentum scan error for {symbol}: {e}")
        return None


def scan_weekly_momentum(symbol: str, df_weekly: pd.DataFrame, market_cap_cr: float = 0.0) -> dict | None:
    """
    Scans weekly OHLCV data for the Chartink-style Weekly Momentum Breakout Filter.

    All conditions on WEEKLY candles:
    1.  Weekly Volume > SMA(Volume, 20)
    2.  Weekly Close > 200
    3.  Weekly Close > Previous Week Close
    4.  Weekly Open  > Previous Week Close   (gap-up / strong open)
    5.  Weekly CCI(20) > 90
    6.  Market Cap > 5000 Crore
    7.  Weekly RSI(14) > 60
    8.  Weekly Close > SMA(Close, 20)
    """
    if df_weekly is None or len(df_weekly) < 22:
        return None

    try:
        close  = df_weekly['Close'].reset_index(drop=True)
        open_  = df_weekly['Open'].reset_index(drop=True)
        high   = df_weekly['High'].reset_index(drop=True)
        low    = df_weekly['Low'].reset_index(drop=True)
        volume = df_weekly['Volume'].reset_index(drop=True)

        cmp        = float(close.iloc[-1])
        curr_open  = float(open_.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else cmp

        # Condition 2: Price > 200
        if cmp < 200.0:
            return None

        # Condition 6: Market Cap > 5000 Cr
        if market_cap_cr > 0 and market_cap_cr < 5000.0:
            return None

        # Condition 3: Close > Previous Week Close
        if not (cmp > prev_close):
            return None

        # Condition 4: Open > Previous Week Close
        if not (curr_open > prev_close):
            return None

        # Condition 1: Volume > SMA(Volume, 20)
        if len(volume) < 20:
            return None
        vol_sma20 = float(volume.rolling(window=20).mean().iloc[-1])
        curr_vol  = float(volume.iloc[-1])
        if not (curr_vol > vol_sma20):
            return None

        # Condition 8: Close > SMA(Close, 20)
        if len(close) < 20:
            return None
        close_sma20 = float(close.rolling(window=20).mean().iloc[-1])
        if not (cmp > close_sma20):
            return None

        # Condition 7: RSI(14) > 60
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        rsi_val  = float((100 - (100 / (1 + rs))).iloc[-1])
        if not (rsi_val > 60.0):
            return None

        # Condition 5: CCI(20) > 90
        tp       = (high + low + close) / 3.0
        tp_sma20 = tp.rolling(window=20).mean()
        mad20    = tp.rolling(window=20).apply(lambda x: (abs(x - x.mean())).mean(), raw=True)
        cci_val  = float(((tp - tp_sma20) / (0.015 * (mad20 + 1e-9))).iloc[-1])
        if not (cci_val > 90.0):
            return None

        # ---- All conditions passed — trade setup ----
        buy_price    = round(cmp, 2)
        last_low     = float(low.iloc[-1])
        exit_price   = round(min(last_low * 0.98, cmp * 0.95), 2)
        target_price = round(cmp * 1.15, 2)

        # 1-Month Return calculation (from 4 weeks ago)
        close_4w_ago = float(close.iloc[-5]) if len(close) >= 5 else float(close.iloc[0])
        return_1m = round((cmp - close_4w_ago) / close_4w_ago * 100, 2) if close_4w_ago > 0 else 0.0

        vol_ratio = curr_vol / vol_sma20 if vol_sma20 > 0 else 1.0

        score = 0.0
        if 65 <= rsi_val <= 80:
            score += 30.0
        elif 60 < rsi_val <= 85:
            score += 20.0
        if cci_val >= 150:
            score += 25.0
        elif cci_val >= 100:
            score += 18.0
        else:
            score += 10.0
        if vol_ratio >= 2.0:
            score += 25.0
        elif vol_ratio >= 1.5:
            score += 18.0
        else:
            score += 10.0
        dist_pct = (cmp - close_sma20) / close_sma20 * 100
        if dist_pct <= 5:
            score += 20.0
        elif dist_pct <= 10:
            score += 12.0
        else:
            score += 5.0
        score = round(min(score, 100.0), 1)

        confidence = f"High ({score}%)" if score >= 75 else f"Medium-High ({score}%)" if score >= 55 else f"Medium ({score}%)"
        weekly_chg_pct = round((cmp - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0.0

        base_rec = (
            f"Weekly Momentum Breakout: Price ₹{cmp:.2f} above 20W SMA (₹{close_sma20:.2f}). "
            f"Volume {vol_ratio:.2f}x 20W avg confirms institutional participation. "
            f"RSI(14W)={rsi_val:.1f}, CCI(20W)={cci_val:.1f} — strong bullish momentum. "
            f"Open (₹{curr_open:.2f}) > Last Week Close (₹{prev_close:.2f}) = gap-up continuation. "
            f"Buy ₹{buy_price:.2f} | Stop ₹{exit_price:.2f} | Target ₹{target_price:.2f} (+15%)."
        )
        recommendation = compute_rich_analysis(df_weekly, symbol, "Weekly Momentum Breakout", base_rec)

        from config import get_company_name
        company_name = get_company_name(symbol)

        return {
            "symbol": symbol.strip().upper(),
            "company_name": company_name,
            "cmp": buy_price,
            "weekly_chg_pct": weekly_chg_pct,
            "prev_close": round(prev_close, 2),
            "curr_open": round(curr_open, 2),
            "close_sma20": round(close_sma20, 2),
            "rsi_weekly": round(rsi_val, 2),
            "cci_weekly": round(cci_val, 2),
            "volume": int(curr_vol),
            "vol_sma20": round(vol_sma20, 0),
            "vol_ratio": round(vol_ratio, 2),
            "market_cap_cr": round(market_cap_cr, 1),
            "weekly_score": score,
            "buy_price": buy_price,
            "exit_price": exit_price,
            "target_price": target_price,
            "confidence": confidence,
            "recommendation": recommendation,
            "return_1m": return_1m,
        }

    except Exception as e:
        print(f"Weekly momentum scan error for {symbol}: {e}")
        return None

def run_monthly_momentum_update(base_date_str: str, today_str: str) -> list[dict]:
    """
    Retrieves the locked stock list from base_date_str, fetches their current CMP,
    calculates updated return percentages since base_date_str, and preserves all other fields.
    """
    import database
    import yfinance as yf
    
    # 1. Fetch the original base records from database
    base_results = database.get_cached_monthly_momentum(base_date_str)
    if not base_results:
        return []
        
    symbols = [r['symbol'] for r in base_results]
    tickers = [f"{s}.NS" for s in symbols]
    
    # 2. Batch download today's daily quotes to get current CMP and daily price changes
    cmp_map = {}
    prev_close_map = {}
    try:
        quotes_df = yf.download(tickers=tickers, period="1d", progress=False, threads=False)
        if not quotes_df.empty:
            if isinstance(quotes_df.columns, pd.MultiIndex):
                # multi-ticker Close and Open
                for tk in tickers:
                    sym_clean = tk.replace(".NS", "").upper()
                    try:
                        close_series = quotes_df['Close'][tk] if 'Close' in quotes_df else None
                        open_series = quotes_df['Open'][tk] if 'Open' in quotes_df else None
                        if close_series is not None and not close_series.empty:
                            cmp_map[sym_clean] = float(close_series.iloc[-1])
                        if open_series is not None and not open_series.empty:
                            prev_close_map[sym_clean] = float(open_series.iloc[-1])
                    except Exception:
                        pass
            else:
                # single ticker
                sym_clean = symbols[0]
                cmp_map[sym_clean] = float(quotes_df['Close'].iloc[-1])
                prev_close_map[sym_clean] = float(quotes_df['Open'].iloc[-1]) if 'Open' in quotes_df else float(quotes_df['Close'].iloc[-1])
    except Exception as e:
        print(f"Error fetching real-time updates for monthly momentum stocks: {e}")
        
    # 3. Update the return_1m based on the base price
    updated_results = []
    for r in base_results:
        sym = r['symbol']
        curr_cmp = cmp_map.get(sym, r['cmp'])  # fallback to base CMP if fetch fails
        base_cmp = r['cmp']
        
        # Calculate monthly return %age since base price (purchase price)
        if base_cmp > 0:
            return_1m = (curr_cmp - base_cmp) / base_cmp * 100
        else:
            return_1m = 0.0
            
        # Daily change %age
        prev_close = prev_close_map.get(sym, curr_cmp)
        day_change_pct = (curr_cmp - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        
        # Build updated record - KEEP all indicators & setups exactly the same
        updated_record = dict(r)
        updated_record['cmp'] = round(curr_cmp, 2)
        updated_record['return_1m'] = round(return_1m, 2)
        updated_record['day_change_pct'] = round(day_change_pct, 2)
        updated_record['scan_date'] = today_str
        updated_results.append(updated_record)
        
    return updated_results

def run_weekly_momentum_update(base_date_str: str, today_str: str) -> list[dict]:
    """
    Retrieves the locked stock list from base_date_str, fetches their current CMP,
    calculates updated weekly and monthly return percentages, and preserves all other fields.
    """
    import database
    import yfinance as yf
    
    # 1. Fetch the original base records from database
    base_results = database.get_cached_weekly_momentum(base_date_str)
    if not base_results:
        return []
        
    symbols = [r['symbol'] for r in base_results]
    tickers = [f"{s}.NS" for s in symbols]
    
    # 2. Batch download today's daily quotes to get current CMP
    cmp_map = {}
    try:
        quotes_df = yf.download(tickers=tickers, period="1d", progress=False, threads=False)
        if not quotes_df.empty:
            if isinstance(quotes_df.columns, pd.MultiIndex):
                for tk in tickers:
                    sym_clean = tk.replace(".NS", "").upper()
                    try:
                        close_series = quotes_df['Close'][tk] if 'Close' in quotes_df else None
                        if close_series is not None and not close_series.empty:
                            cmp_map[sym_clean] = float(close_series.iloc[-1])
                    except Exception:
                        pass
            else:
                sym_clean = symbols[0]
                cmp_map[sym_clean] = float(quotes_df['Close'].iloc[-1])
    except Exception as e:
        print(f"Error fetching real-time updates for weekly momentum stocks: {e}")
        
    # 3. Update the return_1m based on the base price
    updated_results = []
    for r in base_results:
        sym = r['symbol']
        curr_cmp = cmp_map.get(sym, r['cmp'])  # fallback to base CMP if fetch fails
        base_cmp = r['cmp']
        
        # Calculate return %age since base price (purchase price)
        if base_cmp > 0:
            return_since_base = (curr_cmp - base_cmp) / base_cmp * 100
        else:
            return_since_base = 0.0
            
        # Standard weekly return since prev week close
        prev_close_val = r.get('prev_close') or base_cmp
        if prev_close_val > 0:
            weekly_chg_pct = (curr_cmp - prev_close_val) / prev_close_val * 100
        else:
            weekly_chg_pct = 0.0
            
        # Build updated record - KEEP all indicators & setups exactly the same
        updated_record = dict(r)
        updated_record['cmp'] = round(curr_cmp, 2)
        updated_record['return_1m'] = round(return_since_base, 2)  # User actual trade return since purchase
        updated_record['weekly_chg_pct'] = round(weekly_chg_pct, 2)
        updated_record['scan_date'] = today_str
        updated_results.append(updated_record)
        
    return updated_results

def scan_vcs(symbol: str, df: pd.DataFrame, lenShort=13, lenLong=63, lenVol=50, sensitivity=2.0, max_score=10.0) -> dict | None:
    """
    Scans a stock for Volatility Contraction (VCS) based on ATR, StdDev, and Volume contraction.
    """
    if df is None or len(df) < max(lenLong, lenVol) + 2:
        return None

    try:
        df_copy = df.copy()

        df_copy["prev_close"] = df_copy["Close"].shift(1)

        tr1 = df_copy["High"] - df_copy["Low"]
        tr2 = (df_copy["High"] - df_copy["prev_close"]).abs()
        tr3 = (df_copy["Low"] - df_copy["prev_close"]).abs()

        df_copy["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR contraction (Normalized by price)
        df_copy["TR_pct"] = df_copy["TR"] / df_copy["prev_close"].replace(0, pd.NA)
        df_copy["trShort"] = df_copy["TR_pct"].rolling(lenShort).mean()
        df_copy["trLong"] = df_copy["TR_pct"].rolling(lenLong).mean()
        df_copy["ratioATR"] = df_copy["trShort"] / df_copy["trLong"]

        # STANDARD DEVIATION CONTRACTION (Using daily returns)
        df_copy["ret"] = df_copy["Close"].pct_change()
        df_copy["stdShort"] = df_copy["ret"].rolling(lenShort).std()
        df_copy["stdLong"] = df_copy["ret"].rolling(lenLong).std()
        df_copy["ratioStd"] = df_copy["stdShort"] / df_copy["stdLong"]

        # VOLUME CONTRACTION
        df_copy["volAvg"] = df_copy["Volume"].rolling(lenVol).mean()
        df_copy["volShort"] = df_copy["Volume"].rolling(5).mean()
        df_copy["volRatio"] = df_copy["volShort"] / df_copy["volAvg"]

        # TREND FILTER (Minervini VCP requires an uptrend)
        df_copy["sma50"] = df_copy["Close"].rolling(50).mean()
        df_copy["sma200"] = df_copy["Close"].rolling(200).mean()

        # SCORE CALCULATION
        df_copy["s_atr"] = df_copy["ratioATR"] * sensitivity
        df_copy["s_std"] = df_copy["ratioStd"] * sensitivity
        df_copy["s_vol"] = df_copy["volRatio"]

        df_copy["rawScore"] = (
            df_copy["s_atr"] * 0.4 +
            df_copy["s_std"] * 0.4 +
            df_copy["s_vol"] * 0.2
        )

        df_copy["finalScore"] = df_copy["rawScore"] * 10.0
        
        today_score = df_copy["finalScore"].iloc[-1]
        
        if pd.isna(today_score):
            return None
            
        today_close = df_copy["Close"].iloc[-1]
        today_sma50 = df_copy["sma50"].iloc[-1]
        today_sma200 = df_copy["sma200"].iloc[-1]
        today_vol_avg = df_copy["volAvg"].iloc[-1]
        
        # Ensure stock is in a solid uptrend and has reasonable liquidity
        if pd.isna(today_sma200) or today_close < today_sma50 or today_close < today_sma200 or today_sma50 < today_sma200 or today_vol_avg < 50000:
            return None

        if today_score <= max_score:
            today = df_copy.iloc[-1]
            yesterday = df_copy.iloc[-2] if len(df_copy) >= 2 else today
            day_change_pct = ((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if len(df_copy) >= 2 else 0.0
            
            buy_price = round(float(today['Close']), 2)
            exit_price = round(buy_price * 0.95, 2)
            target_price = round(buy_price * 1.15, 2)
            
            from config import get_company_name
            company_name = get_company_name(symbol)
            
            confidence = "High" if today_score < 5 else "Medium"
            
            base_rec = f"VCS final score is {today_score:.2f} (below {max_score}), indicating strict setup conditions met. Buy near ₹{buy_price:.2f}, SL ₹{exit_price:.2f}, Target ₹{target_price:.2f}."
            recommendation = compute_rich_analysis(df_copy, symbol, "VCS Setup", base_rec)

            return {
                "symbol": symbol.strip().upper(),
                "company_name": company_name,
                "cmp": buy_price,
                "day_change_pct": round(day_change_pct, 2),
                "vcs_score": round(today_score, 2),
                "volume": int(today['Volume']),
                "buy_price": buy_price,
                "exit_price": exit_price,
                "target_price": target_price,
                "confidence": confidence,
                "recommendation": recommendation
            }
            
        return None
        
    except Exception as e:
        print(f"VCS scan error for {symbol}: {e}")
        return None

def scan_monthly_early_stage2(symbol: str, df_monthly: pd.DataFrame, max_run_up_pct: float = 20.0, market_cap_cr: float = 0.0) -> dict | None:
    """
    Scans monthly OHLCV data for an Early Stage 2 Base Breakout.
    """
    if df_monthly is None or len(df_monthly) < 36: # Need 3 years to assess trend and base
        return None
        
    try:
        import pandas as pd
        df_copy = df_monthly.copy()
        df_copy['SMA7'] = df_copy['Close'].rolling(window=7).mean()
        
        if pd.isna(df_copy['SMA7'].iloc[-1]):
            return None
            
        today = df_copy.iloc[-1]
        prev = df_copy.iloc[-2]
        cmp = float(today['Close'].iloc[0] if isinstance(today['Close'], pd.Series) else today['Close'])
        open_today = float(today['Open'].iloc[0] if isinstance(today['Open'], pd.Series) else today['Open'])
        close_prev = float(prev['Close'].iloc[0] if isinstance(prev['Close'], pd.Series) else prev['Close'])
        open_prev = float(prev['Open'].iloc[0] if isinstance(prev['Open'], pd.Series) else prev['Open'])
        close_prev2 = float(df_copy.iloc[-3]['Close'].iloc[0] if isinstance(df_copy.iloc[-3]['Close'], pd.Series) else df_copy.iloc[-3]['Close'])

        if cmp < 10.0: # Ignore penny stocks
            return None
            
        # 1. Continuous Down Trend (Historical high vs Base low)
        historical_high = float(df_copy.iloc[-36:-12]['High'].max())
        
        # 2. Created a proper base (Last 12 months consolidation)
        base_period = df_copy.iloc[-12:]
        base_high = float(base_period['High'].max())
        base_bottom = float(base_period['Low'].min())
        
        # Must have fallen at least ~25% from the historical high to form the base
        if historical_high < base_bottom * 1.25: 
            return None
            
        # Base should be relatively tight, not extremely volatile (max 70% range)
        base_range_pct = ((base_high - base_bottom) / base_bottom) * 100.0
        if base_range_pct > 70.0: 
            return None
            
        # 3. Uptrend Initiation: CMP > 7-month SMA
        sma7 = float(today['SMA7'].iloc[0] if isinstance(today['SMA7'], pd.Series) else today['SMA7'])
        if cmp <= sma7:
            return None
            
        # 4. Recent Bullish Action: Current or prev month is a green candle
        curr_green = (cmp > open_today) or (cmp > close_prev)
        prev_green = (close_prev > open_prev) or (close_prev > close_prev2)
        if not (curr_green or prev_green):
            return None
            
        # 5. Run-up Limit: Not extended more than max_run_up_pct% above the 7-month SMA
        extension_from_sma = ((cmp - sma7) / sma7) * 100.0
        if extension_from_sma > max_run_up_pct or extension_from_sma < 0:
            return None
            
        # Ensure the SMA is not still a steep falling knife
        sma7_6m_ago = float(df_copy['SMA7'].iloc[-7])
        if sma7 < sma7_6m_ago * 0.85: 
            return None

        # Base recommendation
        day_change_pct = ((cmp - float(prev['Close'])) / float(prev['Close']) * 100) if len(df_copy) >= 2 else 0.0
        
        buy_price = round(cmp, 2)
        exit_price = round(base_bottom * 0.95, 2) # Stop below base bottom
        target_price = round(buy_price * 1.30, 2) # 30% target for Stage 2 breakout
        
        score = 100.0 - (extension_from_sma * 2.0) # Lower extension = higher score
        score = round(max(0, min(100.0, score)), 1)
        
        confidence = "High" if score >= 80 else "Medium-High"
        
        base_rec = (
            f"Early Stage 2 Base Breakout detected on Monthly timeframe! "
            f"Stock formed a base at ₹{base_bottom:.2f} and is up {extension_from_sma:.1f}% from the 7-month SMA. "
            f"It has now crossed the 7-Month SMA (₹{sma7:.2f}) with bullish monthly candles. "
            f"Buy around ₹{buy_price:.2f}, SL below base ₹{exit_price:.2f}, target ₹{target_price:.2f}."
        )
        
        # Calculate RSI and CCI for display
        close_series = df_copy['Close']
        delta = close_series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / (ema_loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = float(rsi_series.iloc[-1])
        
        tp = (df_copy['High'] + df_copy['Low'] + close_series) / 3
        sma_tp = tp.rolling(window=14).mean()
        mad_manual = tp.rolling(window=14).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        cci_series = (tp - sma_tp) / (0.015 * mad_manual + 1e-9)
        cci_val = float(cci_series.iloc[-1])
        
        recommendation = compute_rich_analysis(df_copy, symbol, "Stage 2 Breakout", base_rec)
        
        from config import get_company_name
        company_name = get_company_name(symbol)
        
        return {
            'symbol': symbol.strip().upper(),
            'company_name': company_name,
            'buy_price': buy_price,
            'exit_price': exit_price,
            'target_price': target_price,
            'confidence': confidence,
            'score': score,
            'recommendation': recommendation,
            'historical_high': round(historical_high, 2),
            'base_bottom': round(base_bottom, 2),
            'sma7': round(sma7, 2),
            'extension': round(extension_from_sma, 1),
            'rsi': round(rsi_val, 2),
            'cci': round(cci_val, 2)
        }

    except Exception as e:
        print(f"Stage 2 Early Breakout scan error for {symbol}: {e}")
        return None

def calc_atr(df: pd.DataFrame, length: int) -> pd.Series:
    """Calculate Average True Range (RMA method as in Pine Script atr)"""
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Pine Script atr() uses RMA (alpha = 1/length)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    return atr

def calc_vpa_trends(df: pd.DataFrame) -> dict:
    """
    Computes the VPA Major, Mid, and Minor trends based on RWI High/Low.
    Short term: 2, 8
    Long term: 10, 40
    """
    if df is None or len(df) < 45:
        return {"major": 0, "mid": 0, "minor": 0}
        
    def get_rsh(ps):
        low_shifted = df['Low'].shift(ps)
        atr_val = calc_atr(df, ps)
        return (df['High'] - low_shifted) / (atr_val * np.sqrt(ps))

    def get_rsl(ps):
        high_shifted = df['High'].shift(ps)
        atr_val = calc_atr(df, ps)
        return (high_shifted - df['Low']) / (atr_val * np.sqrt(ps))
        
    # Short term RWI High/Low
    rshmin = get_rsh(2)
    rshmax = get_rsh(8)
    rslmin = get_rsl(2)
    rslmax = get_rsl(8)
    
    RWIHi = np.maximum(rshmin, rshmax)
    RWILo = np.maximum(rslmin, rslmax)
    
    ground = RWIHi
    sky = RWILo
    
    # Long term RWI High/Low
    rlhmin = get_rsh(10)
    rlhmax = get_rsh(40)
    rllmin = get_rsl(10)
    rllmax = get_rsl(40)
    
    RWILHi = np.maximum(rlhmin, rlhmax)
    RWILLo = np.maximum(rllmin, rllmax)
    
    j = RWILHi - RWILLo
    j2 = RWILHi
    k2 = RWILLo
    
    latest_ground = ground.iloc[-1]
    latest_sky = sky.iloc[-1]
    latest_j = j.iloc[-1]
    latest_j2 = j2.iloc[-1]
    latest_k2 = k2.iloc[-1]
    
    # Evaluate Mid trend
    mid_trend = 1 if latest_ground > 1 else (-1 if latest_sky > 1 else 0)
    
    # Evaluate Major trend
    major_trend = 1 if latest_j > 1 else (-1 if latest_j < -1 else 0)
    
    # Evaluate Minor trend
    minor_trend = 1 if latest_j2 > 1 else (-1 if latest_k2 > 1 else 0)
    
    return {
        "major": major_trend,
        "mid": mid_trend,
        "minor": minor_trend
    }

def scan_vpa_trend(symbol: str, df: pd.DataFrame) -> dict | None:
    """
    Scans a stock for VPA Trend across Daily, Weekly, and Monthly timeframes.
    Returns a dictionary of results.
    """
    if df is None or len(df) < 50:
        return None
        
    try:
        # Calculate Daily
        daily_trends = calc_vpa_trends(df)
        
        # Resample to Weekly
        df_weekly = df.set_index('Date').resample('W-FRI').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        weekly_trends = calc_vpa_trends(df_weekly)
        
        # Resample to Monthly
        df_monthly = df.set_index('Date').resample('ME').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
        monthly_trends = calc_vpa_trends(df_monthly)
        
        # Determine overall score (simple heuristic: up trend +1, down trend -1)
        score = 0
        for t in [daily_trends, weekly_trends, monthly_trends]:
            score += t['major'] * 3
            score += t['mid'] * 2
            score += t['minor'] * 1
            
        cmp = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else cmp
        day_change_pct = ((cmp - prev_close) / prev_close) * 100
        
        from config import get_company_name
        company_name = get_company_name(symbol)
        
        return {
            "symbol": symbol.strip().upper(),
            "company_name": company_name,
            "cmp": round(cmp, 2),
            "day_change_pct": round(day_change_pct, 2),
            "volume": int(df['Volume'].iloc[-1]),
            "daily": daily_trends,
            "weekly": weekly_trends,
            "monthly": monthly_trends,
            "score": score
        }
    except Exception as e:
        print(f"Error in VPA scan for {symbol}: {e}")
        return None
