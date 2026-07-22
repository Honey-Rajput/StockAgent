# scanner.py
import pandas as pd
import numpy as np
from config import DRY_VOLUME_THRESHOLD, MIN_VOLUME_RATIO, MIN_PRICE_CHANGE, DRY_ZONE_MIN_DAYS, DRY_ZONE_MAX_DAYS

def calculate_trade_levels(df: pd.DataFrame, cmp: float, indicators: dict = None) -> tuple[float, float, float, float, float]:
    """
    Calculates technical trade levels based on Support and Resistance.
    Returns: (buy_price, exit_price, target_price, primary_support, primary_resistance)
    """
    try:
        # 1. Find Support Levels below CMP
        support_candidates = []
        
        # 20-day and 50-day SMA Support
        if indicators and 'sma20' in indicators and 'sma50' in indicators:
            sma20 = float(indicators['sma20']) if indicators['sma20'] is not None else float(df['Close'].rolling(20).mean().iloc[-1])
            sma50 = float(indicators['sma50']) if indicators['sma50'] is not None else float(df['Close'].rolling(50).mean().iloc[-1])
            if sma20 < cmp: support_candidates.append(sma20)
            if sma50 < cmp: support_candidates.append(sma50)
        else:
            sma20 = float(df['Close'].rolling(20).mean().iloc[-1])
            sma50 = float(df['Close'].rolling(50).mean().iloc[-1])
            if sma20 < cmp: support_candidates.append(sma20)
            if sma50 < cmp: support_candidates.append(sma50)
            
        # 20-day swing low
        swing_low_20d = float(df['Low'].iloc[-20:].min())
        if swing_low_20d < cmp: support_candidates.append(swing_low_20d)
        
        # Determine Primary Support
        valid_supports = [s for s in support_candidates if s < cmp * 0.99] # Must be at least 1% below CMP
        if valid_supports:
            primary_support = max(valid_supports)
        else:
            primary_support = cmp * 0.95 # Fallback to 5% standard stop
            
        # 2. Find Resistance Levels above CMP
        # 60-day swing high
        swing_high_60d = float(df['High'].iloc[-60:].max())
        
        if swing_high_60d > cmp * 1.05:
            primary_resistance = swing_high_60d
        else:
            primary_resistance = cmp * 1.20 # Fallback 20% target if at ATH
            
        # 3. Derive actionable trade levels
        # Buy price is exactly the support level (or up to 1% above it)
        buy_price = round(primary_support * 1.01, 2)
        
        # Exit price is safely below support
        exit_price = round(primary_support * 0.96, 2)
        
        # Target price aims at resistance
        target_price = round(primary_resistance * 0.98, 2) # Just below resistance
        
        return buy_price, exit_price, target_price, primary_support, primary_resistance
    except Exception:
        # Absolute fallback if DataFrame doesn't have enough history
        return round(cmp, 2), round(cmp * 0.95, 2), round(cmp * 1.15, 2), round(cmp * 0.95, 2), round(cmp * 1.15, 2)

# Pre-load scipy at module level to avoid repeated import overhead in scan_structural_vcp / volume profile
try:
    from scipy.signal import find_peaks as _scipy_find_peaks
    from scipy.signal import argrelextrema as _scipy_argrelextrema
except ImportError:
    _scipy_find_peaks = None
    _scipy_argrelextrema = None

# Import the pre-computation accelerator
from indicators import build_rich_analysis_from_indicators


# --- config knobs (optimized VDU logic) ---
BASELINE_EXCLUDE_BARS = 90
BASELINE_LOOKBACK_BARS = 150
RECENT_SEARCH_BARS = 10
RECENCY_WEIGHT = 0.01
MIN_RANGE_PCT = 0.07
MIN_AVG_DOLLAR_VOL = None
PRICE_CHANGE_MODE = "either"  # Compatible with older logic

def scan_stock(
    symbol: str, 
    df: pd.DataFrame, 
    min_dry_days: int = DRY_ZONE_MIN_DAYS, 
    max_dry_days: int = DRY_ZONE_MAX_DAYS,
    min_volume_ratio: float = MIN_VOLUME_RATIO,
    min_price_change: float = MIN_PRICE_CHANGE,
    min_dry_spikes: int = 2,
    indicators: dict = None
) -> dict | None:
    min_required_len = max(50, BASELINE_EXCLUDE_BARS + BASELINE_LOOKBACK_BARS // 2)
    if df is None or len(df) < min_required_len:
        return None

    df = df.reset_index(drop=True)
    n = len(df)
    min_dry_days = max(1, min_dry_days)

    # --- STEP 0: guards on today's row ---
    today = df.iloc[-1]
    yesterday = df.iloc[-2] if n >= 2 else today

    if pd.isna(today['Open']) or today['Open'] <= 0:
        return None
    if pd.isna(today['Close']) or pd.isna(today['Volume']):
        return None

    # --- STEP 1: Baseline volume — computed BEFORE the dry-zone search range,
    # so it can't be dragged down by the very window it's benchmarking. ---
    baseline_end = n - 1 - BASELINE_EXCLUDE_BARS
    baseline_start = max(0, baseline_end - BASELINE_LOOKBACK_BARS)
    if baseline_end <= baseline_start:
        # not enough history to build a clean, non-overlapping baseline
        baseline_subset = df.iloc[:-1].iloc[-90:]  # fallback to original behavior
    else:
        baseline_subset = df.iloc[baseline_start:baseline_end]

    baseline_avg_vol = baseline_subset['Volume'].mean()
    if not baseline_avg_vol or baseline_avg_vol <= 0 or pd.isna(baseline_avg_vol):
        return None

    if MIN_AVG_DOLLAR_VOL is not None:
        avg_dollar_vol = (baseline_subset['Volume'] * baseline_subset['Close']).mean()
        if pd.isna(avg_dollar_vol) or avg_dollar_vol < MIN_AVG_DOLLAR_VOL:
            return None

    # --- STEP 2: Vectorized dry-zone search ---
    history_df = df.iloc[:-1].reset_index(drop=True)
    h_vol = history_df['Volume']
    h_high = history_df['High']
    h_low = history_df['Low']

    is_not_dry_mask = h_vol > (0.40 * baseline_avg_vol)

    search_start_idx = len(history_df) - 1
    search_end_idx = max(0, len(history_df) - RECENT_SEARCH_BARS)

    best = None  # (start_idx, end_idx, L, spikes, dry_avg_vol, score)

    for L in range(min_dry_days, max_dry_days + 1):
        roll_vol_mean = h_vol.rolling(L).mean()
        roll_high = h_high.rolling(L).max()
        roll_low = h_low.rolling(L).min()
        roll_spikes = is_not_dry_mask.rolling(L).sum()

        # only consider window END indices within the recent search range,
        # and only where the window fully fits (idx - L + 1 >= 0)
        candidate_idx = range(search_start_idx, search_end_idx - 1, -1)

        for idx in candidate_idx:
            if idx - L + 1 < 0:
                continue

            dry_avg_vol = roll_vol_mean.iloc[idx]
            if pd.isna(dry_avg_vol) or dry_avg_vol <= 0:
                continue
            if dry_avg_vol > (0.90 * baseline_avg_vol):
                continue

            zone_low = roll_low.iloc[idx]
            zone_high = roll_high.iloc[idx]
            if pd.isna(zone_low) or zone_low <= 0:
                continue
            if (zone_high - zone_low) / zone_low < MIN_RANGE_PCT:
                continue

            spikes = roll_spikes.iloc[idx]
            if pd.isna(spikes) or spikes < min_dry_spikes:
                continue

            # recency-weighted score: lower is better.
            bars_back = search_start_idx - idx
            score = dry_avg_vol * (1 + RECENCY_WEIGHT * bars_back)

            if best is None or score < best[5]:
                start_idx = idx - L + 1
                best = (start_idx, idx, L, int(spikes), dry_avg_vol, score)

    if best is None:
        return None

    start_idx, end_idx, dry_days_count, dry_spikes, dry_avg_vol, _ = best

    # --- STEP 3: Today's breakout check ---
    volume_ratio = today['Volume'] / dry_avg_vol
    volume_surge_ok = volume_ratio >= min_volume_ratio

    has_green_body = today['Close'] > today['Open']
    closed_above_prior = today['Close'] > yesterday['Close']
    bullish_candle_ok = has_green_body and closed_above_prior

    pct_change_intraday = (today['Close'] - today['Open']) / today['Open'] * 100
    if n >= 2 and yesterday['Close'] > 0:
        pct_change_close = (today['Close'] - yesterday['Close']) / yesterday['Close'] * 100
    else:
        pct_change_close = pct_change_intraday

    if PRICE_CHANGE_MODE == "close_to_close":
        price_change_ok = pct_change_close >= min_price_change
    elif PRICE_CHANGE_MODE == "intraday":
        price_change_ok = pct_change_intraday >= min_price_change
    else:
        price_change_ok = (pct_change_intraday >= min_price_change) or (pct_change_close >= min_price_change)

    is_breakout = volume_surge_ok and bullish_candle_ok and price_change_ok
    is_pre_breakout = False

    if not is_breakout:
        # Pre-breakout condition: It's NOT a breakout today, BUT the current day is STILL a dry day.
        # Specifically, if today's volume is also very dry and the price is consolidating tightly.
        today_vol_dry = today['Volume'] <= (0.75 * baseline_avg_vol)
        is_tight = abs(pct_change_close) < 2.0
        # Check if the historical dry zone we found extends up to yesterday
        is_current_dry_zone = (len(history_df) - 1 - end_idx) <= 2
        
        if today_vol_dry and is_tight and is_current_dry_zone:
            is_pre_breakout = True
        else:
            return None
            
    setup_type = "VDU Breakout" if is_breakout else "VDU Pre-Breakout"
        
    # --- STEP 4: Signal Strength Score (0 to 100) & Indicators ---
    # Use pre-computed indicators DataFrame if available, otherwise compute
    if indicators is not None and 'df' in indicators:
        df_indicators = indicators['df']
    else:
        df_indicators = df.copy()
        df_indicators['MA50'] = df_indicators['Close'].rolling(window=50).mean()
        df_indicators['MA200'] = df_indicators['Close'].rolling(window=200).mean()
    
    today_ma = df_indicators['MA50'].iloc[-1] if 'MA50' in df_indicators.columns else np.nan
    above_50dma = False
    if not pd.isna(today_ma):
        above_50dma = today['Close'] > today_ma
        
    today_ma200 = df_indicators['MA200'].iloc[-1] if 'MA200' in df_indicators.columns else np.nan
    above_200dma = False
    if not pd.isna(today_ma200):
        above_200dma = today['Close'] > today_ma200
        
    # --- STEP 4b: Relative Volume (RVOL) vs 50-day avg ---
    # RVOL > 2.0 = institutional-grade breakout conviction
    vol_50d_baseline = df['Volume'].iloc[-50:].mean() if len(df) >= 50 else baseline_avg_vol
    rvol = round(today['Volume'] / vol_50d_baseline, 2) if vol_50d_baseline > 0 else 1.0

    # --- STEP 4c: ATR-based Stop Loss (1.5x ATR) ---
    # More adaptive than fixed % stops — matches each stock's actual volatility
    atr_14 = None
    try:
        if indicators is not None and 'df' in indicators and 'TR' in indicators['df'].columns:
            atr_14 = float(indicators['df']['TR'].rolling(14).mean().iloc[-1])
        else:
            tr = pd.concat([
                df['High'] - df['Low'],
                (df['High'] - df['Close'].shift(1)).abs(),
                (df['Low']  - df['Close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_14 = float(tr.rolling(14).mean().iloc[-1])
    except Exception:
        atr_14 = None

    # --- STEP 4d: Signal Strength Score (0-100) — balanced volume + price ---
    # Expert fix: volume and price must BOTH be strong for high score
    score = 0.0
    dryness_ratio = dry_avg_vol / baseline_avg_vol

    if setup_type == "VDU Pre-Breakout":
        # For Pre-Breakouts, reward extreme tightness and dryness instead of volume surge
        # 1. Tightness (Max 35 points) - closer to 0% is better
        tightness = abs(pct_change_close)
        score += max(0.0, (2.0 - tightness) / 2.0 * 35.0)
        
        # 2. Dry Zone Quality (Max 30 points) - how dry the consolidation was
        score += max(0.0, (0.90 - dryness_ratio) / 0.90 * 30.0)
        
        # 3. Today's Dryness (Max 20 points) - ensure today is also extremely low volume
        today_ratio = today['Volume'] / baseline_avg_vol
        if today_ratio <= 0.75:
            score += max(0.0, (0.75 - today_ratio) / 0.75 * 20.0)
            
        # 4. Moving Averages Context
        if above_50dma:
            score += 10.0
        if above_200dma:
            score += 5.0
    else:
        # Breakout Setup Scoring
        # 1. Volume ratio: Up to 25 points (was 40 — reduced to avoid volume-only high scores)
        score += min(volume_ratio / 5.0 * 25.0, 25.0)
        # 2. Price move: Up to 25 points (raised from 30 to match volume weight)
        best_pct = max(pct_change_intraday, pct_change_close)
        score += min(best_pct / 3.0 * 25.0, 25.0)
        # 3. COMBO BONUS: Both volume >= 3x AND price move >= 2% → institutional quality
        if volume_ratio >= 3.0 and best_pct >= 2.0:
            score += 10.0
        # 4. Dry zone TIGHTNESS quality score:
        #    Reward how "dry" the consolidation was relative to baseline.
        dryness_score = max(0.0, (0.90 - dryness_ratio) / 0.90 * 20.0)  # up to 20 pts
        score += dryness_score
        # 5. Moving Average filter: above 50 SMA = uptrend confirmed
        if above_50dma:
            score += 8.0
        # 5b. Above 200 SMA bonus (Stage-2 context)
        if above_200dma:
            score += 5.0
        # 6. MACD cross-up bonus: momentum shift confirmation
        if indicators and indicators.get('macd_cross_up', False):
            score += 5.0
        # 7. RVOL bonus: institutional participation vs 50-day baseline
        if rvol >= 3.0:
            score += 7.0
        elif rvol >= 2.0:
            score += 4.0

    score = round(min(score, 100.0), 1)

    # Calculate day-over-day price change (standard Close-to-Close change)
    day_change_pct = pct_change_close

    # Correct 52-week high/low using last 252 trading bars (not full history)
    history_252 = df.iloc[-252:] if len(df) >= 252 else df
    high_52w = float(history_252['High'].max())
    low_52w  = float(history_252['Low'].min())

    # 8. 52-week high proximity bonus: stocks approaching 52W highs attract institutions
    cmp = float(today['Close'])
    dist_from_52w_high_pct = (high_52w - cmp) / high_52w * 100 if high_52w > 0 else 100.0
    if dist_from_52w_high_pct <= 5.0:
        score += 10.0   # Very near 52W high — near breakout zone
    elif dist_from_52w_high_pct <= 10.0:
        score += 5.0    # Approaching breakout zone

    score = round(min(score, 100.0), 1)

    # Advanced Trading Setup Calculations based on Support & Resistance:
    buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(history_df, cmp, indicators)

    # ATR-based alternative stop (tighter, more professional)
    if atr_14 and atr_14 > 0:
        atr_stop = round(cmp - 1.5 * atr_14, 2)
        # Use ATR stop if it's tighter (less risk) than support-based stop
        if atr_stop > exit_price:
            exit_price = atr_stop

    # Risk:Reward ratio
    risk   = buy_price - exit_price
    reward = target_price - buy_price
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

    # Expert filter: require minimum 1.5:1 R:R for a tradeable setup
    # (below 1.5 the math doesn't work in your favour over a series of trades)
    if rr_ratio < 1.5 and is_breakout:
        # Widen target to nearest resistance if R:R is too low
        history_60 = df.iloc[-60:] if len(df) >= 60 else df
        alt_target = round(float(history_60['High'].max()) * 0.98, 2)
        alt_reward = alt_target - buy_price
        alt_rr = round(alt_reward / risk, 2) if risk > 0 else 0.0
        if alt_rr >= 1.5:
            target_price = alt_target
            rr_ratio = alt_rr

    # Confidence text based on algorithmic signal strength
    if score >= 75:
        confidence = f"High Conviction ({score}%)"
    elif score >= 50:
        confidence = f"Medium-High ({score}%)"
    else:
        confidence = f"Medium ({score}%)"

    if setup_type == "VDU Breakout":
        base_rec = (
            f"Strong institutional VDU breakout! Volume ratio is {volume_ratio:.1f}x with signal score {score}%. "
            f"R:R Ratio: {rr_ratio:.1f}:1. "
            f"Buy Range: [₹{buy_price:.2f} to ₹{cmp:.2f}] (Nearest Support is ₹{support:.2f}). "
            f"Set stop loss securely below support at ₹{exit_price:.2f} "
            f"with a target near overhead resistance at ₹{target_price:.2f}."
        )
    else:
        base_rec = (
            f"Tight VDU Pre-Breakout Consolidation! Stock is trading with extremely dry volume ({dryness_ratio:.1%} of avg). "
            f"R:R Ratio: {rr_ratio:.1f}:1. "
            f"Buy near support at ₹{support:.2f} or wait for a volume surge above ₹{cmp * 1.02:.2f}. "
            f"Set tight stop loss at ₹{exit_price:.2f} "
            f"with a breakout target at ₹{target_price:.2f}."
        )
        
    recommendation = compute_rich_analysis(df_indicators, symbol, setup_type, base_rec, indicators=indicators)

    from config import get_company_name
    company_name = get_company_name(symbol)

    return {
        "symbol":           symbol.strip().upper(),
        "company_name":     company_name,
        "cmp":              cmp,
        "setup_type":       setup_type,
        "day_change_pct":   round(day_change_pct, 2),
        "today_volume":     int(today['Volume']),
        "dry_avg_vol":      round(dry_avg_vol, 1),
        "volume_ratio":     round(volume_ratio, 2),
        "rvol":             rvol,
        "dry_days_count":   int(dry_days_count),
        "dry_spikes":       dry_spikes,
        "dry_start_date":   pd.to_datetime(history_df['Date'].iloc[start_idx]),
        "dry_end_date":     pd.to_datetime(history_df['Date'].iloc[end_idx]),
        "signal_strength":  score,
        "pct_change_today": round(day_change_pct, 2),
        "above_50dma":      above_50dma,
        "above_200dma":     above_200dma,
        "df":               df_indicators,
        "high_52w":         high_52w,
        "low_52w":          low_52w,
        "dist_52w_high_pct": round(dist_from_52w_high_pct, 1),
        "buy_price":        buy_price,
        "exit_price":       exit_price,
        "target_price":     target_price,
        "atr_14":           round(atr_14, 2) if atr_14 else None,
        "rr_ratio":         rr_ratio,
        "confidence":       confidence,
        "recommendation":   recommendation,
    }

def scan_wt_cross(symbol: str, df: pd.DataFrame, wt_oversold_threshold: float = -40.0, indicators: dict = None) -> dict | None:
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
        
    # Use pre-computed WaveTrend from indicators if available
    if indicators is not None and 'df' in indicators and 'WT1' in indicators['df'].columns:
        df_copy = indicators['df']
        wt1 = df_copy['WT1']
        wt2 = df_copy['WT2']
    else:
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

    # Price filter: exclude stocks below ₹100
    last_close = float(df.iloc[-1]['Close'])
    if last_close < 100.0:
        return None
        
    # Check if wt1 is in oversold zone (below threshold)
    if today_wt1 > wt_oversold_threshold:
        return None
    
    # EXPERT FIX: WaveTrend oversold bounce only valid in UPTRENDS (above 200 SMA).
    # An oversold reading in a downtrend = dead cat bounce = losing trade.
    # Minervini, O'Neil, and Zanger all filter oversold signals by trend first.
    if len(df) >= 200:
        sma200_val = float(df['Close'].rolling(200).mean().iloc[-1])
        if not pd.isna(sma200_val) and last_close < sma200_val:
            return None  # Must be ABOVE 200 SMA — no bounces in downtrends
    
    today = df_copy.iloc[-1]
    yesterday = df_copy.iloc[-2] if len(df_copy) >= 2 else today
    day_change_pct = ((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if len(df_copy) >= 2 else 0.0
    
    # Issue #6: Detect buy signal (green dot) — reduced to 2-bar lookback for fresher signals
    buy_signal = False
    buy_signal_age = None
    for offset in range(1, 3):
        if len(wt1) > offset and len(wt2) > offset:
            t_wt1 = wt1.iloc[-offset]
            t_wt2 = wt2.iloc[-offset]
            p_wt1 = wt1.iloc[-offset - 1]
            p_wt2 = wt2.iloc[-offset - 1]
            if not pd.isna(t_wt1) and not pd.isna(t_wt2) and not pd.isna(p_wt1) and not pd.isna(p_wt2):
                if (p_wt1 <= p_wt2) and (t_wt1 > t_wt2):
                    buy_signal = True
                    buy_signal_age = offset - 1  # 0 = today, 1 = yesterday
                    break
    
    # Issue #7: Bullish divergence detection
    bullish_divergence = False
    if len(df) >= 20 and len(wt1) >= 20:
        price_low_recent = df['Low'].iloc[-5:].min()
        price_low_prev = df['Low'].iloc[-15:-5].min()
        wt_low_recent = wt1.iloc[-5:].min()
        wt_low_prev = wt1.iloc[-15:-5].min()
        if not pd.isna(wt_low_recent) and not pd.isna(wt_low_prev):
            bullish_divergence = (price_low_recent < price_low_prev) and (wt_low_recent > wt_low_prev)
    
    # Advanced Trading Setup Calculations for WaveTrend
    cmp = float(today['Close'])
    buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(df_copy, cmp, indicators)
    
    # Use pre-computed SMAs if available, otherwise compute
    if 'SMA20' not in df_copy.columns:
        df_copy['SMA20'] = df_copy['Close'].rolling(window=20).mean()
    if 'SMA50' not in df_copy.columns:
        df_copy['SMA50'] = df_copy['Close'].rolling(window=50).mean()
    today_sma20 = df_copy['SMA20'].iloc[-1]
    today_sma50 = df_copy['SMA50'].iloc[-1]
    above_20sma = bool(cmp > today_sma20) if not pd.isna(today_sma20) else False
    above_50sma = bool(cmp > today_sma50) if not pd.isna(today_sma50) else False

    # Compute RSI (14) for recommendation
    close_series = df_copy['Close']
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / (ema_loss + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_val = round(float(rsi_series.iloc[-1]), 1)

    # Compute CCI (14) for recommendation
    tp = (df_copy['High'] + df_copy['Low'] + close_series) / 3
    sma_tp = tp.rolling(window=14).mean()
    mad_val = tp.rolling(window=14).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    cci_series = (tp - sma_tp) / (0.015 * mad_val + 1e-9)
    cci_val = round(float(cci_series.iloc[-1]), 1) if not pd.isna(cci_series.iloc[-1]) else 0.0

    if buy_signal and bullish_divergence:
        confidence = "🔥 High (Buy Signal + Divergence)"
    elif buy_signal:
        confidence = "High (WT Buy Signal)"
    elif bullish_divergence:
        confidence = "High (Bullish Divergence)"
    else:
        confidence = "Medium (WT Oversold)"

    # Issue #8: Use rich analysis for recommendation instead of basic string
    signal_parts = []
    signal_parts.append(f"WaveTrend oversold at {today_wt1:.1f} (WT2: {today_wt2:.1f}).")
    if buy_signal:
        freshness = "today" if buy_signal_age == 0 else "yesterday"
        signal_parts.append(f"🟢 Buy signal detected {freshness} (WT1 crossed above WT2).")
    if bullish_divergence:
        signal_parts.append("📈 Bullish divergence: price made lower low but WT made higher low.")
    signal_parts.append(f"RSI: {rsi_val} | CCI: {cci_val}")
    base_rec = " ".join(signal_parts)
    recommendation = compute_rich_analysis(df, symbol, "WaveTrend Oversold", base_rec, indicators=indicators)

    from config import get_company_name
    company_name = get_company_name(symbol)
    
    return {
        "symbol": symbol.strip().upper(),
        "company_name": company_name,
        "cmp": cmp,
        "day_change_pct": round(day_change_pct, 2),
        "wt_value": round(today_wt1, 2),
        "wt2_value": round(today_wt2, 2),
        "buy_signal": buy_signal,
        "buy_signal_age": buy_signal_age,
        "bullish_divergence": bullish_divergence,
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

def compute_rich_analysis(df, symbol, strategy_name, base_rec_text, indicators=None):
    """
    Computes CCI, RSI, EMA, SMA and checklist triggers, 
    and returns a structured JSON recommendation string.
    When indicators dict is provided, delegates to the fast pre-computed version.
    """
    # FAST PATH: use pre-computed indicators if available
    if indicators is not None:
        return build_rich_analysis_from_indicators(indicators, symbol, strategy_name, base_rec_text)

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
        rsi_sma = rsi_series.rolling(window=14).mean()
        rsi_val = float(rsi_series.iloc[-1])
        rsi_sma_val = float(rsi_sma.iloc[-1]) if len(rsi_sma) > 0 else 0.0
        
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

        # OI interpretation (when futures OI column is present)
        oi_status = None
        oi_interp = None
        oi_reasoning = ""
        if 'Open Interest' in df.columns and len(df) >= 2:
            try:
                prev_close = close_series.iloc[-2]
                curr_close = close_series.iloc[-1]
                prev_oi = df['Open Interest'].iloc[-2]
                curr_oi = df['Open Interest'].iloc[-1]
                if pd.notna(prev_oi) and pd.notna(curr_oi):
                    price_up = curr_close > prev_close
                    oi_up = curr_oi > prev_oi
                    if price_up and oi_up:
                        oi_status = "Long Build-up"
                        oi_interp = "Bullish – New long positions are being created. Strong buying interest."
                    elif not price_up and oi_up:
                        oi_status = "Short Build-up"
                        oi_interp = "Bearish – New short positions are being created. Strong selling interest."
                    elif price_up and not oi_up:
                        oi_status = "Short Covering"
                        oi_interp = "Bullish – Existing short sellers are closing positions, causing price to rise."
                    else:
                        oi_status = "Long Unwinding"
                        oi_interp = "Bearish – Existing long holders are exiting positions."
                    oi_reasoning = f"OI Analysis: {oi_status} ({oi_interp})."
            except Exception:
                pass

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

        final_text = f"{rsi_reasoning} {cci_reasoning} {ma_reasoning} {oi_reasoning} {sl_target_part}"

        analysis_payload = {
            "is_rich": True,
            "text": final_text,
            "rsi": round(rsi_val, 1),
            "rsi_status": rsi_status,
            "rsi_interp": rsi_interp,
            "cci": round(cci_val, 1),
            "cci_status": cci_status,
            "cci_interp": cci_interp,
            "oi_status": oi_status,
            "oi_interp": oi_interp,
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
    6. RSI(14) > SMA(RSI, 14)
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

        # Condition 4: ROC >= 10.0% (Matched UI 10-80%)
        if not (roc6 >= 10.0):
            return None
        # Condition 5: ROC <= 80.0%
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
        rsi_sma = rsi_series.rolling(window=14).mean()
        rsi_val = float(rsi_series.iloc[-1])
        rsi_sma_val = float(rsi_sma.iloc[-1]) if len(rsi_sma) > 0 else 0.0

        # Condition 6: RSI > 14 SMA RSI
        if not (rsi_val > rsi_sma_val):
            return None
        # Condition 7: RSI 55 to 85 (Matched UI)
        if not (rsi_val >= 55.0 and rsi_val <= 85.0):
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
        buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(df_monthly, cmp)

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
            f"Buy Range: [₹{buy_price:.2f} to ₹{cmp:.2f}] (Nearest Support is ₹{support:.2f}). "
            f"Set stop loss securely below support at ₹{exit_price:.2f} "
            f"with a target near overhead resistance at ₹{target_price:.2f}."
        )
        recommendation = compute_rich_analysis(df_monthly, symbol, "Monthly EMA Momentum", base_rec)

        from config import get_company_name
        company_name = get_company_name(symbol)

        return {
            "symbol": symbol.strip().upper(),
            "company_name": company_name,
            "cmp": cmp,
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
    7. Weekly RSI(14) > SMA(RSI, 14)
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
        # Condition 7: RSI(14) > SMA(RSI, 14)
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs       = avg_gain / (avg_loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        rsi_sma = rsi_series.rolling(window=14).mean()
        rsi_val = float(rsi_series.iloc[-1])
        rsi_sma_val = float(rsi_sma.iloc[-1]) if len(rsi_sma) > 0 else 0.0
        
        if not (rsi_val > rsi_sma_val):
            return None

        # Condition 5: CCI(20) > 90
        tp       = (high + low + close) / 3.0
        tp_sma20 = tp.rolling(window=20).mean()
        mad20    = tp.rolling(window=20).apply(lambda x: (abs(x - x.mean())).mean(), raw=True)
        cci_val  = float(((tp - tp_sma20) / (0.015 * (mad20 + 1e-9))).iloc[-1])
        if not (cci_val > 90.0):
            return None

        # ---- All conditions passed — trade setup ----
        buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(df_weekly, cmp)

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
            f"Buy Range: [₹{buy_price:.2f} to ₹{cmp:.2f}] (Nearest Support is ₹{support:.2f}). "
            f"Set stop loss securely below support at ₹{exit_price:.2f} "
            f"with a target near overhead resistance at ₹{target_price:.2f}."
        )
        recommendation = compute_rich_analysis(df_weekly, symbol, "Weekly Momentum Breakout", base_rec)

        from config import get_company_name
        company_name = get_company_name(symbol)

        return {
            "symbol": symbol.strip().upper(),
            "company_name": company_name,
            "cmp": cmp,
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
    
    # 2. Batch download last 2 days of quotes to get current CMP and true previous close
    cmp_map = {}
    prev_close_map = {}
    try:
        quotes_df = yf.download(tickers=tickers, period="2d", progress=False, threads=False)
        if not quotes_df.empty:
            if isinstance(quotes_df.columns, pd.MultiIndex):
                # multi-ticker: use Close.iloc[-1] as CMP and Close.iloc[-2] as previous day close
                for tk in tickers:
                    sym_clean = tk.replace(".NS", "").upper()
                    try:
                        close_series = quotes_df['Close'][tk] if 'Close' in quotes_df else None
                        if close_series is not None and not close_series.dropna().empty:
                            cs = close_series.dropna()
                            cmp_map[sym_clean] = float(cs.iloc[-1])
                            # Use actual previous day close (not open) for correct daily change
                            if len(cs) >= 2:
                                prev_close_map[sym_clean] = float(cs.iloc[-2])
                    except Exception:
                        pass
            else:
                # single ticker
                sym_clean = symbols[0]
                cs = quotes_df['Close'].dropna()
                cmp_map[sym_clean] = float(cs.iloc[-1])
                if len(cs) >= 2:
                    prev_close_map[sym_clean] = float(cs.iloc[-2])
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

def scan_vcs(symbol: str, df: pd.DataFrame, lenShort=13, lenLong=63, lenVol=50, sensitivity=2.0, max_score=10.0, indicators: dict = None) -> dict | None:
    """
    Scans a stock for Volatility Contraction (VCS) based on ATR, StdDev, and Volume contraction.
    """
    if df is None or len(df) < max(lenLong, lenVol) + 2:
        return None

    try:
        # Use pre-computed indicators DataFrame if available (all VCS columns are pre-computed)
        if indicators is not None and 'df' in indicators:
            df_copy = indicators['df'].copy()
            # The ratioATR, ratioStd, volRatio_vcs columns are already computed
            if 'ratioATR' in df_copy.columns and 'ratioStd' in df_copy.columns:
                # Rename volRatio_vcs back to volRatio for VCS scoring
                if 'volRatio_vcs' in df_copy.columns:
                    df_copy['volRatio'] = df_copy['volRatio_vcs']
            else:
                # Fallback: compute VCS ratios (shouldn't happen but safety)
                indicators = None

        if indicators is None:
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
        is_weekly = False
        try:
            if isinstance(df_copy.index, pd.DatetimeIndex) and len(df_copy) > 1:
                avg_days = (df_copy.index[-1] - df_copy.index[0]).days / len(df_copy)
                if avg_days > 4:
                    is_weekly = True
            elif 'Date' in df_copy.columns and pd.api.types.is_datetime64_any_dtype(df_copy['Date']) and len(df_copy) > 1:
                avg_days = (df_copy['Date'].iloc[-1] - df_copy['Date'].iloc[0]).days / len(df_copy)
                if avg_days > 4:
                    is_weekly = True
        except Exception:
            pass

        sma_fast_len = 10 if is_weekly else 50
        sma_slow_len = 40 if is_weekly else 200
        vol_min = 250000 if is_weekly else 50000

        df_copy["sma50"] = df_copy["Close"].rolling(sma_fast_len).mean()
        df_copy["sma200"] = df_copy["Close"].rolling(sma_slow_len).mean()

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
        if pd.isna(today_sma200) or today_close < today_sma50 or today_close < today_sma200 or today_sma50 < today_sma200 or today_vol_avg < vol_min:
            return None

        if today_score <= max_score:
            today = df_copy.iloc[-1]
            yesterday = df_copy.iloc[-2] if len(df_copy) >= 2 else today
            day_change_pct = ((today['Close'] - yesterday['Close']) / yesterday['Close'] * 100) if len(df_copy) >= 2 else 0.0
            
            cmp = float(today['Close'])
            buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(df_copy, cmp, indicators)
            
            from config import get_company_name
            company_name = get_company_name(symbol)
            
            confidence = "High" if today_score < 5 else "Medium"
            
            base_rec = (
                f"VCS final score is {today_score:.2f} (below {max_score}), indicating strict setup conditions met. "
                f"Buy Range: [₹{buy_price:.2f} to ₹{cmp:.2f}] (Nearest Support is ₹{support:.2f}). "
                f"Set stop loss securely below support at ₹{exit_price:.2f} "
                f"with a target near overhead resistance at ₹{target_price:.2f}."
            )
            recommendation = compute_rich_analysis(df_copy, symbol, "VCS Setup", base_rec, indicators=indicators)

            return {
                "symbol": symbol.strip().upper(),
                "company_name": company_name,
                "cmp": cmp,
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
        
        support = round(sma7, 2)
        buy_price = round(support * 1.01, 2)
        exit_price = round(support * 0.95, 2)
        target_price = round(cmp * 1.25, 2)
        
        score = 100.0 - (extension_from_sma * 2.0) # Lower extension = higher score
        score = round(max(0, min(100.0, score)), 1)
        
        confidence = "High" if score >= 80 else "Medium-High"
        
        base_rec = (
            f"Early Stage 2 Base Breakout! "
            f"Stock formed a base at ₹{base_bottom:.2f} and is up {extension_from_sma:.1f}% from the 7-month SMA. "
            f"It has now crossed the 7-Month SMA (₹{sma7:.2f}) with bullish monthly candles. "
            f"Buy Range: [₹{buy_price:.2f} to ₹{cmp:.2f}] (Nearest Support is ₹{support:.2f}). "
            f"Set stop loss securely below support at ₹{exit_price:.2f} "
            f"with a target near overhead resistance at ₹{target_price:.2f}."
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
        rsi_sma = rsi_series.rolling(window=14).mean()
        rsi_val = float(rsi_series.iloc[-1])
        rsi_sma_val = float(rsi_sma.iloc[-1]) if len(rsi_sma) > 0 else 0.0
        
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
            'cmp': round(cmp, 2),
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

def calc_atr_from_tr(tr: pd.Series, length: int) -> pd.Series:
    """Calculate Average True Range using RMA method exactly as in Pine Script"""
    if len(tr) < length:
        return tr.ewm(alpha=1/length, adjust=False).mean()
        
    sma = tr.rolling(window=length).mean()
    rma_values = np.zeros(len(tr))
    
    tr_values = tr.values
    sma_values = sma.values
    alpha = 1.0 / length
    
    first_valid_idx = length - 1
    rma_values[:first_valid_idx] = np.nan
    if first_valid_idx < len(tr):
        rma_values[first_valid_idx] = sma_values[first_valid_idx]
        
    for i in range(first_valid_idx + 1, len(tr)):
        rma_values[i] = alpha * tr_values[i] + (1 - alpha) * rma_values[i-1]
        
    return pd.Series(rma_values, index=tr.index)

def calc_vpa_trends(df: pd.DataFrame) -> dict:
    """
    Computes the VPA Major, Mid, and Minor trends based on RWI High/Low.
    Short term: 2, 8
    Long term: 10, 40
    """
    if df is None or len(df) < 3:
        return {"major": 0, "mid": 0, "minor": 0, "rsi": 0.0, "cci": 0.0}
        
    # Pre-compute True Range ONCE instead of 72 times
    close_prev = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - close_prev).abs()
    tr3 = (df['Low'] - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
    def get_rsh(ps):
        low_shifted = df['Low'].shift(ps)
        atr_val = calc_atr_from_tr(tr, ps)
        return (df['High'] - low_shifted) / (atr_val * np.sqrt(ps))

    def get_rsl(ps):
        high_shifted = df['High'].shift(ps)
        atr_val = calc_atr_from_tr(tr, ps)
        return (high_shifted - df['Low']) / (atr_val * np.sqrt(ps))
        
    def get_max_rsh(start_ps, end_ps):
        if start_ps >= len(df):
            return pd.Series(0, index=df.index)
        actual_end_ps = min(end_ps, len(df) - 1)
        if actual_end_ps < start_ps:
            return pd.Series(0, index=df.index)
            
        res = get_rsh(start_ps)
        for ps in range(start_ps + 1, actual_end_ps + 1):
            res = np.maximum(res, get_rsh(ps))
        return res

    def get_max_rsl(start_ps, end_ps):
        if start_ps >= len(df):
            return pd.Series(0, index=df.index)
        actual_end_ps = min(end_ps, len(df) - 1)
        if actual_end_ps < start_ps:
            return pd.Series(0, index=df.index)
            
        res = get_rsl(start_ps)
        for ps in range(start_ps + 1, actual_end_ps + 1):
            res = np.maximum(res, get_rsl(ps))
        return res
        
    # Short term RWI High/Low
    ground = get_max_rsh(2, 8)
    sky = get_max_rsl(2, 8)
    
    # Long term RWI High/Low
    RWILHi = get_max_rsh(10, 40)
    RWILLo = get_max_rsl(10, 40)
    
    j = RWILHi - RWILLo
    j2 = RWILHi
    k2 = RWILLo
    
    latest_ground = ground.iloc[-1]
    latest_sky = sky.iloc[-1]
    latest_j = j.iloc[-1]
    latest_j2 = j2.iloc[-1]
    latest_k2 = k2.iloc[-1]
    
    # Evaluate Minor trend (Short-term RWI: 2 to 8)
    minor_trend = 1 if latest_ground > 1 else (-1 if latest_sky > 1 else 0)
    
    # Evaluate Major trend (Difference of Long-term RWI)
    major_trend = 1 if latest_j > 1 else (-1 if latest_j < -1 else 0)

    # Evaluate Mid trend (Long-term RWI: 10 to 40)
    mid_trend = 1 if latest_j2 > 1 else (-1 if latest_k2 > 1 else 0)
    
    # Calculate RSI (14)
    close_series = df['Close']
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(com=13, adjust=False).mean()
    ema_loss = loss.ewm(com=13, adjust=False).mean()
    rs = ema_gain / (ema_loss + 1e-9)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_val = float(rsi_series.iloc[-1]) if len(rsi_series) > 0 else 0.0

    # Calculate CCI (14)
    tp = (df['High'] + df['Low'] + close_series) / 3
    sma_tp = tp.rolling(window=14).mean()
    mad_manual = tp.rolling(window=14).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    cci_series = (tp - sma_tp) / (0.015 * mad_manual + 1e-9)
    cci_val = float(cci_series.iloc[-1]) if len(cci_series) > 0 and not pd.isna(cci_series.iloc[-1]) else 0.0
    
    return {
        "major": major_trend,
        "mid": mid_trend,
        "minor": minor_trend,
        "major_val": round(float(latest_j), 2) if not pd.isna(latest_j) else 0.0,
        "mid_val": round(float(latest_ground), 2) if not pd.isna(latest_ground) else 0.0,
        "minor_val": round(float(latest_j2), 2) if not pd.isna(latest_j2) else 0.0,
        "rsi": round(rsi_val, 2),
        "cci": round(cci_val, 2)
    }

def scan_vpa_trend(symbol: str, df: pd.DataFrame, indicators: dict = None) -> dict | None:
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
        
        # Determine overall score — weighted by trend STRENGTH (RWI magnitude), not just direction
        raw_score = 0
        max_strength = 0
        for t in [daily_trends, weekly_trends, monthly_trends]:
            major_strength = min(abs(t.get('major_val', 0)), 3.0)
            mid_strength = min(abs(t.get('mid_val', 0)), 3.0)
            minor_strength = min(abs(t.get('minor_val', 0)), 3.0)
            raw_score += t['major'] * 3 * max(major_strength, 1.0)
            raw_score += t['mid'] * 2 * max(mid_strength, 1.0)
            raw_score += t['minor'] * 1 * max(minor_strength, 1.0)
            max_strength = max(max_strength, major_strength, mid_strength, minor_strength)
            
        # Normalize score to a 0-100 scale (max possible ~54 with all 3.0 strengths)
        score = round(max(0, min(100, (raw_score + 54) / 108 * 100)))
            
        # Volume confirmation: check if volume supports the trend
        vol_20 = df['Volume'].tail(20).mean() if len(df) >= 20 else df['Volume'].mean()
        vol_5 = df['Volume'].tail(5).mean()
        vol_rising = vol_5 > vol_20 if vol_20 > 0 else True
        
        if score >= 80 and vol_rising:
            confidence = "High"
        elif score >= 80 and not vol_rising:
            confidence = "Medium (Low Volume Warning)"
        elif score >= 50:
            confidence = "Medium"
        else:
            confidence = "Low"
            
        cmp = float(df['Close'].iloc[-1])
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else cmp
        day_change_pct = ((cmp - prev_close) / prev_close) * 100
        
        buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(df, cmp, indicators)
        
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
            "score": score,
            "confidence": confidence,
            "vol_rising": vol_rising,
            "buy_price": buy_price,
            "exit_price": exit_price,
            "target_price": target_price
        }
    except Exception as e:
        print(f"Error in VPA scan for {symbol}: {e}")
        return None

def scan_vpa_ma_squeeze(symbol: str, df: pd.DataFrame, indicators: dict = None) -> dict | None:
    """
    Scans for stocks where Daily VPA Major, Mid, and Minor are green,
    and the 10, 21, and 50 SMA are tightly clustered (< 6% gap).
    """
    if df is None or len(df) < 200:
        return None
        
    try:
        # Check Daily VPA
        daily_trends = calc_vpa_trends(df)
        if daily_trends['major'] != 1 or (daily_trends['mid'] != 1 and daily_trends['minor'] != 1):
            return None # Must be green for Major, and at least one of Mid/Minor must be green.
            
        # Calculate SMAs
        close_series = df['Close']
        sma10 = float(close_series.rolling(window=10).mean().iloc[-1])
        sma21 = float(close_series.rolling(window=21).mean().iloc[-1])
        sma50 = float(close_series.rolling(window=50).mean().iloc[-1])
        sma200 = float(close_series.rolling(window=200).mean().iloc[-1])
        cmp = float(close_series.iloc[-1])
        
        if pd.isna(sma10) or pd.isna(sma21) or pd.isna(sma50) or pd.isna(sma200):
            return None
            
        # Squeeze logic
        ma_list = [sma10, sma21, sma50]
        max_ma = max(ma_list)
        min_ma = min(ma_list)
        
        # Squeeze: (max - min) / min < 0.08
        if (max_ma - min_ma) / min_ma > 0.08:
            return None
            
        # Ensure price hasn't already broken out (price should be near the squeeze)
        if cmp > max_ma * 1.04:  # If price is > 4% above the highest MA, it's already running
            return None

            
        prev_close = float(df['Close'].iloc[-2]) if len(df) >= 2 else cmp
        day_change_pct = ((cmp - prev_close) / prev_close) * 100
        
        buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(df, cmp, indicators)
        
        from config import get_company_name
        company_name = get_company_name(symbol)
        
        return {
            "symbol": symbol.strip().upper(),
            "company_name": company_name,
            "cmp": round(cmp, 2),
            "day_change_pct": round(day_change_pct, 2),
            "volume": int(df['Volume'].iloc[-1]),
            "sma10": round(sma10, 2),
            "sma21": round(sma21, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "ma_gap_pct": round(((max_ma - min_ma) / min_ma) * 100, 2),
            "compression_score": round(((max_ma - min_ma) / min_ma) * 100, 2),
            "dist_to_200_pct": round(((sma200 - max_ma) / max_ma) * 100, 2),
            "buy_price": buy_price,
            "exit_price": exit_price,
            "target_price": target_price
        }
    except Exception as e:
        print(f"Error in VPA Squeeze scan for {symbol}: {e}")
        return None

def scan_structural_vcp(symbol: str, df: pd.DataFrame, lookback: int = 120, pivot_tolerance: float = 0.05, max_dist_to_pivot: float = 0.10, indicators: dict = None) -> dict | None:
    """
    Scans for a structural Volatility Contraction Pattern (VCP).
    Looks for a flat top resistance (pivot), successive higher lows, and volume dry-up.
    """
    if df is None or len(df) < lookback:
        return None
        
    try:
        # Use module-level scipy import instead of per-call import
        if _scipy_find_peaks is None:
            print("scipy is required for structural VCP scan. Please run: pip install scipy")
            return None
        find_peaks = _scipy_find_peaks
        
        df_recent = df.tail(lookback).copy()
        
        # 1. Find Pivot Resistance
        highs = df_recent['High'].values
        max_high_idx = np.argmax(highs)
        
        if max_high_idx >= len(highs) - 5:
            return None # Pivot is too recent, this is just a run-up, not a base
            
        pivot_price = highs[max_high_idx]
        
        # 2. Check current price proximity to pivot
        cmp = float(df_recent['Close'].iloc[-1])
        if cmp > pivot_price * 1.05 or cmp < pivot_price * (1 - max_dist_to_pivot):
            return None # Price must be tight near pivot
            
        # 3. Find Contractions (Lows)
        lows = df_recent['Low'].values
        inverted_lows = -lows
        minima_indices, properties = find_peaks(inverted_lows, prominence=pivot_price*0.02, distance=5)
        
        valid_minima = [idx for idx in minima_indices if idx > max_high_idx]
        
        if len(valid_minima) < 2:
            return None # We need at least 2 pullbacks after the pivot
            
        depths = [(pivot_price - lows[idx])/pivot_price for idx in valid_minima]
        
        # Last contraction must be under 8% (tighter = better VCP per Minervini rules)
        if depths[-1] > 0.08:
            return None
        # Contractions must be getting progressively shallower (each pullback tighter than last)
        if len(depths) >= 2 and depths[-1] >= depths[-2]:
            return None  # Last pullback is not shallower — not a true VCP compression
            
        # 4. Volume Accumulation
        df_recent['PriceChange'] = df_recent['Close'].diff()
        up_vol = df_recent[df_recent['PriceChange'] > 0]['Volume'].sum()
        down_vol = df_recent[df_recent['PriceChange'] < 0]['Volume'].sum()
        
        if up_vol <= down_vol:
            return None # No institutional accumulation
            
        # 5. Volume Dry-Up (VDU)
        if len(df) >= 50:
            vol_50d_avg = df['Volume'].tail(50).mean()
        else:
            vol_50d_avg = df_recent['Volume'].mean()
            
        vol_5d_avg = df_recent['Volume'].tail(5).mean()
        
        if vol_5d_avg > vol_50d_avg * 0.8:
            return None # Volume hasn't dried up
            
        from config import get_company_name
        company_name = get_company_name(symbol)
        
        score = round((1.0 - (vol_5d_avg / vol_50d_avg)) * 100, 1)
        
        buy_price, exit_price, target_price, support, resistance = calculate_trade_levels(df, cmp, indicators)
        confidence = "High" if len(valid_minima) >= 3 and depths[-1] < 0.05 else "Medium"
        
        base_rec = (
            f"Structural VCP found! Pivot resistance at ₹{pivot_price:.2f}. "
            f"Formed {len(valid_minima)} contractions. "
            f"Volume dried up to {vol_5d_avg/vol_50d_avg*100:.0f}% of 50d avg. "
            f"Buy Range: [₹{buy_price:.2f} to ₹{cmp:.2f}] (Nearest Support is ₹{support:.2f}). "
            f"Set stop loss securely below support at ₹{exit_price:.2f} "
            f"with a target near overhead resistance at ₹{target_price:.2f}."
        )
        
        recommendation = compute_rich_analysis(df, symbol, "Structural VCP", base_rec, indicators=indicators)
            
        return {
            "symbol": symbol.strip().upper(),
            "company_name": company_name,
            "pivot_price": round(pivot_price, 2),
            "contractions": len(valid_minima),
            "depths_pct": [round(d*100, 1) for d in depths],
            "vol_50d": int(vol_50d_avg),
            "vol_5d": int(vol_5d_avg),
            "cmp": cmp,
            "buy_price": buy_price,
            "exit_price": exit_price,
            "target_price": target_price,
            "day_change_pct": round(((cmp - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100, 2) if len(df)>=2 else 0.0,
            "score": score,
            "confidence": confidence,
            "recommendation": recommendation
        }
    except Exception as e:
        print(f"Error in structural VCP scan for {symbol}: {e}")
        return None


# ==========================================
# VOLUME PROFILE SCANNER
# ==========================================

PIVOT_LENGTH = 20
PROFILE_LEVELS = 25
BUY_ZONE_THRESHOLD = 35

def find_last_pivot_swing(df, pivot_length=20):
    if _scipy_argrelextrema is None:
        return None

    highs = df['High'].values
    lows = df['Low'].values

    pivot_highs = _scipy_argrelextrema(
        highs,
        np.greater_equal,
        order=pivot_length
    )[0]

    pivot_lows = _scipy_argrelextrema(
        lows,
        np.less_equal,
        order=pivot_length
    )[0]

    pivots = sorted(
        [(i, 'H') for i in pivot_highs] +
        [(i, 'L') for i in pivot_lows],
        key=lambda x: x[0]
    )

    if len(pivots) < 2:
        return None

    start_idx = pivots[-2][0]
    end_idx = pivots[-1][0]

    if start_idx >= end_idx:
        return None

    return df.iloc[start_idx:end_idx + 1].copy()

def build_volume_profile(swing_df, levels=25):
    low_price = swing_df['Low'].min()
    high_price = swing_df['High'].max()

    if high_price == low_price:
        return None

    bins = np.linspace(low_price, high_price, levels + 1)
    volume_profile = np.zeros(levels)

    lows = swing_df['Low'].values
    highs = swing_df['High'].values
    vols = swing_df['Volume'].values

    for candle_low, candle_high, vol in zip(lows, highs, vols):
        touched = np.where(
            (bins[:-1] <= candle_high) &
            (bins[1:] >= candle_low)
        )[0]

        if len(touched) > 0:
            volume_profile[touched] += vol / len(touched)

    return bins, volume_profile

def calculate_vp_levels(bins, profile, value_area_pct=0.68):
    poc_idx = np.argmax(profile)
    poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

    total_volume = profile.sum()
    if total_volume == 0:
        return poc, bins[0], bins[-1]
    target = total_volume * value_area_pct
    cumulative = profile[poc_idx]

    low_idx = poc_idx
    high_idx = poc_idx

    while cumulative < target:
        can_go_up = high_idx < len(profile) - 1
        can_go_down = low_idx > 0
        
        if not can_go_up and not can_go_down:
            break
        
        vol_above = profile[high_idx + 1] if can_go_up else -1
        vol_below = profile[low_idx - 1] if can_go_down else -1

        if vol_above >= vol_below:
            high_idx += 1
            cumulative += vol_above
        else:
            low_idx -= 1
            cumulative += vol_below

    val = bins[low_idx]
    vah = bins[high_idx + 1]
    return poc, val, vah

def classify_vp(price, poc, val, vah):
    if vah <= val:
        return None, None

    position_pct = ((price - val) / (vah - val)) * 100

    if price < val:
        zone = '⛔ Avoid (Below Support)'
    elif price > vah:
        zone = '🚀 Overextended (Breakout)'
    elif price >= poc:
        zone = '🎯 Take Profit (Near Resistance)'
    else:
        if poc > val:
            distance_pct = (price - val) / (poc - val)
            if distance_pct <= 0.5:
                zone = '✅ Can Buy (Near Support)'
            else:
                zone = '⏳ Hold (Moving to Target)'
        else:
            zone = '✅ Can Buy (Near Support)'

    return round(position_pct, 2), zone

def analyze_vp_timeframe(df):
    if df is None or len(df) < 50:
        return None
    swing = find_last_pivot_swing(df, PIVOT_LENGTH)
    if swing is None:
        return None
    result = build_volume_profile(swing, PROFILE_LEVELS)
    if result is None:
        return None
    bins, profile = result
    poc, val, vah = calculate_vp_levels(bins, profile)
    price = float(df['Close'].iloc[-1])
    position_pct, zone = classify_vp(price, poc, val, vah)
    if zone is None:
        return None
    return {
        'price': price,
        'poc': poc,
        'val': val,
        'vah': vah,
        'position_pct': position_pct,
        'zone': zone
    }

def scan_volume_profile(symbol: str, df: pd.DataFrame, market_cap: float) -> dict | None:
    if df is None or len(df) < 100:
        return None
    close_price = float(df['Close'].iloc[-1])
    if close_price < 100 or (market_cap > 0 and market_cap < 2000):
        return None

    try:
        df_weekly = df.set_index('Date').resample('W-FRI').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna().reset_index()
        df_monthly = df.set_index('Date').resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna().reset_index()

        daily_result = analyze_vp_timeframe(df)
        weekly_result = analyze_vp_timeframe(df_weekly)
        monthly_result = analyze_vp_timeframe(df_monthly)

        if not daily_result and not weekly_result and not monthly_result:
            return None

        return {
            'daily': daily_result,
            'weekly': weekly_result,
            'monthly': monthly_result,
            'cmp': close_price,
            'symbol': symbol,
            'market_cap_cr': market_cap
        }
    except Exception as e:
        return None

def scan_support_rsi(symbol: str, df: pd.DataFrame, market_cap: float = 0.0, 
                     rsi_threshold: float = 35.0, support_proximity_pct: float = 3.0) -> dict | None:
    """
    Scans for stocks that are near a historical support level AND have oversold RSI.
    
    Support Detection:
      1. Find swing lows (pivot points where price bounced) over the last 6-12 months
      2. Cluster nearby lows into support zones (within 1.5% of each other)
      3. Check if current price is within support_proximity_pct of a support zone
      4. Count touches at each zone — more touches = stronger support
    
    RSI Filter:
      - RSI(14) <= rsi_threshold (default 35 = oversold territory)
    
    Returns dict with support details or None if conditions not met.
    """
    if df is None or len(df) < 100:
        return None
    
    try:
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        cmp = float(close[-1])
        
        # Price and market cap filters
        if cmp < 100:
            return None
        if market_cap > 0 and market_cap < 2000:
            return None
        
        # --- RSI(14) Calculation ---
        delta = pd.Series(close).diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        ema_gain = gain.ewm(com=13, adjust=False).mean()
        ema_loss = loss.ewm(com=13, adjust=False).mean()
        rs = ema_gain / (ema_loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        current_rsi = float(rsi_series.iloc[-1])
        
        if pd.isna(current_rsi) or current_rsi > rsi_threshold:
            return None
        
        # --- Swing Low (Support) Detection ---
        # Use last 250 bars (~1 year of daily data)
        lookback = min(len(df), 250)
        lows_arr = low[-lookback:]
        dates_arr = df['Date'].values[-lookback:] if 'Date' in df.columns else list(range(lookback))
        
        # Find swing lows with a window of 5 bars on each side
        swing_window = 5
        swing_lows = []
        for i in range(swing_window, len(lows_arr) - swing_window):
            is_swing_low = True
            for j in range(1, swing_window + 1):
                if lows_arr[i] > lows_arr[i - j] or lows_arr[i] > lows_arr[i + j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows.append({
                    'price': float(lows_arr[i]),
                    'index': i
                })
        
        if len(swing_lows) < 2:
            return None
        
        # --- Cluster swing lows into support zones ---
        # Sort by price and merge nearby levels (within 1.5% of each other)
        swing_lows.sort(key=lambda x: x['price'])
        support_zones = []
        current_zone = [swing_lows[0]]
        
        for i in range(1, len(swing_lows)):
            zone_avg = sum(s['price'] for s in current_zone) / len(current_zone)
            if abs(swing_lows[i]['price'] - zone_avg) / zone_avg <= 0.015:
                current_zone.append(swing_lows[i])
            else:
                support_zones.append(current_zone)
                current_zone = [swing_lows[i]]
        support_zones.append(current_zone)
        
        # --- Find the nearest support zone to current price ---
        best_zone = None
        best_distance_pct = float('inf')
        
        for zone in support_zones:
            zone_price = sum(s['price'] for s in zone) / len(zone)
            touches = len(zone)
            
            # Only consider zones BELOW current price (support, not resistance)
            if zone_price >= cmp * 1.01:
                continue
            
            distance_pct = ((cmp - zone_price) / zone_price) * 100
            
            if distance_pct <= support_proximity_pct and distance_pct < best_distance_pct:
                best_distance_pct = distance_pct
                best_zone = {
                    'support_price': round(zone_price, 2),
                    'touches': touches,
                    'distance_pct': round(distance_pct, 2)
                }
        
        if best_zone is None:
            return None
        
        # --- Calculate SMA positions ---
        sma20 = float(pd.Series(close).rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        sma50 = float(pd.Series(close).rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = float(pd.Series(close).rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        
        above_20sma = cmp > sma20 if sma20 else False
        above_50sma = cmp > sma50 if sma50 else False
        above_200sma = cmp > sma200 if sma200 else False
        
        # --- CCI(14) ---
        tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
        sma_tp = tp.rolling(14).mean()
        mad = tp.rolling(14).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        cci_series = (tp - sma_tp) / (0.015 * mad + 1e-9)
        current_cci = float(cci_series.iloc[-1]) if not pd.isna(cci_series.iloc[-1]) else 0.0
        
        # --- Day change ---
        prev_close = float(close[-2]) if len(close) >= 2 else cmp
        day_change_pct = ((cmp - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
        
        # --- Volume ---
        current_volume = int(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0
        
        # --- Score: higher = better setup ---
        # More touches = stronger support, lower RSI = more oversold, closer to support = better entry
        score = (best_zone['touches'] * 15) + ((rsi_threshold - current_rsi) * 2) + (support_proximity_pct - best_zone['distance_pct']) * 5
        score = round(min(score, 100), 1)
        
        # --- Trade levels ---
        buy_price = round(best_zone['support_price'] * 1.005, 2)  # Just above support
        exit_price = round(best_zone['support_price'] * 0.97, 2)  # 3% below support
        target_price = round(cmp * 1.12, 2)  # 12% upside target (mean reversion)
        
        # Confidence
        if best_zone['touches'] >= 3 and current_rsi <= 30:
            confidence = "High"
        elif best_zone['touches'] >= 2 and current_rsi <= 35:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        from config import get_company_name
        return {
            'symbol': symbol,
            'company_name': get_company_name(symbol),
            'cmp': round(cmp, 2),
            'day_change_pct': round(day_change_pct, 2),
            'rsi': round(current_rsi, 2),
            'cci': round(current_cci, 2),
            'support_price': best_zone['support_price'],
            'support_touches': best_zone['touches'],
            'distance_to_support_pct': best_zone['distance_pct'],
            'above_20sma': above_20sma,
            'above_50sma': above_50sma,
            'above_200sma': above_200sma,
            'volume': current_volume,
            'score': score,
            'buy_price': buy_price,
            'exit_price': exit_price,
            'target_price': target_price,
            'confidence': confidence,
            'recommendation': f"Near {best_zone['touches']}-touch support at ₹{best_zone['support_price']:,.2f} ({best_zone['distance_pct']:.1f}% away). RSI oversold at {current_rsi:.1f}. Wait for bounce confirmation before entry.",
            'market_cap_cr': market_cap
        }
    except Exception as e:
        return None

def _compute_bb_squeeze_for_tf(df: pd.DataFrame, lookback_bars: int, max_width_pct: float) -> tuple:
    """
    Compute BB squeeze for a specific timeframe with the correct lookback window.

    Args:
        df            : OHLCV DataFrame for the timeframe
        lookback_bars : Number of bars to define "6 months" for this timeframe
                        Daily=126, Weekly=26, Monthly=6
        max_width_pct : Absolute upper limit on BB width to be considered a squeeze.
                        Prevents flagging visually wide bands as squeezes.
                        E.g. 0.20 means width > 20% of price => never a squeeze.

    Returns:
        (is_squeeze: bool, bb_width: float, min_width: float)
    """
    if df is None or len(df) < 20:
        return False, 0.0, 0.0
    try:
        close = df['Close'].astype(float)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, float('nan'))

        current_width = float(bb_width.iloc[-1]) if not pd.isna(bb_width.iloc[-1]) else None
        if current_width is None:
            return False, 0.0, 0.0

        # Absolute max-width guard: if bands are obviously wide, skip the squeeze check
        if current_width > max_width_pct:
            return False, round(current_width, 4), 0.0

        # 6-month rolling minimum using timeframe-correct lookback
        actual_lookback = min(lookback_bars, len(df))
        rolling_min = bb_width.rolling(actual_lookback).min()
        min_width = float(rolling_min.iloc[-1]) if not pd.isna(rolling_min.iloc[-1]) else None
        if min_width is None or min_width <= 0:
            return False, round(current_width, 4), 0.0

        # Squeeze = current width is within 115% of the historical minimum
        is_squeeze = current_width <= min_width * 1.15
        return bool(is_squeeze), round(current_width, 4), round(min_width, 4)
    except Exception:
        return False, 0.0, 0.0


def _compute_bb_squeeze_score(
    d_squeeze: bool, w_squeeze: bool, m_squeeze: bool,
    d_width: float, d_min_width: float,
    above_50: bool,
    df_daily: pd.DataFrame
) -> tuple:
    """
    Compute a 0-100 confidence score for a BB Squeeze setup.

    Scoring breakdown:
      - Timeframe signals   : up to 40 pts  (Daily=15, Weekly=15, Monthly=10)
      - Band tightness      : up to 25 pts  (daily: how close to hist. min)
      - Above 50 DMA        : 10 pts
      - RSI neutral (40-60) : up to 15 pts
      - Volume contracting  : up to 10 pts

    Returns:
        (score: int, confidence: str, breakdown: list[str])
    """
    score = 0
    breakdown = []

    # 1. Timeframe signals (0–40 pts)
    if d_squeeze:
        score += 15
        breakdown.append("Daily Squeeze (+15)")
    if w_squeeze:
        score += 15
        breakdown.append("Weekly Squeeze (+15)")
    if m_squeeze:
        score += 10
        breakdown.append("Monthly Squeeze (+10)")

    # 2. Daily band tightness (0–25 pts)
    # How close is current width to the historical minimum?
    # At exactly min: full 25 pts. At 115% of min (threshold): 0 pts.
    if d_squeeze and d_width > 0 and d_min_width > 0:
        try:
            threshold = d_min_width * 1.15
            span = threshold - d_min_width
            if span > 0:
                tightness = 1.0 - (d_width - d_min_width) / span
                tightness = max(0.0, min(1.0, tightness))
            else:
                tightness = 1.0
            tight_pts = round(tightness * 25)
            score += tight_pts
            breakdown.append(f"Band Tightness (+{tight_pts})")
        except Exception:
            pass

    # 3. Above 50 DMA (0–10 pts)
    if above_50:
        score += 10
        breakdown.append("Above 50 DMA (+10)")

    # 4. RSI position (0–15 pts) — computed from daily data
    try:
        close = df_daily['Close'].astype(float)
        delta = close.diff()
        avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])
        if 40 <= rsi <= 60:
            score += 15
            breakdown.append(f"RSI Neutral {rsi:.0f} (+15)")
        elif 35 <= rsi < 40 or 60 < rsi <= 65:
            score += 8
            breakdown.append(f"RSI Near-Neutral {rsi:.0f} (+8)")
        else:
            breakdown.append(f"RSI Extreme {rsi:.0f} (+0)")
    except Exception:
        pass

    # 5. Volume contraction (0–10 pts)
    try:
        vol = df_daily['Volume'].astype(float)
        avg_vol_50 = float(vol.rolling(50).mean().iloc[-1])
        recent_vol  = float(vol.rolling(5).mean().iloc[-1])
        if avg_vol_50 > 0:
            if recent_vol < avg_vol_50 * 0.65:
                score += 10
                breakdown.append("Volume Dry-Up (+10)")
            elif recent_vol < avg_vol_50 * 0.85:
                score += 5
                breakdown.append("Volume Contracting (+5)")
    except Exception:
        pass

    score = min(100, max(0, score))

    if score >= 80:
        confidence = "🔥 Very High"
    elif score >= 60:
        confidence = "🟢 High"
    elif score >= 40:
        confidence = "🟡 Medium"
    elif score >= 20:
        confidence = "🟠 Low"
    else:
        confidence = "⚪ Very Low"

    return score, confidence, breakdown


def scan_bb_squeeze(symbol: str, df_daily: pd.DataFrame, df_weekly: pd.DataFrame, df_monthly: pd.DataFrame, market_cap: float = 0.0) -> dict | None:
    """
    Scans a stock across Daily, Weekly, and Monthly timeframes to find active Bollinger Band Squeezes.

    v2 Fixes:
      - Timeframe-appropriate lookback (Daily=126, Weekly=26, Monthly=6 bars)
      - Absolute max-width guards to eliminate false positives on expanded bands
      - Confidence score (0-100) + label (Very High / High / Medium / Low / Very Low)
    """
    if df_daily is None or len(df_daily) < 50:
        return None

    try:
        from config import get_company_name

        cmp = float(df_daily['Close'].iloc[-1])
        if cmp < 100 or (market_cap > 0 and market_cap < 2000):
            return None

        # --- Per-timeframe squeeze with correct lookback & width guard ---
        d_squeeze, d_width, d_min_w = _compute_bb_squeeze_for_tf(df_daily,   lookback_bars=126, max_width_pct=0.20)
        w_squeeze, w_width, _       = _compute_bb_squeeze_for_tf(df_weekly,   lookback_bars=26,  max_width_pct=0.30)
        m_squeeze, m_width, _       = _compute_bb_squeeze_for_tf(df_monthly,  lookback_bars=6,   max_width_pct=0.40)

        # Must have at least one valid squeeze signal
        if not (d_squeeze or w_squeeze or m_squeeze):
            return None

        # Bullish filter: Price > 50 DMA
        close_d = df_daily['Close'].astype(float)
        sma50   = float(close_d.rolling(50).mean().iloc[-1]) if len(df_daily) >= 50 else 0.0
        above_50 = (cmp > sma50) if sma50 > 0 else False

        # Day change %
        prev_close    = float(df_daily['Close'].iloc[-2]) if len(df_daily) >= 2 else cmp
        day_change_pct = ((cmp - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

        # --- Confidence Score ---
        squeeze_score, confidence, score_breakdown = _compute_bb_squeeze_score(
            d_squeeze=d_squeeze, w_squeeze=w_squeeze, m_squeeze=m_squeeze,
            d_width=d_width, d_min_width=d_min_w,
            above_50=above_50,
            df_daily=df_daily
        )

        return {
            'symbol':           symbol.strip().upper(),
            'company_name':     get_company_name(symbol),
            'cmp':              round(cmp, 2),
            'day_change_pct':   round(day_change_pct, 2),
            'daily_squeeze':    d_squeeze,
            'weekly_squeeze':   w_squeeze,
            'monthly_squeeze':  m_squeeze,
            'daily_bb_width':   d_width,
            'weekly_bb_width':  w_width,
            'monthly_bb_width': m_width,
            'above_50dma':      above_50,
            'squeeze_score':    squeeze_score,
            'confidence':       confidence,
            'score_breakdown':  ' | '.join(score_breakdown),
            'market_cap_cr':    market_cap
        }
    except Exception:
        return None

def scan_ema_support(symbol: str, df: pd.DataFrame, max_dist_9ema: float = 3.0, max_dist_21ema: float = 5.0) -> dict | None:
    if df is None or len(df) < 22:
        return None
    try:
        from config import get_company_name
        cmp = float(df['Close'].iloc[-1])
        if cmp < 100:
            return None
        
        ema9 = df['Close'].ewm(span=9, adjust=False).mean()
        ema21 = df['Close'].ewm(span=21, adjust=False).mean()
        today_ema9 = float(ema9.iloc[-1])
        today_ema21 = float(ema21.iloc[-1])
        yest_ema9 = float(ema9.iloc[-2])
        yest_ema21 = float(ema21.iloc[-2])
        prev_close = float(df['Close'].iloc[-2])
        day_change_pct = ((cmp - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
        
        dist_9ema = (cmp - today_ema9) / today_ema9 * 100
        dist_21ema = (cmp - today_ema21) / today_ema21 * 100
        
        # Support logic: Distance must be within threshold AND price must be above or very close to the EMA (taking support).
        # We'll allow slight dips below the EMA (e.g. -0.5%) for support.
        support_9 = (-0.5 <= dist_9ema <= max_dist_9ema)
        support_21 = (-0.5 <= dist_21ema <= max_dist_21ema)
        
        if not (support_9 or support_21):
            return None
            
        crossover = False
        crossover_text = ""
        if (yest_ema9 <= yest_ema21) and (today_ema9 > today_ema21):
            crossover = True
            crossover_text = "🔥 9 EMA just crossed above 21 EMA today!"
        elif (ema9.iloc[-3] <= ema21.iloc[-3]) and (yest_ema9 > yest_ema21):
            crossover = True
            crossover_text = "🟢 9 EMA crossed above 21 EMA yesterday."
            
        if support_9 and support_21:
            setup = "9 & 21 EMA Support"
        elif support_9:
            setup = "9 EMA Support"
        else:
            setup = "21 EMA Support"
            
        buy_price = round(today_ema21 if support_21 else today_ema9, 2)
        sl = round(buy_price * 0.96, 2)
        target = round(cmp * 1.12, 2)
        score = 85.0 if crossover else 70.0
        conf = "High" if crossover else "Medium"
        
        rec = f"{crossover_text} Stock is taking solid support at the {setup}. Accumulate near ₹{buy_price:,.2f} with tight SL below ₹{sl:,.2f}. Target ₹{target:,.2f}."
        
        return {
            'symbol': symbol.strip().upper(),
            'company_name': get_company_name(symbol),
            'cmp': round(cmp, 2),
            'day_change_pct': round(day_change_pct, 2),
            'setup': setup,
            'dist_9ema': round(dist_9ema, 2),
            'dist_21ema': round(dist_21ema, 2),
            'crossover': crossover,
            'score': score,
            'buy_price': buy_price,
            'exit_price': sl,
            'target_price': target,
            'confidence': conf,
            'recommendation': rec
        }
    except Exception as e:
        print(f"Error in EMA Support scan for {symbol}: {e}")
        return None

def scan_stage_analysis(symbol: str, df: pd.DataFrame, bench_ret: float) -> dict | None:
    """
    Minervini Trend Template - Stage Analyzer.
    Pine Script Translation:
    Stage 2 requires 8 conditions.
    Returns: stage, score, template string (e.g. '7/8').
    """
    if df is None or len(df) < 252:
        return None
        
    try:
        from config import get_company_name
        cmp = float(df['Close'].iloc[-1])
        
        # Calculate MAs
        ma50_series = df['Close'].rolling(window=50).mean()
        ma150_series = df['Close'].rolling(window=150).mean()
        ma200_series = df['Close'].rolling(window=200).mean()
        
        if pd.isna(ma200_series.iloc[-1]):
            return None
            
        c = float(df['Close'].iloc[-1])
        ma50 = float(ma50_series.iloc[-1])
        ma150 = float(ma150_series.iloc[-1])
        ma200 = float(ma200_series.iloc[-1])
        
        # ma200old = ta.sma(close, 200)[21] -> 21 trading days ago
        if len(ma200_series) >= 200 + 21:
            ma200old = float(ma200_series.iloc[-22])
        else:
            return None
            
        lo52 = float(df['Low'].iloc[-252:].min())
        hi52 = float(df['High'].iloc[-252:].max())
        
        if len(df) >= 127:
            cOld = float(df['Close'].iloc[-127])
        else:
            return None
            
        sRet = (c - cOld) / cOld if cOld > 0 else 0
        
        # 8 Trend Template criteria
        c1 = c > ma150 and c > ma200
        c2 = ma150 > ma200
        c3 = ma200 > ma200old  # 200MA rising >= 1 month
        c4 = ma50 > ma150 and ma50 > ma200
        c5 = c > ma50
        c6 = c >= lo52 * 1.30
        c7 = c >= hi52 * 0.75
        c8 = sRet > bench_ret
        
        score = sum([c1, c2, c3, c4, c5, c6, c7, c8])
        
        ma200Rising = ma200 > ma200old
        ma200Falling = ma200 < ma200old * 0.998
        
        stage = 1
        if score >= 7 and c1 and c2 and c3:
            stage = 2
        elif c < ma200 and ma200Falling:
            stage = 4
        elif c > ma200 and not ma200Rising:
            stage = 3
        else:
            stage = 1
            
        return {
            'symbol': symbol.strip().upper(),
            'company_name': get_company_name(symbol),
            'cmp': round(c, 2),
            'stage': stage,
            'score': int(score),
            'template_str': f"{int(score)}/8",
            'sret': round(sRet, 4),
            'lo52': round(lo52, 2),
            'hi52': round(hi52, 2)
        }
    except Exception as e:
        print(f"Error in stage analysis for {symbol}: {e}")
        return None

def scan_near_30sma(symbol: str, df: pd.DataFrame, max_dist_pct: float = 3.0) -> dict | None:
    """
    Scans for stocks where the current close is just above the 30-day SMA,
    but within `+/- max_dist_pct` (e.g. 3%) of it (can be slightly below or above).
    """
    try:
        if len(df) < 30:
            return None
        
        # We need SMA 30
        if 'SMA_30' not in df.columns:
            df_copy = df.copy()
            df_copy['SMA_30'] = df_copy['Close'].rolling(window=30).mean()
        else:
            df_copy = df
            
        today_close = df_copy['Close'].iloc[-1]
        today_sma30 = df_copy['SMA_30'].iloc[-1]
        
        if pd.isna(today_sma30):
            return None
            
        # Condition: Price must be above SMA30, but not more than 3% above it
        dist_pct = ((today_close - today_sma30) / today_sma30) * 100.0
        
        if -max_dist_pct <= dist_pct <= max_dist_pct:
            today_vol = int(df_copy['Volume'].iloc[-1])
            prev_close = df_copy['Close'].iloc[-2] if len(df_copy) >= 2 else today_close
            day_change_pct = ((today_close - prev_close) / prev_close) * 100.0
            
            return {
                'symbol': symbol,
                'cmp': round(today_close, 2),
                'day_change_pct': round(day_change_pct, 2),
                'volume': today_vol,
                'sma30': round(today_sma30, 2),
                'dist_pct': round(dist_pct, 2)
            }
            
        return None
    except Exception as e:
        print(f"Error in scan_near_30sma for {symbol}: {e}")
        return None

