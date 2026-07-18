# scan_orchestrator.py
# Per-symbol scan pipeline extracted from app.py (gap-up, SMA setups, Minervini, VDU, WT, VCS, VPA).

from datetime import datetime

import pandas as pd

from config import IST_TIMEZONE, get_company_name
from indicators import precompute_indicators
from scanner import (
    compute_rich_analysis,
    scan_stock,
    scan_vcs,
    scan_vpa_trend,
    scan_wt_cross,
    scan_monthly_early_stage2,
    scan_stage_analysis,
    scan_volume_profile,
    scan_ema_support,
    scan_support_rsi,
    scan_monthly_momentum,
    scan_weekly_momentum
)
from vcp_minervini import MinerviniVCPAnalyzer, VCPConfig
from zanger_scanner import scan_zanger, ZangerConfig
from local_cache_manager import resample_ohlcv

def process_single_symbol(sym, df, benchmark_df, open_price_map, close_price_map, high_price_map, low_price_map, volume_map,
                          min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes,
                          min_signal_str, above_50dma_only, above_200dma_only, vcp_max_tightness,
                          sma20_lower_bound, sma20_upper_bound, sma50_lower_bound, sma50_upper_bound, sma20_min_volume, sma_timeframe, scan_mode="all"):
    res = {
        "failed": False,
        "gapup": None,
        "above_ma": None,
        "support_ma": None,
        "crossover_ma": None,
        "minervini": None,
        "flagged": None,

        "wt": None,
        "vcs": None,
        "structural_vcp": None,
        "vpa": None,
        
        "zanger": None,
        "stage2": None,
        "minervini_vcp": None,
        "stage_analysis": None,
        "volume_profile": None,
        "ema_support": None,
        "support_rsi": None,
        "rsi_wt_combo": None, # Wait, is there a scan_rsi_wt_combo?
        "monthly_momentum": None,
        "weekly_momentum": None
    }
    if df is None or len(df) < 5:
        res["failed"] = True
        return res
        
    df = df.sort_values('Date').reset_index(drop=True)
    last_df_date = df['Date'].iloc[-1].date()
    today_date = datetime.now(IST_TIMEZONE).date()
    
    # Only append the live quote if we are scanning on a Daily timeframe.
    # Otherwise, injecting a single day's quote into a Weekly/Monthly dataframe ruins the final candle.
    if last_df_date < today_date and "Daily" in sma_timeframe:
        sym_clean = sym.strip().upper()
        if sym_clean in open_price_map and sym_clean in close_price_map:
            live_close = close_price_map[sym_clean]
            live_vol = volume_map.get(sym_clean, 0)
            last_hist_close = df['Close'].iloc[-1]
            last_hist_vol = df['Volume'].iloc[-1]
            
            import math
            # Prevent appending weekend/holiday duplicate candles.
            # If the 1d "live" quote from yfinance matches the last historical candle's
            # close, it means the market is closed (or price is entirely flat).
            if not math.isclose(live_close, last_hist_close, rel_tol=1e-4):
                new_row = {
                    'Date': pd.to_datetime(today_date),
                    'Open': open_price_map[sym_clean],
                    'High': high_price_map.get(sym_clean, close_price_map[sym_clean]),
                    'Low': low_price_map.get(sym_clean, close_price_map[sym_clean]),
                    'Close': close_price_map[sym_clean],
                    'Volume': volume_map.get(sym_clean, 0)
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
    today_close_val = df['Close'].iloc[-1]
    if today_close_val <= 200.0:
        return res
    
    # =====================================================================
    # PRE-COMPUTE ALL INDICATORS ONCE (eliminates 5-8x redundant recalc)
    # =====================================================================
    from indicators import precompute_indicators
    ind = precompute_indicators(df)
    # Use the enriched DataFrame from pre-computation for all subsequent work
    if ind is not None and 'df' in ind:
        df = ind['df']
    
    today_open_val = float(df['Open'].iloc[-1])
    today_close_val = float(df['Close'].iloc[-1])
    yesterday_close_val = float(df['Close'].iloc[-2]) if len(df) >= 2 else today_open_val
    if scan_mode == "all":
        if today_open_val > yesterday_close_val and today_close_val > yesterday_close_val and today_close_val >= (today_open_val * 0.97):
            gap_pct = (today_open_val - yesterday_close_val) / yesterday_close_val * 100
            # EXPERT FIX: Larger gaps have GREATER continuation potential (High Tight Flag logic)
            # Original logic was inverted - a big 8%+ gap deserves a bigger target, not smaller
            if gap_pct >= 8.0:
                target_multiplier = 1.15; target_pct_str = "+15%"   # HTF - massive continuation
            elif gap_pct >= 5.0:
                target_multiplier = 1.10; target_pct_str = "+10%"   # Strong gap
            else:
                target_multiplier = 1.06; target_pct_str = "+6%"    # Modest gap, modest target
                
            gap_buy_price = round(min(today_open_val, yesterday_close_val) * 0.99, 2)  # Support = gap base (previous close)
            gap_exit_price = round(yesterday_close_val * 0.97, 2)  # Stop below gap fill level
            gap_target_price = round(today_close_val * target_multiplier, 2) 
            gap_confidence = "High (Gap-Up Momentum)" if gap_pct > 3.0 else "Medium (Gap-Up)"
            base_gap_rec = (f"Bullish gap-up breakout of {gap_pct:.2f}% on strong momentum. Buy near support ₹{gap_buy_price:.2f} "
                            f"with a stop loss below today's open price at ₹{gap_exit_price:.2f} "
                            f"targeting dynamic swing target ₹{gap_target_price:.2f} ({target_pct_str}).")
            gap_recommendation = compute_rich_analysis(df, sym, "Gap-Up", base_gap_rec, indicators=ind)
            res["gapup"] = {
                "symbol": sym.strip().upper(), "company_name": get_company_name(sym),
                "prev_close": yesterday_close_val, "open_price": today_open_val, "cmp": today_close_val,
                "gap_pct": round(gap_pct, 2), "volume": int(df['Volume'].iloc[-1]),
                "day_change_pct": round(((today_close_val - yesterday_close_val) / yesterday_close_val * 100), 2),
                "buy_price": gap_buy_price, "exit_price": gap_exit_price, "target_price": gap_target_price,
                "confidence": gap_confidence, "recommendation": gap_recommendation
            }
        
    # Use pre-computed SMAs from indicators — no need to recalculate
    df_ma = df  # Already has SMA20, SMA50, SMA65, SMA150, SMA200 from precompute_indicators()
    
    if len(df_ma) >= 200:
        today_row = df_ma.iloc[-1]; yesterday_row = df_ma.iloc[-2]
        c_val = float(today_row['Close']); l_val = float(today_row['Low'])
        sma20 = float(today_row['SMA20']); sma50 = float(today_row['SMA50'])
        sma65 = float(today_row['SMA65']); sma150 = float(today_row['SMA150'])
        sma200 = float(today_row['SMA200'])
        
        # Multi-Timeframe 20 & 50 SMA Strategy
        dist_20 = (c_val - sma20) / sma20 * 100 if sma20 else 0
        dist_50 = (c_val - sma50) / sma50 * 100 if sma50 else 0
        dist_200 = (c_val - sma200) / sma200 * 100 if sma200 else 0
        
        # Track if stock has already run >10% above 20 SMA or 50 SMA (overextended)
        is_overextended = (dist_20 > 10 or dist_50 > 10)

        def check_sma_conditions(d_frame):
            if len(d_frame) < 50:
                return False
            try:
                d_close = float(d_frame['Close'].iloc[-1])
                
                sma20_series = d_frame['Close'].rolling(window=20).mean()
                sma50_series = d_frame['Close'].rolling(window=50).mean()
                
                d_sma20 = float(sma20_series.iloc[-1])
                d_sma50 = float(sma50_series.iloc[-1])
                
                d_sma200 = float(d_frame['Close'].rolling(window=200).mean().iloc[-1]) if len(d_frame) >= 200 else float('nan')
                d_vol_sma20 = float(d_frame['Volume'].rolling(window=20).mean().iloc[-1])
                
                is_rounding = False
                if len(d_frame) >= 150:
                    left_avg = d_frame['Close'].iloc[-150:-100].mean()
                    mid_avg = d_frame['Close'].iloc[-100:-50].mean()
                    right_avg = d_frame['Close'].iloc[-50:].mean()
                    # Require meaningful dip: mid must be at least 5% below left to be a real cup/rounding bottom
                    dip_depth = (left_avg - mid_avg) / left_avg if left_avg > 0 else 0
                    if left_avg > mid_avg and right_avg > mid_avg and d_close > mid_avg * 1.05 and dip_depth >= 0.05:
                        is_rounding = True
                elif len(d_frame) >= 90:
                    left_avg = d_frame['Close'].iloc[-90:-60].mean()
                    mid_avg = d_frame['Close'].iloc[-60:-30].mean()
                    right_avg = d_frame['Close'].iloc[-30:].mean()
                    dip_depth = (left_avg - mid_avg) / left_avg if left_avg > 0 else 0
                    if left_avg > mid_avg and right_avg > mid_avg and d_close > mid_avg * 1.05 and dip_depth >= 0.05:
                        is_rounding = True
                        
                # 1. MA stacking: Close >= SMA20 >= SMA50 >= SMA200
                stacking_ok = (d_close >= d_sma20) and (d_sma20 >= d_sma50)
                if pd.notna(d_sma200) and not is_rounding:
                    stacking_ok = stacking_ok and (d_sma50 >= d_sma200)
                
                # Price-to-SMA gap constraints
                max_20_gap = 0.25 if is_rounding else 0.15
                max_50_gap = 0.35 if is_rounding else 0.20
                cond_20_gap = abs(d_close - d_sma20) / d_sma20 <= max_20_gap if pd.notna(d_sma20) else False
                cond_50_gap = abs(d_close - d_sma50) / d_sma50 <= max_50_gap if pd.notna(d_sma50) else False
                cond_200_gap = True  # Relaxed for strong uptrends
                
                # 3. Tightness over last 5 bars (10% for daily, 20% for rounding)
                high_5 = d_frame['High'].iloc[-5:].max()
                low_5 = d_frame['Low'].iloc[-5:].min()
                max_tightness = 0.20 if is_rounding else 0.10
                tightness_ok = ((high_5 - low_5) / low_5) <= max_tightness if low_5 > 0 else False
                
                # 2. Upward slope over 5 bars (not just 1)
                d_sma20_5ago = float(sma20_series.iloc[-5]) if len(sma20_series) >= 5 else d_sma20
                d_sma50_5ago = float(sma50_series.iloc[-5]) if len(sma50_series) >= 5 else d_sma50
                slope_ok = (d_sma20 > d_sma20_5ago) and (d_sma50 > d_sma50_5ago)
                
                # SMA bounds
                sma50_bounds_ok = True
                close_200_ok = True
                if not is_rounding:
                    sma50_bounds_ok = (d_sma50 <= d_sma200 * sma50_upper_bound) and (d_sma50 >= d_sma200 * sma50_lower_bound)
                    close_200_ok = (d_close >= d_sma200 * 0.98)
                
                condition_daily = (
                    pd.notna(d_sma20) and pd.notna(d_sma50) and pd.notna(d_close) and pd.notna(d_sma200) and
                    stacking_ok and
                    cond_20_gap and cond_50_gap and cond_200_gap and
                    tightness_ok and slope_ok and
                    (d_sma20 <= d_sma50 * sma20_upper_bound) and
                    (d_sma20 >= d_sma50 * sma20_lower_bound) and
                    sma50_bounds_ok and close_200_ok and
                    (d_vol_sma20 >= sma20_min_volume)
                )
                return condition_daily, is_rounding
            except Exception:
                return False, False

        df_resample = df_ma.copy()
        if not isinstance(df_resample.index, pd.DatetimeIndex) and 'Date' in df_resample.columns:
            df_resample['Date'] = pd.to_datetime(df_resample['Date'])
            df_resample.set_index('Date', inplace=True)
        
        passes_daily, is_rounding = check_sma_conditions(df_resample)
        
        def check_weekly_monthly_sma(d_frame, rounding_flag=False):
            if len(d_frame) < 50: return False
            try:
                w_close = float(d_frame['Close'].iloc[-1])
                
                sma20_series = d_frame['Close'].rolling(window=20).mean()
                sma50_series = d_frame['Close'].rolling(window=50).mean()
                
                w_sma20 = float(sma20_series.iloc[-1])
                w_sma50 = float(sma50_series.iloc[-1])
                
                w_sma40 = float(d_frame['Close'].rolling(window=40).mean().iloc[-1]) if len(d_frame) >= 40 else float('nan')
                
                # 1. MA stacking: Close >= SMA20 >= SMA40 >= SMA50
                if pd.notna(w_sma40) and not rounding_flag:
                    stacking_ok = (w_close >= w_sma20) and (w_sma20 >= w_sma40) and (w_sma40 >= w_sma50)
                else:
                    stacking_ok = (w_close >= w_sma20) and (w_sma20 >= w_sma50)
                
                # Price-to-SMA gap constraints
                max_20_gap = 0.25 if rounding_flag else 0.15
                max_50_gap = 0.35 if rounding_flag else 0.20
                cond_20_gap = abs(w_close - w_sma20) / w_sma20 <= max_20_gap if pd.notna(w_sma20) else False
                cond_50_gap = abs(w_close - w_sma50) / w_sma50 <= max_50_gap if pd.notna(w_sma50) else False
                cond_200_gap = True  # Relaxed for strong uptrends
                
                # 3. Tightness over last 5 weekly bars (20% for weekly)
                high_5 = d_frame['High'].iloc[-5:].max()
                low_5 = d_frame['Low'].iloc[-5:].min()
                max_tightness = 0.30 if rounding_flag else 0.20
                tightness_ok = ((high_5 - low_5) / low_5) <= max_tightness if low_5 > 0 else False
                
                # 2. Upward slope over 5 bars
                w_sma20_5ago = float(sma20_series.iloc[-5]) if len(sma20_series) >= 5 else w_sma20
                w_sma50_5ago = float(sma50_series.iloc[-5]) if len(sma50_series) >= 5 else w_sma50
                slope_ok = (w_sma20 > w_sma20_5ago) and (w_sma50 > w_sma50_5ago)
                
                # 4. Volume contraction: recent 5-bar avg volume not spiking vs 20-bar avg
                vol_5 = d_frame['Volume'].iloc[-5:].mean()
                vol_20 = d_frame['Volume'].rolling(window=20).mean().iloc[-1]
                vol_contracting = vol_5 <= vol_20 * 1.2 if pd.notna(vol_20) and vol_20 > 0 else True
                
                sma50_bounds_ok = True
                close_200_ok = True
                if not rounding_flag:
                    sma50_bounds_ok = (w_sma50 <= w_sma40 * 1.20) and (w_sma50 >= w_sma40 * 0.90)
                    close_200_ok = (w_close >= w_sma40)
                
                condition_weekly = (
                    pd.notna(w_sma20) and pd.notna(w_sma50) and pd.notna(w_close) and pd.notna(w_sma40) and
                    stacking_ok and
                    cond_20_gap and cond_50_gap and cond_200_gap and
                    tightness_ok and slope_ok and vol_contracting and
                    (w_sma20 <= w_sma50 * sma20_upper_bound) and
                    (w_sma20 >= w_sma50 * sma20_lower_bound) and
                    sma50_bounds_ok and close_200_ok
                )
                return condition_weekly
            except Exception: return False
            
        df_weekly = df_resample.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        passes_weekly = check_weekly_monthly_sma(df_weekly, is_rounding)
        df_monthly = df_resample.resample('ME' if hasattr(pd.tseries.offsets, 'MonthEnd') else 'M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        passes_monthly = check_weekly_monthly_sma(df_monthly, is_rounding)

        # VPA requirement removed per user configuration
        if (passes_daily or passes_weekly or passes_monthly) and not is_overextended:
            above_buy_price = round(sma20, 2)  # Support = 20 SMA
            above_exit_price = round(sma50 * 0.97, 2) 
            above_target_price = round(today_close_val * 1.12, 2) 
            
            ema9 = float(df_resample['Close'].ewm(span=9, adjust=False).mean().iloc[-1])
            ema21 = float(df_resample['Close'].ewm(span=21, adjust=False).mean().iloc[-1])
            
            base_above_rec = (f"Passes strict 20 & 50 SMA constraints. "
                              f"Short and Mid-term VPA trends are Green (Uptrend). "
                              f"If price goes down below the 9 EMA support (₹{ema9:.2f}), and 21 EMA support (₹{ema21:.2f}) is 2nd Support, "
                              f"if not overcome then exit from the position. Target momentum ₹{above_target_price:.2f}.")
            
            if is_rounding and pd.notna(sma200) and sma200 > 0 and c_val < sma200 * 1.05:
                base_above_rec += " 🌟 Note: Added despite being near or below 200 SMA because a clean Rounding Bottom/Cup pattern was detected!"
            
            # 5. Breakout proximity: price within 3% of 20-day high
            high_20d = df_resample['High'].iloc[-20:].max()
            near_breakout = ((high_20d - today_close_val) / today_close_val) <= 0.03 if today_close_val > 0 else False
            
            res["above_ma"] = {
                "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                "dist_20sma_pct": round(dist_20, 2),
                "dist_50sma_pct": round(dist_50, 2),
                "dist_200sma_pct": round(dist_200, 2),
                "setup_type": "above_ma", "buy_price": above_buy_price, "exit_price": above_exit_price,
                "target_price": above_target_price,
                "confidence": "🔥 High (Near Breakout + Multi-TF)" if (passes_daily and passes_weekly and near_breakout) else "High (Multi-TF Uptrend Convergence)" if (passes_daily and passes_weekly and passes_monthly) else "Medium (Uptrend)",
                "recommendation": compute_rich_analysis(df_ma, sym, "20&50 SMA Multi-TF", base_above_rec, indicators=ind),
                "passes_daily": passes_daily,
                "passes_weekly": passes_weekly,
                "passes_monthly": passes_monthly,
                "near_breakout": near_breakout
            }

        yesterday_l = float(yesterday_row['Low']); yesterday_sma65 = float(yesterday_row['SMA65'])
        tested_today = l_val <= sma65 * 1.01; tested_yesterday = yesterday_l <= yesterday_sma65 * 1.01
        o_val = float(today_row['Open']); yesterday_c = float(yesterday_row['Close'])
        is_green_candle = c_val > o_val; is_up_move = c_val > yesterday_c; holds_above = c_val > sma65
        
        # Gap filter: skip stocks where price is too far from 65 SMA (overextended)
        dist_65 = (c_val - sma65) / sma65 * 100
        if (tested_today or tested_yesterday) and holds_above and is_green_candle and is_up_move and dist_65 <= 8:
            support_buy_price = round(sma65, 2)  # Support = 65 SMA (the actual support level)
            support_exit_price = round(sma65 * 0.97, 2) 
            support_target_price = round(today_close_val * 1.15, 2) 
            support_confidence = "High (Pullback Support)" if today_close_val > yesterday_row['Close'] else "Medium (Pullback Support)"
            base_support_rec = (f"Institutional pullback testing critical 65 SMA support (₹{sma65:.2f}). "
                                f"Buy near support ₹{support_buy_price:.2f} (65 SMA) with tight stop just below SMA at ₹{support_exit_price:.2f} targeting bounce to ₹{support_target_price:.2f}.")
            res["support_ma"] = {
                "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                "dist_65sma_pct": round(dist_65, 2), "setup_type": "support_ma",
                "buy_price": support_buy_price, "exit_price": support_exit_price, "target_price": support_target_price,
                "confidence": support_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "65 SMA Support", base_support_rec, indicators=ind)
            }
            
        crossed_golden = (yesterday_row['SMA50'] <= yesterday_row['SMA200']) and (today_row['SMA50'] > today_row['SMA200'])
        crossed_150 = (yesterday_row['SMA50'] <= yesterday_row['SMA150']) and (today_row['SMA50'] > today_row['SMA150'])
        price_crossed_50 = (yesterday_row['Close'] <= yesterday_row['SMA50']) and (today_row['Close'] > today_row['SMA50'])
        price_crossed_150 = (yesterday_row['Close'] <= yesterday_row['SMA150']) and (today_row['Close'] > today_row['SMA150'])
        price_crossed_200 = (yesterday_row['Close'] <= yesterday_row['SMA200']) and (today_row['Close'] > today_row['SMA200'])
        
        # Gap filter: skip stocks where price is too far from crossover MAs (overextended)
        cross_dist_50 = (c_val - sma50) / sma50 * 100
        cross_dist_200 = (c_val - sma200) / sma200 * 100
        if (crossed_golden or crossed_150 or price_crossed_50 or price_crossed_150 or price_crossed_200) and cross_dist_50 <= 15 and cross_dist_200 <= 20:
            cross_support = max(s for s in [sma50, sma150, sma200] if s < c_val) if any(s < c_val for s in [sma50, sma150, sma200]) else c_val * 0.94
            cross_buy_price = round(cross_support * 1.01, 2)  # Support = nearest MA below price
            cross_exit_price = round(cross_support * 0.96, 2) 
            cross_target_price = round(today_close_val * 1.18, 2) 
            cross_confidence = "High (Golden Cross)" if crossed_golden else "Medium-High (Crossover)"
            base_cross_rec = (f"Technical moving average crossover signal! Buy near support ₹{cross_buy_price:.2f} "
                              f"to ride the emerging uptrend. Set stop loss at ₹{cross_exit_price:.2f} targeting swing high ₹{cross_target_price:.2f}.")
            res["crossover_ma"] = {
                "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                "dist_50sma_pct": round(cross_dist_50, 2),
                "dist_200sma_pct": round(cross_dist_200, 2), "setup_type": "crossover_ma",
                "buy_price": cross_buy_price, "exit_price": cross_exit_price, "target_price": cross_target_price,
                "confidence": cross_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "MA Crossover", base_cross_rec, indicators=ind)
            }

    if scan_mode in ("all", "full"):
        if len(df_ma) >= 250:
            today_row = df_ma.iloc[-1]; yesterday_row = df_ma.iloc[-2]; c_val = float(today_row['Close'])
            sma50 = float(today_row['SMA50']); sma150 = float(today_row['SMA150']); sma200 = float(today_row['SMA200'])
            # EXPERT FIX: Use 20-bar lookback for 200 SMA trend (Minervini uses ~1 month, not 10 days)
            # 10 bars = 2 weeks = too short, causes whipsaw on minor corrections
            sma200_20d_ago = float(df_ma['SMA200'].iloc[-21]) if len(df_ma) >= 220 else sma200
            high_52w = float(df_ma['High'].iloc[-250:].max()); low_52w = float(df_ma['Low'].iloc[-250:].min())
            
            if c_val > sma150 and c_val > sma200 and sma150 > sma200 and sma200 > sma200_20d_ago and sma50 > sma150 and sma50 > sma200 and c_val > sma50 and c_val >= 1.25 * low_52w and c_val >= 0.75 * high_52w:
                run_up_200 = round(((c_val - sma200) / sma200 * 100), 2)
                run_up_52w = round(((c_val - low_52w) / low_52w * 100), 2)
                is_early = bool(c_val <= 1.20 * sma200)
                exit_price = round(min(sma200 * 0.98, c_val * 0.94), 2)
                distance_200 = (c_val - sma200) / sma200
                target_mult = 1.40 - min(0.15, distance_200 * 0.7) if is_early else 1.18 - min(0.06, (distance_200 - 0.20) * 0.4)
                target_price = round(max(high_52w * 1.05, c_val * target_mult), 2)
                min_confidence = "High (Minervini Stage-2)" if is_early else "Medium-High (Minervini Extended)"
                rem_pct = ((target_price - c_val) / c_val * 100)
                stage_label = "Early Stage-2 Accumulation" if is_early else "Extended Stage-2 Uptrend"
                base_minervini_rec = (f"Mark Minervini Stage-2 Trend Template verified! The stock is in an active '{stage_label}' "
                                      f"having run up {run_up_52w:.1f}% from its 52w low and holding {run_up_200:.1f}% above its 200 SMA support. "
                                      f"Buy around CMP ₹{c_val:.2f}. Set stop loss at ₹{exit_price:.2f} (tight support lock) "
                                      f"targeting momentum swing target of ₹{target_price:.2f} (remaining potential +{rem_pct:.1f}%).")
                min_support = max(s for s in [sma50, sma150, sma200] if s < c_val) if any(s < c_val for s in [sma50, sma150, sma200]) else sma200
                res["minervini"] = {
                    "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                    "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                    "setup_type": "minervini", "run_up_200": run_up_200, "run_up_52w": run_up_52w, "is_early": is_early,
                    "buy_price": round(min_support * 1.01, 2), "exit_price": exit_price, "target_price": target_price,
                    "confidence": min_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "Minervini Stage-2", base_minervini_rec, indicators=ind)
                }
                
        scan_res = scan_stock(symbol=sym, df=df, min_dry_days=min_dry, max_dry_days=max_dry, min_volume_ratio=min_vol_ratio, min_price_change=min_price_chg, min_dry_spikes=min_dry_spikes, indicators=ind)
        if scan_res and scan_res.get('signal_strength', 0) >= min_signal_str:
            if (not above_50dma_only or scan_res.get('above_50dma', False)) and (not above_200dma_only or scan_res.get('above_200dma', False)):
                res["flagged"] = scan_res
                
    if scan_mode in ("all", "full"):
        df_wt = df
        if df_wt is not None and len(df_wt) >= 40:
            wt_res = scan_wt_cross(sym, df_wt, indicators=ind)
            if wt_res is not None:
                wt_res['timeframe'] = "Daily"
                res["wt"] = wt_res
                
        if df is not None:
            res["vcs"] = scan_vcs(sym, df, indicators=ind)
            res["structural_vcp"] = MinerviniVCPAnalyzer(sym, df=df, benchmark_df=benchmark_df).run()
            res["vpa"] = scan_vpa_trend(sym, df, indicators=ind)
            
            # --- New Scanners ---
            try:
                # Dan Zanger
                from zanger_scanner import scan_zanger, ZangerConfig, get_latest_signal
                cfg_zanger = ZangerConfig()
                z_df = scan_zanger(df, cfg_zanger)
                if not z_df.empty:
                    last_zanger = get_latest_signal(z_df)
                    if last_zanger.get("zanger_signal", False):
                        from data_fetcher import get_stock_sector
                        last_zanger["symbol"] = sym
                        last_zanger["sector"] = get_stock_sector(sym)
                        res["zanger"] = last_zanger
                        
                # Volume Profile
                res["volume_profile"] = scan_volume_profile(sym, df, market_cap=0.0)
                
                # Support RSI
                res["support_rsi"] = scan_support_rsi(sym, df, market_cap=0.0, rsi_threshold=35.0)
                
                # Weekly & Monthly Resampled Data
                m_df = resample_ohlcv(df, '1ME')
                w_df = resample_ohlcv(df, '1W-MON')
                
                if m_df is not None and not m_df.empty and w_df is not None and not w_df.empty:
                    # EMA Support (repurposed from ema_support)
                    from scanner import scan_ema_support
                    res["ema_support"] = scan_ema_support(sym, df)
                    
                    # Stage Analysis
                    bRet = 0.0
                    if benchmark_df is not None and len(benchmark_df) > 250:
                        bC = float(benchmark_df['Close'].iloc[-1])
                        bCold = float(benchmark_df['Close'].iloc[-250])
                        if bCold > 0:
                            bRet = (bC - bCold) / bCold
                    res["stage_analysis"] = scan_stage_analysis(sym, df, bench_ret=bRet)
                    
                    # Monthly Early Stage 2
                    res["stage2"] = scan_monthly_early_stage2(sym, m_df, max_run_up_pct=20.0, market_cap_cr=0.0)
                    
                    # Monthly Momentum
                    res["monthly_momentum"] = scan_monthly_momentum(sym, m_df, market_cap_cr=0.0)
                    
                    # Weekly Momentum
                    res["weekly_momentum"] = scan_weekly_momentum(sym, w_df, market_cap_cr=0.0)
            except Exception as e:
                # print(f"Error running extended background scans for {sym}: {e}")
                pass
                
    return res
