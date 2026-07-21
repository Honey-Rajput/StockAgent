import threading
import concurrent.futures
import re
import json
import time

import streamlit as st
import pandas as pd

from datetime import datetime, timedelta
import os
import yfinance as yf
import requests as _requests_session_lib
_YF_SESSION = _requests_session_lib.Session()
_YF_SESSION.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from config import IST_TIMEZONE, get_company_name, DRY_ZONE_MIN_DAYS, DRY_ZONE_MAX_DAYS, MIN_VOLUME_RATIO, MIN_PRICE_CHANGE
from data_fetcher import fetch_ohlcv, get_index_stocks, fetch_ohlcv_timeframe, get_stock_sector, get_market_condition, fetch_nifty50_returns, calculate_rs_rating
from scanner import scan_stock, scan_wt_cross, compute_rich_analysis, scan_monthly_momentum, scan_weekly_momentum, scan_vcs, scan_monthly_early_stage2, scan_vpa_trend, scan_structural_vcp
from indicators import precompute_indicators

def get_market_date(for_display=False):
    today = datetime.now(IST_TIMEZONE)
    if today.isoweekday() == 7:
        target_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
    elif today.isoweekday() == 6:
        target_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        target_date = today.strftime('%Y-%m-%d')
        
    if for_display:
        import database
        if not database.has_scanned_today(target_date):
            avail = database.get_available_scan_dates()
            if avail:
                return avail[0]
                
    return target_date


import watchlist
import tabs.tab_watchlist
from utils import inject_premium_css, get_signal_badge_html, get_day_change_badge_html

from ui_components import (
    extract_clean_recommendation,
    matches_sma_timeframe_filter,
    render_quick_trade_board,
    render_trading_setup_card,
    render_unified_strategy_table,
)
import database
import ai_detector


def render():
    scan_data = st.session_state.get('scan_results', [])
    is_admin = st.session_state.get('is_admin', False)
    # 0. Timeframe & Threshold selector inside tab
    wt_col1, wt_col2 = st.columns(2)
    with wt_col1:
        wt_timeframe = st.selectbox(
            "🌊 Select WaveTrend Timeframe:",
            options=["Daily", "15 Min", "1 Hour", "Weekly", "Monthly"],
            index=0,
            key="wt_tab_timeframe_selector_v2",
            help="Select the WaveTrend chart interval. Changing this dynamically runs a real-time parallel scan for active stocks."
        )
    with wt_col2:
        default_wt_thresh = -20.0 if wt_timeframe in ["Weekly", "Monthly"] else -40.0
        wt_oversold_threshold = st.number_input(
            "📉 Oversold Threshold:",
            min_value=-100.0,
            max_value=0.0,
            value=default_wt_thresh,
            step=5.0,
            key=f"wt_oversold_threshold_{wt_timeframe}",
            help="Define the WT1 value below which a stock is considered oversold. Default is -40.0."
        )
        
    wt_cache_key = f"{wt_timeframe}_{wt_oversold_threshold}"
    
    # Reactive Loader
    if 'wt_results_by_tf' not in st.session_state:
        st.session_state.wt_results_by_tf = {}
        
    run_wt_btn = st.button("🌊 Run Advanced WaveTrend Scan", key="run_wt_scan_btn", width="stretch", disabled=not is_admin)
    
    if run_wt_btn:
        # Universe is hardcoded to Top 1000 NSE stocks
        universe_key = "TOP 1000"
        from data_fetcher import get_top1000_nse_symbols
        raw_symbols = get_top1000_nse_symbols()
            
        symbols_to_scan = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
        with st.spinner(f"Running Advanced WaveTrend {wt_timeframe} scan on Top 1000 ({len(symbols_to_scan)} stocks)..."):
            from scanner import scan_wt_cross
            
            # Map timeframes
            interval_map = {"Daily": "1d", "15 Min": "15m", "1 Hour": "60m", "Weekly": "1wk", "Monthly": "1mo"}
            period_map = {"Daily": "300d", "15 Min": "60d", "1 Hour": "730d", "Weekly": "5y", "Monthly": "10y"}
            
            interval = interval_map[wt_timeframe]
            period = period_map[wt_timeframe]
            
            wt_tf_results = []
            chunk_size = 50
            sym_chunks = [symbols_to_scan[i:i + chunk_size] for i in range(0, len(symbols_to_scan), chunk_size)]
            
            for chunk in sym_chunks:
                chunk_ns = [s if s.endswith('.NS') else f"{s}.NS" for s in chunk]
                try:
                    # yfinance 1.x: group_by, threads, auto_adjust=False removed
                    df_bulk = yf.download(tickers=chunk_ns, period=period, interval=interval, progress=False, threads=False, timeout=15)
                    for sym in chunk:
                        sym_ns = sym if sym.endswith('.NS') else f"{sym}.NS"
                        try:
                            if isinstance(df_bulk.columns, pd.MultiIndex):
                                # yfinance 1.x: (price_type, ticker) MultiIndex
                                all_tickers_wt = df_bulk.columns.get_level_values(1).unique().tolist()
                                matched_wt = next((t for t in all_tickers_wt if t.upper() == sym_ns.upper()), None)
                                if matched_wt is None:
                                    continue
                                ticker_df = df_bulk.xs(matched_wt, axis=1, level=1).copy()
                            else:
                                if len(chunk_ns) == 1:
                                    ticker_df = df_bulk.copy()
                                else:
                                    continue

                            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                            if all(col in ticker_df.columns for col in required_cols):
                                ticker_df = ticker_df[required_cols].dropna(subset=['Close'])
                                if interval not in ["15m", "60m"]:
                                    ticker_df = ticker_df[ticker_df['Volume'] > 0]
                                if len(ticker_df) >= 40:
                                    ticker_df = ticker_df.reset_index()
                                    ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)

                                    wt_res = scan_wt_cross(sym, ticker_df, wt_oversold_threshold=wt_oversold_threshold)
                                    if wt_res is not None:
                                        # Market cap filter: exclude stocks below ₹2000 Cr
                                        try:
                                            mcap = getattr(yf.Ticker(sym_ns).fast_info, 'market_cap', None) or 0
                                            mcap_cr = mcap / 1e7  # Convert to Crore
                                            if mcap_cr < 2000.0:
                                                continue
                                            wt_res['market_cap_cr'] = round(mcap_cr, 1)
                                        except Exception:
                                            pass  # Allow through if market cap fetch fails
                                        wt_res['timeframe'] = wt_timeframe
                                        # Inject threshold logic
                                        wt_res['is_oversold'] = wt_res['wt_value'] <= wt_oversold_threshold
                                        wt_tf_results.append(wt_res)
                        except Exception as sym_wt_ex:
                            print(f"Error extracting {sym_ns} from WaveTrend bulk download: {sym_wt_ex}")
                except Exception as chunk_ex:
                    print(f"Error bulk downloading WaveTrend chunk: {chunk_ex}")
            
            st.session_state.wt_results_by_tf[wt_cache_key] = wt_tf_results
            st.toast(f"🌊 WaveTrend {wt_timeframe} scan complete!", icon="✅")
            
    # Pick up background scan results if available
    if not st.session_state.wt_results_by_tf.get(wt_cache_key) and ALL_TAB_SCAN_STATUS["wt_results"] is not None:
        st.session_state.wt_results_by_tf["Daily_-40.0"] = ALL_TAB_SCAN_STATUS["wt_results"]
        st.session_state.wt_results = ALL_TAB_SCAN_STATUS["wt_results"]

    wt_data = st.session_state.wt_results_by_tf.get(wt_cache_key, None)
    
    st.markdown(f"### 🌊 WaveTrend Oversold Buy Signals ({wt_timeframe} Timeframe)")
    st.markdown(f"<p style='font-size:0.9rem; color:#94a3b8;'>Scan for stocks in the WaveTrend oversold zone (WT1 below {wt_oversold_threshold}) using LazyBear's WaveTrend with Crosses indicator. <span style=\"color:#ffa000; font-weight:600;\">Filters: Price ≥ ₹100 | Market Cap ≥ ₹2000 Cr</span>. Stocks showing a <b style=\"color:#00e676;\">green dot 🟢 buy signal</b> (WT1 crossing above WT2) in oversold territory are prime mean-reversion candidates.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. Premium Metrics Row
    wt_m1, wt_m2, wt_m3, wt_m4 = st.columns(4)
    
    if wt_data:
        wt_total = len(wt_data)
        wt_buy_signals = [r for r in wt_data if r.get('buy_signal', False)]
        wt_buy_count = len(wt_buy_signals)
        wt_deepest = min(r['wt_value'] for r in wt_data)
        wt_avg = sum(r['wt_value'] for r in wt_data) / wt_total
    else:
        wt_total = 0
        wt_buy_count = 0
        wt_deepest = 0.0
        wt_avg = 0.0
    
    wt_m1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Oversold Stocks (WT1 < {wt_oversold_threshold})</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{wt_total}</h3></div>', unsafe_allow_html=True)
    wt_m2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🟢 Buy Signals (Green Dot)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{wt_buy_count}</h3></div>', unsafe_allow_html=True)
    wt_m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Deepest WT1 Value</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{wt_deepest:.1f}</h3></div>', unsafe_allow_html=True)
    wt_m4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg WT1 Value</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{wt_avg:.1f}</h3></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filter toggle
    wt_filter_col1, wt_filter_col2, wt_filter_col3 = st.columns(3)
    wt_show_buy_only = wt_filter_col1.checkbox(
        "🟢 Show Buy Signals Only (Green Dot)",
        value=False,
        help="Show only stocks where WT1 has crossed above WT2 in the oversold zone (bullish crossover buy signal)"
    )
    wt_above_20sma = wt_filter_col2.checkbox(
        "📈 Above 20 SMA Only",
        value=False,
        help="Show only stocks currently trading above their 20 SMA trend filter"
    )
    wt_above_50sma = wt_filter_col3.checkbox(
        "🛡️ Above 50 SMA Only",
        value=False,
        help="Show only stocks currently trading above their 50 SMA trend filter"
    )
    wt_above_200sma = wt_filter_col3.checkbox(
        "🛡️ Above 200 DMA Only",
        value=False,
        help="Show only stocks currently trading above their 200 SMA long-term trend filter"
    )
    
    # Background scan progress indicator
    if wt_data is None and ALL_TAB_SCAN_STATUS["is_running"]:
        _bg_scanner = ALL_TAB_SCAN_STATUS["current_scanner"]
        _bg_status = ALL_TAB_SCAN_STATUS["status_text"]
        _bg_progress = ALL_TAB_SCAN_STATUS["progress"]
        st.markdown(f"""
        <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
            <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
            </h4>
            <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. WaveTrend results will appear here when ready!</p>
            <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
        </div>
        <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
        """, unsafe_allow_html=True)
        st.progress(_bg_progress)
        if st.button("🔄 Refresh Scanner Status", key="refresh_bg_wt_status_btn"):
            st.rerun()

    # 2. Main Scan Table
    if wt_data is None:
        st.info("💡 Run the scanner from the sidebar to identify WaveTrend oversold buy signals.")
    elif len(wt_data) == 0:
        st.info(f"ℹ️ No stocks found in the WaveTrend oversold zone (WT1 < {wt_oversold_threshold}) on {wt_timeframe} timeframe today.")
    else:
        # Apply filters
        display_wt = list(wt_data)
        if wt_show_buy_only:
            display_wt = [r for r in display_wt if r.get('buy_signal', False)]
        if wt_above_20sma:
            display_wt = [r for r in display_wt if r.get('above_20sma', False)]
        if wt_above_50sma:
            display_wt = [r for r in display_wt if r.get('above_50sma', False)]
        if wt_above_200sma:
            display_wt = [r for r in display_wt if r.get('above_200sma', False)]
        
        # Sort by WT value ascending (deepest oversold first)
        sorted_wt = sorted(display_wt, key=lambda x: x['wt_value'])
        
        if len(sorted_wt) == 0:
            st.info("ℹ️ No stocks match the active filters found today. Try unchecking some filters above to see more oversold stocks.")
        else:
            # Download WaveTrend results
            export_wt = []
            for r in sorted_wt:
                export_wt.append({
                    "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                        "CMP (₹)": r['cmp'],
                    "Day Change %": r['day_change_pct'],
                    "WT1": r['wt_value'],
                    "WT2": r['wt2_value'],
                    "WT Diff (WT1-WT2)": r.get('wt_diff', r['wt_value'] - r['wt2_value']),
                    "Buy Signal": r.get('buy_signal', False),
                    "Above 20 SMA": r.get('above_20sma', False),
                    "Above 50 SMA": r.get('above_50sma', False),
                   "Above 200 SMA": r.get('above_200sma', False),
                    "Volume": int(r.get('volume', 0)),
                    "Buy Range (₹)": r.get('buy_price', r.get('cmp', 0)),
                    "Stop Loss (₹)": r.get('exit_price', 0),
                    "Target (₹)": r.get('target_price', 0),
                    "Confidence": r.get('confidence', ''),
                    "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
                })
            export_wt_df = pd.DataFrame(export_wt)
            csv_wt_data = export_wt_df.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Download WaveTrend Results (CSV)",
                data=csv_wt_data,
                file_name=f"wavetrend_signals_{wt_timeframe}_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="wt_download_csv_btn"
            )
            
            st.markdown("---")
            # Render the unified Trade Execution Matrix
            st.markdown(f"### 🌊 Active Oversold Trade Execution Sheet ({wt_timeframe})")
            render_unified_strategy_table(sorted_wt, "wavetrend", "wt_tab")
    
    # WaveTrend indicator explanation
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 How WaveTrend Indicator Works"):
        st.markdown("""
        **WaveTrend with Crosses (LazyBear)**
        
        This indicator is a momentum oscillator that identifies overbought/oversold conditions:
        
        - **WT1 (Green Line):** The smoothed trend oscillator calculated from typical price deviations
        - **WT2 (Red Line):** A 4-period SMA of WT1, used as a signal line
        - **Buy Signal (Green Dot 🟢):** Occurs when WT1 crosses ABOVE WT2 in the oversold zone (below -40)
        - **Oversold Zone:** WT1 values below -40 indicate the stock is in deep oversold territory
        - **Extreme Oversold:** WT1 values below -60 suggest extreme selling pressure
        
        **Strategy:** Look for stocks with WT1 below -40 showing a green dot buy signal (WT1 crossing above WT2). 
        These are high-probability mean-reversion setups where selling pressure is exhausting and a bounce is likely.
        
        **Parameters Used:** Channel Length = 10, Average Length = 21
        """, unsafe_allow_html=False)
