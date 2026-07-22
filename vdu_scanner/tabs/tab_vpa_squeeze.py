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
    st.markdown("### 📉 VPA Green + MA Squeeze")
    st.info("Finds stocks where VPA is Green (Minor, Mid, Major) and the 10/21/50 SMA are tightly clustered (<6% gap).")
    
    if "vpa_squeeze_results" not in st.session_state:
        st.session_state.vpa_squeeze_results = []
    if "vpa_squeeze_weekly_results" not in st.session_state:
        st.session_state.vpa_squeeze_weekly_results = []
    if "vpa_squeeze_monthly_results" not in st.session_state:
        st.session_state.vpa_squeeze_monthly_results = []
        
    if not st.session_state.vpa_squeeze_results:
        try:
            today_str = get_market_date()
            import database
            st.session_state.vpa_squeeze_results = database.get_cached_vpa_squeeze(today_str)
            st.session_state.vpa_squeeze_weekly_results = database.get_cached_vpa_squeeze_weekly(today_str)
            st.session_state.vpa_squeeze_monthly_results = database.get_cached_vpa_squeeze_monthly(today_str)
        except Exception as e:
            pass
        
    run_vpa_squeeze_btn = st.button("🚀 Run VPA Squeeze Scan", width="stretch", disabled=not is_admin)
    if run_vpa_squeeze_btn:
        st.session_state.vpa_squeeze_results = []
        with st.spinner("Running VPA Squeeze Scan..."):
            try:
                from data_fetcher import get_all_nse_symbols
                from scanner import scan_vpa_ma_squeeze
                from local_cache_manager import bulk_get_cached_ohlcv
                
                raw_symbols = get_all_nse_symbols()
                symbols_to_scan = [s.strip().upper().replace('.NS', '') for s in raw_symbols if str(s).strip()]
                
                st.info(f"Step 1/2 — Fetching history from Master Cache...")
                prog = st.progress(0)
                status = st.empty()
                
                # Fetch all cached data at once
                bulk_cached = bulk_get_cached_ohlcv(symbols_to_scan, "1d")
                
                st.info("Step 2/2 — Calculating MA Squeeze (Instant)...")
                status.empty()
                prog.progress(1.0)
                
                results = []
                for sym, df in bulk_cached.items():
                    res = scan_vpa_ma_squeeze(sym, df)
                    if res is not None:
                        results.append(res)
                            
                prog.empty()
                status.empty()
                st.session_state.vpa_squeeze_results = results
                
                # Save to database
                today_str = get_market_date()
                database.save_vpa_squeeze_only(today_str, results)
                
                st.success(f"Scan complete! Found {len(results)} matches.")
                
            except Exception as e:
                st.error(f"Error running scan: {e}")

    tf_vpa_sq = st.selectbox("Select Timeframe", ["Daily", "Weekly", "Monthly"], key="tf_vpa_sq")
    if tf_vpa_sq == "Daily":
        display_results = st.session_state.get('vpa_squeeze_results', [])
    elif tf_vpa_sq == "Weekly":
        display_results = st.session_state.get('vpa_squeeze_weekly_results', [])
    else:
        display_results = st.session_state.get('vpa_squeeze_monthly_results', [])

    if display_results:
        df_res = pd.DataFrame(display_results)
        df_res['symbol'] = df_res['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
        st.write(f"### Found {len(display_results)} stocks")
        # Sort by Compression Score (Gap %) ascending so tightest squeezes appear first
        if 'ma_gap_pct' in df_res.columns:
            df_res = df_res.sort_values('ma_gap_pct', ascending=True)
        display_cols = ['symbol', 'cmp', 'day_change_pct', 'sma10', 'sma21', 'sma50', 'ma_gap_pct', 'dist_to_200_pct']
        # Keep only columns that actually exist (handles older cached results without the new field)
        display_cols = [c for c in display_cols if c in df_res.columns]
        st.dataframe(
            df_res[display_cols],
            use_container_width=True,
            column_config={
                "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)", width="small"),
                "cmp": st.column_config.NumberColumn("CMP (₹)", width="small"),
                "day_change_pct": st.column_config.NumberColumn("Chg %", width="small"),
                "sma10": st.column_config.NumberColumn("10 SMA", width="small"),
                "sma21": st.column_config.NumberColumn("21 SMA", width="small"),
                "sma50": st.column_config.NumberColumn("50 SMA", width="small"),
                "ma_gap_pct": st.column_config.NumberColumn("Gap %", help="(Max SMA − Min SMA) / Min SMA × 100. Lower = tighter.", width="small"),
                "compression_score": st.column_config.NumberColumn(
                    "Compression Score",
                    help="Max(10,21,50 SMA) − Min(10,21,50 SMA) / Min(10,21,50 SMA). Lower = tighter squeeze = stronger breakout candidate.",
                    format="%.2f %%",
                    width="small"
                ),
                "dist_to_200_pct": st.column_config.NumberColumn("200 Dist %", width="small")
            },
            hide_index=True
        )
    else:
        st.info("No stocks found matching the VPA Squeeze criteria. Run the scanner.")
