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
    st.markdown("### 📉 Near 30 SMA")
    st.info("Finds stocks where the price is within +/- 3% of the 30-day SMA (can be slightly above or below it).")
    
    if not st.session_state.get('near_30sma_results') and ALL_TAB_SCAN_STATUS.get("near_30sma_results") is not None:
        st.session_state.near_30sma_results = ALL_TAB_SCAN_STATUS["near_30sma_results"]
        
    if "near_30sma_weekly_results" not in st.session_state:
        st.session_state.near_30sma_weekly_results = database.get_cached_near_30sma_weekly(latest_date_str) if is_startup and latest_date_str else []
    if "near_30sma_monthly_results" not in st.session_state:
        st.session_state.near_30sma_monthly_results = database.get_cached_near_30sma_monthly(latest_date_str) if is_startup and latest_date_str else []

    # Auto load from DB if missing in session
    if 'near_30sma_results' not in st.session_state or not st.session_state.near_30sma_results:
        try:
            today_str = get_market_date()
            import database
            st.session_state.near_30sma_results = database.get_cached_near_30sma(today_str) or []
            st.session_state.near_30sma_weekly_results = database.get_cached_near_30sma_weekly(today_str) or []
            st.session_state.near_30sma_monthly_results = database.get_cached_near_30sma_monthly(today_str) or []
        except Exception as e:
            pass
            
    tf_near30 = st.selectbox("Select Timeframe", ["Daily", "Weekly", "Monthly"], key="tf_near30")
    if tf_near30 == "Daily":
        results = st.session_state.get('near_30sma_results', [])
    elif tf_near30 == "Weekly":
        results = st.session_state.get('near_30sma_weekly_results', [])
    else:
        results = st.session_state.get('near_30sma_monthly_results', [])

    if results:
        df_res = pd.DataFrame(results)
        df_res['symbol'] = df_res['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
        st.write(f"### Found {len(results)} stocks")
        st.dataframe(
            df_res,
            use_container_width=True,
            column_config={
                "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)", width="small"),
                "company_name": "Company",
                "cmp": st.column_config.NumberColumn("CMP (₹)", width="small"),
                "day_change_pct": st.column_config.NumberColumn("Chg %", width="small"),
                "sma30": st.column_config.NumberColumn("30 SMA (₹)", width="small"),
                "dist_pct": st.column_config.NumberColumn("Dist to SMA %", format="%.2f %%", width="small"),
                "volume": st.column_config.NumberColumn("Volume", width="small")
            },
            hide_index=True
        )
    else:
        st.info("No stocks found matching the Near 30 SMA criteria. Run the scanner.")
