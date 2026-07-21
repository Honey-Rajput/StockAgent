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
    st.markdown("### 📈 Stocks Trading Above 20 SMA & 50 SMA")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Identify stocks in a strong medium-term uptrend where price is trading comfortably above both their 20-day and 50-day Simple Moving Averages.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        sma_timeframe_tab = st.selectbox("SMA Strategy Timeframe", ["Daily", "Weekly", "All (Daily + Weekly Convergence)"], index=0, key="sma_timeframe_tab")
    
    above_ma_data = st.session_state.above_ma_results
    
    if above_ma_data is None:
        st.info("💡 Run the scanner from the sidebar to identify stocks trading above their 20 SMA and 50 SMA.")
    else:
        # Dynamically filter based on the UI dropdown
        filtered_above = []
        for r in above_ma_data:
            if matches_sma_timeframe_filter(r, sma_timeframe_tab):
                filtered_above.append(r)
                
        if len(filtered_above) == 0:
            st.info(f"ℹ️ No stocks found today matching the '{sma_timeframe_tab}' 20 & 50 SMA uptrend criteria.")
        else:
            # Sort by day change descending
            sorted_above = sorted(filtered_above, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
            
            # Download results option
            export_above = []
            for r in sorted_above:
                export_above.append({
                    "Symbol": r['symbol'],
                    "Sector": get_stock_sector(r['symbol']),
                    "CMP (₹)": r['cmp'],
                    "Day Change %": r['day_change_pct'],
                    "Setup Type": r['setup_type'],
                    "Dist to 20 SMA (%)": float(r.get('dist_20sma_pct') or 0.0),
                    "Dist to 50 SMA (%)": float(r.get('dist_50sma_pct') or 0.0),
                    "Dist to 200 SMA (%)": float(r.get('dist_200sma_pct') or 0.0),
                    "Suggested Buy (₹)": r['buy_price'],
                    "Suggested Exit/SL (₹)": r['exit_price'],
                    "Suggested Target (₹)": r['target_price'],
                    "Confidence": r['confidence'],
                    "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
                })
            export_a_df = pd.DataFrame(export_above)
            csv_a_data = export_a_df.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Download Above 20/50 SMA Results (CSV)",
                data=csv_a_data,
                file_name=f"above_20_50_sma_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_above_ma_btn"
            )
            
            st.markdown("---")
            # Render the unified Trade Execution Matrix
            st.markdown("### 📈 Active Uptrend Trade Execution Sheet")
            render_unified_strategy_table(sorted_above, "above_ma", "above_ma_tab")
