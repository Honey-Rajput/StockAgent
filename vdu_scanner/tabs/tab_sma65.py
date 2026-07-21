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
    st.markdown("### 🛡️ Stocks Taking Support at 65 SMA")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Scan for institutional pullbacks where the price is testing or bouncing precisely off the 65-day Simple Moving Average (65 SMA), offering high-probability low-risk entries.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    support_ma_data = st.session_state.support_ma_results
    
    if support_ma_data is None:
        st.info("💡 Run the scanner from the sidebar to identify stocks taking support at their 65 SMA.")
    elif len(support_ma_data) == 0:
        st.info("ℹ️ No stocks found today taking support at their 65 SMA.")
    else:
        # Sort by day change descending
        sorted_support = sorted(support_ma_data, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
        
        # Download results option
        export_support = []
        for r in sorted_support:
            export_support.append({
                "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                "CMP (₹)": r['cmp'],
                "Day Change %": r['day_change_pct'],
                "Setup Type": r['setup_type'],
                "Dist to 65 SMA (%)": r.get('dist_65sma_pct', 0.0),
                "Suggested Buy (₹)": r['buy_price'],
                "Suggested Exit/SL (₹)": r['exit_price'],
                "Suggested Target (₹)": r['target_price'],
                "Confidence": r['confidence'],
                "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
            })
        export_s_df = pd.DataFrame(export_support)
        csv_s_data = export_s_df.to_csv(index=False).encode('utf-8-sig')
        
        st.download_button(
            label="📥 Download 65 SMA Support Results (CSV)",
            data=csv_s_data,
            file_name=f"65_sma_support_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_support_ma_btn"
        )
        
        st.markdown("---")
        # Render the unified Trade Execution Matrix
        st.markdown("### 🛡️ Active 65 SMA Support Trade Execution Sheet")
        render_unified_strategy_table(sorted_support, "support_ma", "support_ma_tab")
