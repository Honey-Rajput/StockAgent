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
    st.markdown("### 📈 9/21 EMA Support")
    st.markdown("Stocks taking support at their 9 or 21 EMA with tight proximity, plus crossover signals.")
    
    col_btn, col_note = st.columns([1, 2])
    run_bb_btn = col_btn.button("🔍 Run EMA Support Scan", type="primary", use_container_width=True)
    
    if run_bb_btn:
        st.session_state.ema_support_results = None
        st.session_state.ALL_TAB_SCAN_STATUS["ema_support_results"] = None
        run_background_ema_support_scan(force=True)
        st.rerun()
        
    if st.session_state.get('ema_support_results') is None and st.session_state.ALL_TAB_SCAN_STATUS.get("ema_support_results") is not None:
        st.session_state.ema_support_results = st.session_state.ALL_TAB_SCAN_STATUS["ema_support_results"]
        
    if st.session_state.get('ema_support_results') is not None:
        ema_list = st.session_state.ema_support_results
        
        # Apply Universe Filter
        if "ALL NSE" not in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)').upper() and len(ema_list) > 0:
            from data_fetcher import get_index_stocks
            resolved_univ = "ALL NSE"
            if "NIFTY 500" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): resolved_univ = "NIFTY 500"
            elif "NIFTY 100" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): resolved_univ = "NIFTY 100"
            elif "NIFTY 50" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): resolved_univ = "NIFTY 50"
            elif "WATCHLIST" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)').upper(): resolved_univ = "WATCHLIST"
            if resolved_univ != "ALL NSE":
                raw_symbols = get_index_stocks(resolved_univ)
                valid_set = set([str(s).replace('.NS', '').strip().upper() for s in raw_symbols if str(s).strip()])
                ema_list = [r for r in ema_list if r['symbol'] in valid_set]
                
        if len(ema_list) > 0:
            st.markdown("### 📊 9/21 EMA Support Setups")
            
            # --- Download Button Logic ---
            from datetime import datetime
            today_str = get_market_date()
            df_ema = pd.DataFrame([{
                'Symbol': r.get('symbol', ''),
                'CMP': r.get('cmp', 0.0),
                'Change %': round(float(r.get('day_change_pct') or 0.0), 2),
                'Dist to 9 EMA %': round(float(r.get('dist_9ema') or 0.0), 2),
                'Dist to 21 EMA %': round(float(r.get('dist_21ema') or 0.0), 2),
                'Crossover': r.get('crossover', False),
                'Setup': r.get('setup', ''),
                'Buy Price': r.get('buy_price', 0.0),
                'Exit Price': r.get('exit_price', 0.0),
                'Target Price': r.get('target_price', 0.0),
                'Confidence': r.get('confidence', ''),
                'Recommendation': r.get('recommendation', '')
            } for r in ema_list])
            
            csv_ema = df_ema.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download EMA Support Results (CSV)",
                data=csv_ema,
                file_name=f"EMA_Support_{today_str}.csv",
                mime="text/csv",
                key="dl_ema_support_csv"
            )
            
            render_unified_strategy_table(ema_list, "ema_support", "ema_support_tab")
        else:
            if st.session_state.get("ema_support_running", False) or st.session_state.ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
                st.info("⏳ Background scanner is analyzing EMA Support... Please wait.")
            else:
                st.info("✅ Scan completed — no EMA Support setups found for the selected universe.")
    else:
        if st.session_state.get("ema_support_running", False) or st.session_state.ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
            st.info("⏳ Background scanner is analyzing EMA Support... Please wait.")
        else:
            st.warning("⚠️ Scan has not been run yet. Click **'Run EMA Support Scan'** above to start, or enable **Auto-Background Scans** in the sidebar.")
