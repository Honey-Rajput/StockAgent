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
    st.markdown("### 🚀 Early Stage 2 Base Breakout Scanner")
    st.markdown("Identifies stocks moving out of a long-term Stage 1 base on the monthly timeframe.")
    
    if 'stage2_results' not in st.session_state:
        st.session_state.stage2_results = None

    # Pick up background scan results if available
    if st.session_state.stage2_results is None and st.session_state.ALL_TAB_SCAN_STATUS["stage2_results"] is not None:
        st.session_state.stage2_results = st.session_state.ALL_TAB_SCAN_STATUS["stage2_results"]
        # Try loading from DB
        today_str = get_market_date(for_display=True)
        try:
            cached_stage2 = database.get_cached_stage2(today_str)
            if cached_stage2 is not None:
                st.session_state.stage2_results = cached_stage2
                # Note: No need to show success message on silent load, just let the table render
        except Exception as e:
            print(f"Failed to load cached stage2: {e}")
        
    s2_col1, s2_col2 = st.columns([2, 8])
    with s2_col1:
        s2_max_runup = st.number_input("Max Run-Up (%)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
        run_stage2_btn = st.button("🔍 Run Stage 2 Scan", width="stretch", type="primary")
        
    if run_stage2_btn:
        with st.spinner(f"Running Monthly Stage 2 Scan on {st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)')}..."):
            s2_universe = st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)')
            if "NIFTY 50" in s2_universe:
                s2_key = "NIFTY 50"
            elif "NIFTY 100" in s2_universe:
                s2_key = "NIFTY 100"
            elif "WATCHLIST" in s2_universe.upper():
                s2_key = "WATCHLIST"
            else:
                s2_key = "NIFTY 500" # Better default for Stage 2 than all NSE
                
            if s2_key == "WATCHLIST":
                import watchlist
                wl = watchlist.load_watchlist()
                s2_cands = [s for s in wl['symbol'].tolist() if pd.notna(s)]
            else:
                s2_cands = get_index_stocks(s2_key)
                
            s2_res = []
            chunk_size = 50
            chunks = [s2_cands[i:i+chunk_size] for i in range(0, len(s2_cands), chunk_size)]
            
            s2_prog = st.progress(0)
            s2_status = st.empty()
            
            def download_s2_chunk(c_idx, chunk):
                chunk_res = []
                tkrs = [f"{s}.NS" for s in chunk]
                try:
                    df_s2 = yf.download(tickers=tkrs, period="5y", interval="1mo", progress=False, threads=False, timeout=15)
                    if not df_s2.empty:
                        for sym in chunk:
                            try:
                                if isinstance(df_s2.columns, pd.MultiIndex):
                                    all_tkrs = df_s2.columns.get_level_values(1).unique().tolist()
                                    matched_t = next((t for t in all_tkrs if t.upper() == f"{sym}.NS".upper()), None)
                                    if not matched_t: continue
                                    t_df = df_s2.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                                else:
                                    t_df = df_s2.dropna(subset=['Close'])
                                    
                                if not t_df.empty and len(t_df) >= 24:
                                    t_df = t_df.reset_index()
                                    t_df.rename(columns={t_df.columns[0]: 'Date'}, inplace=True)
                                    res = scan_monthly_early_stage2(sym, t_df, max_run_up_pct=s2_max_runup)
                                    if res:
                                        chunk_res.append(res)
                            except Exception as parse_ex:
                                pass
                except Exception as down_ex:
                    print(f"Failed to download chunk {c_idx + 1}: {down_ex}")
                return chunk_res

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                for c_idx, chunk in enumerate(chunks):
                    futures.append(executor.submit(download_s2_chunk, c_idx, chunk))
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    s2_status.text(f"Scanning chunks... ({i+1}/{len(chunks)})")
                    s2_res.extend(future.result())
                    s2_prog.progress((i + 1) / len(chunks))
            
            s2_prog.empty()
            s2_status.empty()
            
            # Sort by signal strength (score)
            s2_res = sorted(s2_res, key=lambda x: x.get('score', 0), reverse=True)
            st.session_state.stage2_results = s2_res
            try:
                today_ist_str = get_market_date()
                database.save_stage2_only(today_ist_str, s2_res)
            except Exception as e:
                print(f"Failed to cache stage2 scan: {e}")
            st.success(f"Stage 2 Scan Complete! Found {len(s2_res)} setups.")
            
    st.markdown("---")
    
    if st.session_state.stage2_results is None:
        # Background scan progress indicator
        if st.session_state.ALL_TAB_SCAN_STATUS["is_running"]:
            _bg_scanner = st.session_state.ALL_TAB_SCAN_STATUS["current_scanner"]
            _bg_status = st.session_state.ALL_TAB_SCAN_STATUS["status_text"]
            _bg_progress = st.session_state.ALL_TAB_SCAN_STATUS["progress"]
            st.markdown(f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
                <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. Stage-2 results will appear here when ready!</p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """, unsafe_allow_html=True)
            st.progress(_bg_progress)
            if st.button("🔄 Refresh Scanner Status", key="refresh_bg_s2_status_btn"):
                st.rerun()
                
        st.info("💡 Adjust parameters and click 'Run Stage 2 Scan' to find long-term breakouts.")
    elif len(st.session_state.stage2_results) == 0:
        st.info(f"ℹ️ No early Stage 2 setups found in {st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)')} today.")
    else:
        dl_btn, _ = st.columns([2, 8])
        with dl_btn:
            s2_export_list = []
            for r in st.session_state.stage2_results:
                row = dict(r)
                if 'recommendation' in row:
                    row['Recommendation'] = extract_clean_recommendation(row.pop('recommendation'))
                s2_export_list.append(row)
            s2_df = pd.DataFrame(s2_export_list)
            csv_data = s2_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name="stage2_scan_results.csv",
                mime="text/csv",
                width="stretch"
            )
        render_unified_strategy_table(st.session_state.stage2_results, "stage2", "stage2_tab")
