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
    st.markdown("### 📊 Volume Profile Zones (Daily, Weekly, Monthly)")
    st.info("Scans ALL NSE listed stocks for POC, VAH, VAL levels. Filters: Price > ₹100, Market Cap > 2000 Cr.")
    
    # Auto-load cached results from database on first visit
    # Pick up background scan results if available
    if not st.session_state.get('vp_results') and ALL_TAB_SCAN_STATUS["vp_results"] is not None:
        st.session_state.vp_results = ALL_TAB_SCAN_STATUS["vp_results"]

    if 'vp_results' not in st.session_state or not st.session_state.vp_results:
        try:
            # Try loading today's cached results first, then search last 10 days
            from datetime import timedelta
            for days_back in range(10):
                check_date = (datetime.now(IST_TIMEZONE) - timedelta(days=days_back)).strftime("%Y-%m-%d")
                cached = database.get_cached_volume_profile(check_date)
                if cached:
                    st.session_state.vp_results = cached
                    st.caption(f"📅 Loaded cached results from {check_date}")
                    break
        except Exception as e:
            print(f"Failed to auto-load VP cache: {e}")
    
    col1, col2 = st.columns([3, 7])
    with col1:
        run_vp_btn = st.button("🚀 Run Advanced Volume Profile Scan", width="stretch")
    
    if run_vp_btn:
        st.session_state.vp_results = []
        with st.spinner("Initializing Volume Profile Scan on ALL NSE Stocks..."):
            try:
                vp_list = []
                scan_progress = st.progress(0)
                status_text = st.empty()
                
                from data_fetcher import get_all_nse_symbols
                import yfinance as yf
                import pandas as pd
                import concurrent.futures
                from scanner import scan_volume_profile
                
                raw_symbols = get_all_nse_symbols()
                all_symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
                
                total_symbols = len(all_symbols)
                
                # Phase 1: Bulk OHLCV Download (10 workers, 200 per chunk)
                status_text.text(f"Phase 1: Downloading 2 years of history for {total_symbols} stocks...")
                chunk_size = 200
                sym_chunks = [all_symbols[i:i + chunk_size] for i in range(0, len(all_symbols), chunk_size)]
                
                valid_data = {}
                
                def download_vp_chunk(chunk_idx, chunk):
                    chunk_data = {}
                    try:
                        df_bulk = yf.download(tickers=chunk, period="2y", interval="1d", progress=False, threads=False, timeout=15)
                        if isinstance(df_bulk.columns, pd.MultiIndex):
                            for sym in chunk:
                                try:
                                    if 'Close' in df_bulk.columns.levels[0]:
                                        ticker_df = df_bulk.xs(sym, axis=1, level=1).copy()
                                        ticker_df = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
                                        if len(ticker_df) >= 100:
                                            ticker_df = ticker_df.reset_index()
                                            ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                            ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)
                                            chunk_data[sym] = ticker_df
                                except Exception:
                                    pass
                        else:
                            if len(chunk) == 1 and not df_bulk.empty and 'Close' in df_bulk:
                                ticker_df = df_bulk[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
                                if len(ticker_df) >= 100:
                                    ticker_df = ticker_df.reset_index()
                                    ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)
                                    chunk_data[chunk[0]] = ticker_df
                    except Exception:
                        pass
                    return chunk_data
                    
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                    futures = []
                    for chunk_idx, chunk in enumerate(sym_chunks):
                        futures.append(executor.submit(download_vp_chunk, chunk_idx, chunk))
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            res_data = future.result(timeout=120)
                            valid_data.update(res_data)
                        except Exception:
                            pass
                        scan_progress.progress((i + 1) / len(sym_chunks) * 0.5)
                        status_text.text(f"Phase 1: Downloading history... ({i+1}/{len(sym_chunks)} chunks, {len(valid_data)} stocks loaded)")

                # Phase 2: Compute Volume Profile (simple sequential — fast after numpy optimization)
                status_text.text(f"Phase 2: Computing Volume Profiles for {len(valid_data)} stocks...")
                
                total_to_process = len(valid_data)
                done_count = 0
                if total_to_process == 0:
                    status_text.text("Scan Complete! Found 0 matches.")
                    scan_progress.progress(1.0)
                else:
                    for sym, df in valid_data.items():
                        done_count += 1
                        try:
                            res = scan_volume_profile(sym, df, 0)
                            if res:
                                vp_list.append(res)
                        except Exception:
                            pass
                        
                        if done_count % 50 == 0 or done_count == total_to_process:
                            scan_progress.progress(0.5 + (done_count / total_to_process) * 0.5)
                            status_text.text(f"Scanning Profiles: {done_count}/{total_to_process} | Found: {len(vp_list)}")
                    
                    scan_progress.progress(1.0)
                    status_text.text(f"Scan Complete! Found {len(vp_list)} matches.")
                
                if vp_list:
                    st.session_state.vp_results = vp_list
                    try:
                        today_ist_str = get_market_date()
                        database.save_volume_profile_only(today_ist_str, vp_list)
                    except Exception as e:
                        print(f"Failed to cache Volume Profile scan: {e}")
                    st.success(f"Volume Profile Scan complete! Found {len(vp_list)} stocks.")
                    
            except Exception as e:
                st.error(f"Scan failed: {e}")
                
    if not st.session_state.get('vp_results'):
        # Background scan progress indicator
        if ALL_TAB_SCAN_STATUS["is_running"]:
            _bg_scanner = ALL_TAB_SCAN_STATUS["current_scanner"]
            _bg_status = ALL_TAB_SCAN_STATUS["status_text"]
            _bg_progress = ALL_TAB_SCAN_STATUS["progress"]
            st.markdown(f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
                <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. Volume Profile results will appear here when ready!</p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """, unsafe_allow_html=True)
            st.progress(_bg_progress)
            if st.button("🔄 Refresh Scanner Status", key="refresh_bg_vp_status_btn"):
                st.rerun()
                
        st.info("No Volume Profile data available. Click 'Run Advanced Volume Profile Scan' to process.")
    else:
        vp_data = st.session_state.vp_results
        
        # Helper to safely extract VP level data from a timeframe dict
        def _get_tf(r, tf_key):
            tf = r.get(tf_key)
            if isinstance(tf, dict) and tf:
                return {
                    'zone': tf.get('zone', ''),
                    'va_pct': tf.get('position_pct') if tf.get('position_pct') is not None and tf.get('position_pct') != '' else None,
                    'poc': round(tf['poc'], 2) if tf.get('poc') is not None else None,
                    'val': round(tf['val'], 2) if tf.get('val') is not None else None,
                    'vah': round(tf['vah'], 2) if tf.get('vah') is not None else None
                }
            return {'zone': '', 'va_pct': None, 'poc': None, 'val': None, 'vah': None}
        
        # Format for Dataframe
        import pandas as pd
        vp_export = []
        rank = 1
        
        for r in vp_data:
            d = _get_tf(r, 'daily')
            w = _get_tf(r, 'weekly')
            m = _get_tf(r, 'monthly')
            
            clean_sym = str(r.get('symbol', '')).replace('.NS', '').strip().upper()
            
            vp_export.append({
                'Rank': rank,
                'Symbol': clean_sym,
                'CMP': r.get('cmp', 0),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                # Daily levels
                'D Zone': d['zone'],
                'D Buy Range (VAL)': d['val'],
                'D Target (POC)': d['poc'],
                'D Resistance (VAH)': d['vah'],
                'D VA%': d['va_pct'],
                # Weekly levels
                'W Zone': w['zone'],
                'W Buy Range (VAL)': w['val'],
                'W Target (POC)': w['poc'],
                'W Resistance (VAH)': w['vah'],
                'W VA%': w['va_pct'],
                # Monthly levels
                'M Zone': m['zone'],
                'M Buy Range (VAL)': m['val'],
                'M Target (POC)': m['poc'],
                'M Resistance (VAH)': m['vah'],
                'M VA%': m['va_pct']
            })
            rank += 1
            
        df_vp = pd.DataFrame(vp_export)
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Scanned", len(df_vp))
        with col2:
            st.metric("Daily Buy Zone", len(df_vp[df_vp['D Zone'] == '✅ Can Buy (Near Support)']) if not df_vp.empty else 0)
        with col3:
            st.metric("Weekly Buy Zone", len(df_vp[df_vp['W Zone'] == '✅ Can Buy (Near Support)']) if not df_vp.empty else 0)
        with col4:
            st.metric("Monthly Buy Zone", len(df_vp[df_vp['M Zone'] == '✅ Can Buy (Near Support)']) if not df_vp.empty else 0)
        
        # Column groups per timeframe
        daily_cols = ['Rank', 'Symbol', 'CMP', 'D Zone', 'D Buy Range (VAL)', 'D Target (POC)', 'D Resistance (VAH)', 'D VA%']
        weekly_cols = ['Rank', 'Symbol', 'CMP', 'W Zone', 'W Buy Range (VAL)', 'W Target (POC)', 'W Resistance (VAH)', 'W VA%']
        monthly_cols = ['Rank', 'Symbol', 'CMP', 'M Zone', 'M Buy Range (VAL)', 'M Target (POC)', 'M Resistance (VAH)', 'M VA%']
        
        # Timeframe Tabs
        tab_all, tab_daily, tab_weekly, tab_monthly = st.tabs(["📊 All Stocks", "📅 Daily", "📅 Weekly", "📅 Monthly"])
        
        with tab_all:
            disp_vp = df_vp.copy()
            if 'symbol' in disp_vp.columns:
                disp_vp['symbol'] = disp_vp['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_vp, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_all = df_vp.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download All Stocks (CSV)",
                data=csv_all,
                file_name=f"VP_All_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_all"
            )
        
        with tab_daily:
            df_daily = df_vp[df_vp['D Zone'] != ''][daily_cols].copy()
            df_daily = df_daily.sort_values('D VA%', ascending=True)
            df_daily['Rank'] = range(1, len(df_daily) + 1)
            
            buy_daily = df_daily[df_daily['D Zone'] == '✅ Can Buy (Near Support)']
            st.markdown(f"**{len(buy_daily)}** stocks in Daily Buy Zone | **{len(df_daily)}** total with daily data")
            st.caption("💡 **Buy Range (VAL)** = Support level to buy near | **Target (POC)** = High-volume fair value | **Resistance (VAH)** = Upper boundary")
            disp_daily = df_daily.copy()
            if 'symbol' in disp_daily.columns:
                disp_daily['symbol'] = disp_daily['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_daily, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_daily = df_daily.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Daily Timeframe (CSV)",
                data=csv_daily,
                file_name=f"VP_Daily_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_daily"
            )
        
        with tab_weekly:
            df_weekly = df_vp[df_vp['W Zone'] != ''][weekly_cols].copy()
            df_weekly = df_weekly.sort_values('W VA%', ascending=True)
            df_weekly['Rank'] = range(1, len(df_weekly) + 1)
            
            buy_weekly = df_weekly[df_weekly['W Zone'] == '✅ Can Buy (Near Support)']
            st.markdown(f"**{len(buy_weekly)}** stocks in Weekly Buy Zone | **{len(df_weekly)}** total with weekly data")
            st.caption("💡 **Buy Range (VAL)** = Support level to buy near | **Target (POC)** = High-volume fair value | **Resistance (VAH)** = Upper boundary")
            disp_weekly = df_weekly.copy()
            if 'symbol' in disp_weekly.columns:
                disp_weekly['symbol'] = disp_weekly['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_weekly, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_weekly = df_weekly.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Weekly Timeframe (CSV)",
                data=csv_weekly,
                file_name=f"VP_Weekly_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_weekly"
            )
        
        with tab_monthly:
            df_monthly = df_vp[df_vp['M Zone'] != ''][monthly_cols].copy()
            df_monthly = df_monthly.sort_values('M VA%', ascending=True)
            df_monthly['Rank'] = range(1, len(df_monthly) + 1)
            
            buy_monthly = df_monthly[df_monthly['M Zone'] == '✅ Can Buy (Near Support)']
            st.markdown(f"**{len(buy_monthly)}** stocks in Monthly Buy Zone | **{len(df_monthly)}** total with monthly data")
            st.caption("💡 **Buy Range (VAL)** = Support level to buy near | **Target (POC)** = High-volume fair value | **Resistance (VAH)** = Upper boundary")
            disp_monthly = df_monthly.copy()
            if 'symbol' in disp_monthly.columns:
                disp_monthly['symbol'] = disp_monthly['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_monthly, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_monthly = df_monthly.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Monthly Timeframe (CSV)",
                data=csv_monthly,
                file_name=f"VP_Monthly_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_monthly"
            )
        
        # Combined Excel download with all sheets
        try:
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_vp.to_excel(writer, sheet_name='All Stocks', index=False)
                if not df_vp.empty:
                    df_d_buy = df_vp[df_vp['D Zone'] == '✅ Can Buy (Near Support)']
                    df_w_buy = df_vp[df_vp['W Zone'] == '✅ Can Buy (Near Support)']
                    df_m_buy = df_vp[df_vp['M Zone'] == '✅ Can Buy (Near Support)']
                    
                    if not df_d_buy.empty:
                        df_d_buy[daily_cols].to_excel(writer, sheet_name='Daily Buy Zone', index=False)
                    if not df_w_buy.empty:
                        df_w_buy[weekly_cols].to_excel(writer, sheet_name='Weekly Buy Zone', index=False)
                    if not df_m_buy.empty:
                        df_m_buy[monthly_cols].to_excel(writer, sheet_name='Monthly Buy Zone', index=False)
            
            st.download_button(
                label="📥 Download Complete Report (Excel - All Sheets)",
                data=excel_buffer.getvalue(),
                file_name=f"Volume_Profile_Scan_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_vp_excel"
            )
        except ImportError:
            st.caption("ℹ️ Excel export unavailable — use CSV downloads above instead.")
