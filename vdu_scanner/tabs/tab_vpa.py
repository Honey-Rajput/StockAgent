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
    st.markdown("### 🚥 VPA Trend Indicator (Daily, Weekly, Monthly)")
    st.info("Scans ALL NSE listed stocks. Filters: Price > ₹100. Shows Major, Mid, and Minor trends across timeframes.")
    
    # Pick up background scan results if available
    if not st.session_state.get('vpa_results') and ALL_TAB_SCAN_STATUS["vpa_results"] is not None:
        st.session_state.vpa_results = ALL_TAB_SCAN_STATUS["vpa_results"]

    col1, col2 = st.columns([3, 7])
    with col1:
        run_vpa_btn = st.button("🚀 Run Advanced VPA Scan", width="stretch", disabled=not is_admin)
    
    if run_vpa_btn:
        st.session_state.vpa_results = []
        with st.spinner("Initializing Ultra-Fast VPA Scan on ALL NSE Stocks..."):
            try:
                from data_fetcher import get_all_nse_symbols
                import yfinance as yf
                import pandas as pd
                from concurrent.futures import ThreadPoolExecutor
                import time
                
                raw_symbols = get_all_nse_symbols()
                all_symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
                
                # Phase 1: Bulk OHLCV Download (2 years history for Weekly/Monthly VPA)
                st.info(f"Phase 1: Downloading 5 years of history for {len(all_symbols)} stocks...")
                prog = st.progress(0)
                status = st.empty()
                
                chunk_size = 300
                sym_chunks = [all_symbols[i:i + chunk_size] for i in range(0, len(all_symbols), chunk_size)]
                
                valid_data = {}
                price_filtered = []
                
                # We need at least ~100 days of history for VPA to calculate daily/weekly accurately
                def download_vpa_chunk(chunk_idx, chunk):
                    chunk_data = {}
                    chunk_filtered = []
                    try:
                        df_bulk = yf.download(tickers=chunk, period="5y", interval="1d", progress=False, threads=False, timeout=15)
                        if isinstance(df_bulk.columns, pd.MultiIndex):
                            for sym in chunk:
                                try:
                                    if 'Close' in df_bulk.columns.levels[0]:
                                        ticker_df = df_bulk.xs(sym, axis=1, level=1).copy()
                                        ticker_df = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
                                        if len(ticker_df) >= 45 and ticker_df['Close'].iloc[-1] > 100.0:
                                            ticker_df = ticker_df.reset_index()
                                            ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                            ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)
                                            chunk_data[sym] = ticker_df
                                            chunk_filtered.append(sym)
                                except Exception:
                                    pass
                        else:
                            if len(chunk) == 1 and not df_bulk.empty and 'Close' in df_bulk:
                                ticker_df = df_bulk[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
                                if len(ticker_df) >= 45 and ticker_df['Close'].iloc[-1] > 100.0:
                                    ticker_df = ticker_df.reset_index()
                                    ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)
                                    chunk_data[chunk[0]] = ticker_df
                                    chunk_filtered.append(chunk[0])
                    except Exception:
                        pass
                    return chunk_data, chunk_filtered
                    
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                    futures = []
                    for chunk_idx, chunk in enumerate(sym_chunks):
                        futures.append(executor.submit(download_vpa_chunk, chunk_idx, chunk))
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        res_data, res_filtered = future.result()
                        valid_data.update(res_data)
                        price_filtered.extend(res_filtered)
                        prog.progress((i + 1) / len(sym_chunks))
                        status.text(f"Fetching bulk history chunks... ({i+1}/{len(sym_chunks)})")
                
                # Phase 2: Final VPA Compute (Instant)
                st.info("Phase 2: Calculating VPA Trends (Instant)...")
                status.empty()
                prog.progress(1.0)
                
                vpa_list = []
                for sym in price_filtered:
                    df = valid_data[sym]
                    clean_sym = sym.replace('.NS', '')
                    vpa_res = scan_vpa_trend(clean_sym, df)
                    if vpa_res is not None:
                        vpa_res['market_cap_cr'] = 0  # Default since bulk fetch rate-limits
                        vpa_list.append(vpa_res)
                            
                prog.empty()
                status.empty()
                st.session_state.vpa_results = vpa_list
                try:
                    today_ist_str = get_market_date()
                    database.save_vpa_only(today_ist_str, vpa_list)
                except Exception as e:
                    print(f"Failed to cache custom VPA scan: {e}")
                st.success(f"VPA Scan complete! Found {len(vpa_list)} stocks meeting all criteria and saved to database.")
                
            except Exception as e:
                st.error(f"Scan failed: {e}")
                
    if not st.session_state.get('vpa_results'):
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
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. VPA results will appear here when ready!</p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """, unsafe_allow_html=True)
            st.progress(_bg_progress)
            if st.button("🔄 Refresh Scanner Status", key="refresh_bg_vpa_status_btn"):
                st.rerun()
                
        st.info("No VPA data available. Click 'Run Advanced VPA Scan' to process.")
    else:
        vpa_data = st.session_state.vpa_results
        
        # Sort by score
        vpa_data = sorted(vpa_data, key=lambda x: x.get('score', 0), reverse=True)
        
        # Download Button
        import pandas as pd
        
        def get_action_signal_text(short, mid, max_t, max_val, rsi=0, cci=0):
            # Issue #3: RSI/CCI overbought guard
            if rsi > 80 or cci > 200:
                return "Overbought (Wait for Pullback)"
            if max_val > 4.0:
                return "Hyper-Extended / Parabolic (Avoid Fresh Entry)"
            elif max_val > 2.0:
                return "Slightly Overextended (Avoid Fresh Entry)"
            
            if short == 1 and mid == 1 and max_t == 1:
                return "Perfect Buy / Strong Hold"
            elif short == 1 and mid == 1 and max_t == 0:
                return "Early Breakout Entry"
            elif short == 1 and mid == 1 and max_t == -1:
                return "Counter Trend Buy (Major Down)"
            elif mid == 1 and short <= 0:
                return "Pullback (Wait for Short=Up)"
            elif mid <= 0 and max_t == 1:
                return "Warning (Mid Broken) - Trim"
            elif mid <= 0 and max_t <= 0:
                return "Avoid / Full Exit"
            
            return "Neutral / Choppy"
        
        def get_signal(short, mid, max_t, max_val):
            if short == 1 and mid == 1:
                return "Buy"
            elif mid == 1 or max_t == 1:
                return "Hold"
            return "Sell"

        only_buy_signals = st.checkbox("🟢 Show Only 'Buy' Signals", value=False)
        
        # Issue #2: Move timeframe selection BEFORE filter so we can filter by the correct timeframe
        st.markdown("### Select Timeframe")
        selected_tf = st.selectbox("Timeframe to display", ["Daily", "Weekly", "Monthly"], key="vpa_tf_select")
        
        daily_export = []
        weekly_export = []
        monthly_export = []
        
        rank = 1
        filtered_vpa_data = []
        for r in vpa_data:
            d = r['daily']; w = r['weekly']; m = r['monthly']
            
            # Issue #2: Filter by the SELECTED timeframe signal, not always daily
            if selected_tf == "Weekly":
                tf_data = w
            elif selected_tf == "Monthly":
                tf_data = m
            else:
                tf_data = d
            
            tf_sig = get_signal(tf_data['minor'], tf_data['mid'], tf_data['major'], tf_data.get('major_val', 0))
            if only_buy_signals and tf_sig != "Buy":
                continue
                
            filtered_vpa_data.append((rank, r))
            
            daily_export.append({
                'Rank': rank,
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                'Major Trend': "Up" if d['major'] == 1 else ("Down" if d['major'] == -1 else "Neutral"),
                'Mid Trend': "Up" if d['mid'] == 1 else ("Down" if d['mid'] == -1 else "Neutral"),
                'Minor Trend': "Up" if d['minor'] == 1 else ("Down" if d['minor'] == -1 else "Neutral"),
                'RSI': d.get('rsi', 0.0),
                'CCI': d.get('cci', 0.0),
                'Action': get_action_signal_text(d['minor'], d['mid'], d['major'], d.get('major_val', 0), rsi=d.get('rsi', 0), cci=d.get('cci', 0)),
                'Signal': get_signal(d['minor'], d['mid'], d['major'], d.get('major_val', 0)),
                'Score': r.get('score', 0),
                'Confidence': r.get('confidence', 'N/A')
            })
            rank += 1
            
        for rank, r in filtered_vpa_data:
            w = r['weekly']
            weekly_export.append({
                'Rank': rank,
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                'Major Trend': "Up" if w['major'] == 1 else ("Down" if w['major'] == -1 else "Neutral"),
                'Mid Trend': "Up" if w['mid'] == 1 else ("Down" if w['mid'] == -1 else "Neutral"),
                'Minor Trend': "Up" if w['minor'] == 1 else ("Down" if w['minor'] == -1 else "Neutral"),
                'RSI': w.get('rsi', 0.0),
                'CCI': w.get('cci', 0.0),
                'Action': get_action_signal_text(w['minor'], w['mid'], w['major'], w.get('major_val', 0), rsi=w.get('rsi', 0), cci=w.get('cci', 0)),
                'Signal': get_signal(w['minor'], w['mid'], w['major'], w.get('major_val', 0)),
                'Score': r.get('score', 0),
                'Confidence': r.get('confidence', 'N/A')
            })
            
        for rank, r in filtered_vpa_data:
            m = r['monthly']
            monthly_export.append({
                'Rank': rank,
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                'Major Trend': "Up" if m['major'] == 1 else ("Down" if m['major'] == -1 else "Neutral"),
                'Mid Trend': "Up" if m['mid'] == 1 else ("Down" if m['mid'] == -1 else "Neutral"),
                'Minor Trend': "Up" if m['minor'] == 1 else ("Down" if m['minor'] == -1 else "Neutral"),
                'RSI': m.get('rsi', 0.0),
                'CCI': m.get('cci', 0.0),
                'Action': get_action_signal_text(m['minor'], m['mid'], m['major'], m.get('major_val', 0), rsi=m.get('rsi', 0), cci=m.get('cci', 0)),
                'Signal': get_signal(m['minor'], m['mid'], m['major'], m.get('major_val', 0)),
                'Score': r.get('score', 0),
                'Confidence': r.get('confidence', 'N/A')
            })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download Daily VPA (CSV)",
                data=pd.DataFrame(daily_export).to_csv(index=False).encode('utf-8-sig'),
                file_name="vpa_daily_trend.csv",
                mime="text/csv",
                width="stretch"
            )
        with col2:
            st.download_button(
                label="📥 Download Weekly VPA (CSV)",
                data=pd.DataFrame(weekly_export).to_csv(index=False).encode('utf-8-sig'),
                file_name="vpa_weekly_trend.csv",
                mime="text/csv",
                width="stretch"
            )
        with col3:
            st.download_button(
                label="📥 Download Monthly VPA (CSV)",
                data=pd.DataFrame(monthly_export).to_csv(index=False).encode('utf-8-sig'),
                file_name="vpa_monthly_trend.csv",
                mime="text/csv",
                width="stretch"
            )
        
        # Timeframe selection was moved above the filter loop
        # selected_tf is already defined above
        
        def trend_to_badge(t_val):
            if t_val == 1:
                return "<span style='color: #00e676; font-weight: bold;'>Up (1)</span>"
            elif t_val == -1:
                return "<span style='color: #ef4444; font-weight: bold;'>Dn (-1)</span>"
            return "<span style='color: #fbbf24; font-weight: bold;'>Neu (0)</span>"
            
        def get_action_signal(short, mid, max_t, max_val, rsi=0, cci=0):
            text = get_action_signal_text(short, mid, max_t, max_val, rsi=rsi, cci=cci)
            if "Perfect Buy" in text:
                return f"<span style='color: #00e676; font-weight: bold;'>🟢 {text}</span>"
            elif "Counter Trend Buy" in text:
                return f"<span style='color: #4ade80; font-weight: bold;'>🟢 {text}</span>"
            elif "Early Breakout" in text:
                return f"<span style='color: #3b82f6; font-weight: bold;'>🔵 {text}</span>"
            elif "Pullback" in text:
                return f"<span style='color: #fbbf24; font-weight: bold;'>🟡 {text}</span>"
            elif "Warning" in text:
                return f"<span style='color: #f97316; font-weight: bold;'>🟠 {text}</span>"
            elif "Avoid" in text:
                return f"<span style='color: #ef4444; font-weight: bold;'>🔴 {text}</span>"
            elif "Parabolic" in text or "Overextended" in text:
                return f"<span style='color: #d946ef; font-weight: bold;'>🟣 {text}</span>"
            else:
                return f"<span style='color: #9ca3af; font-weight: bold;'>⚪ {text}</span>"
            
        html_rows = []
        for rank, r in filtered_vpa_data:
            if selected_tf == "Daily":
                tf_data = r['daily']
            elif selected_tf == "Weekly":
                tf_data = r['weekly']
            else:
                tf_data = r['monthly']
            
            t_short = trend_to_badge(tf_data['minor'])
            t_mid = trend_to_badge(tf_data['mid'])
            t_max = trend_to_badge(tf_data['major'])
            
            action = get_action_signal(tf_data['minor'], tf_data['mid'], tf_data['major'], tf_data.get('major_val', 0), rsi=tf_data.get('rsi', 0), cci=tf_data.get('cci', 0))
            
            # Zero indentation to prevent Streamlit markdown codeblock rendering
            row = f"""<tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
<td style="padding: 10px; font-weight: bold; color: #94a3b8;">#{rank}</td>
<td style="padding: 10px;"><strong>{r['symbol']}</strong></td>
<td style="padding: 10px;">{r['cmp']}</td>
<td style="padding: 10px;">{get_day_change_badge_html(r['day_change_pct'])}</td>
<td style="padding: 10px;">{round(r.get('market_cap_cr', 0))}</td>
<td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">{t_short}</td>
<td style="padding: 10px;">{t_mid}</td>
<td style="padding: 10px;">{t_max}</td>
<td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">{action}</td>
<td style="padding: 10px; font-weight: bold; color: #a3e635;">{r.get('score', 0)}</td>
<td style="padding: 10px;">{r.get('confidence', 'N/A')}</td>
</tr>"""
            html_rows.append(row)
            
        rows_str = "".join(html_rows)
        
        table_html = f"""<div style="overflow-x: auto; margin-top: 10px;">
<table style="width: 100%; text-align: left; border-collapse: collapse; font-size: 0.95rem;">
<thead>
<tr style="background-color: rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.1);">
<th style="padding: 10px;">Rank</th>
<th style="padding: 10px;">Symbol</th>
<th style="padding: 10px;">CMP</th>
<th style="padding: 10px;">Chg %</th>
<th style="padding: 10px;">M.Cap (Cr)</th>
<th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">Short Term</th>
<th style="padding: 10px;">Mid Term</th>
<th style="padding: 10px;">Max Term</th>
<th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">Action / Signal</th>
<th style="padding: 10px;">Score</th>
<th style="padding: 10px;">Confidence</th>
</tr>
</thead>
<tbody>
{rows_str}
</tbody>
</table>
</div>"""
        st.markdown(table_html, unsafe_allow_html=True)
