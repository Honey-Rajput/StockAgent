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
    st.markdown("### 🎯 Minervini Ultimate +VCP")
    st.markdown("Scans for stocks passing the Minervini Trend Template with Volatility Contraction Pattern (VCP) squeeze.")

    col_v1, col_v2, col_v3 = st.columns(3)
    vcp_thresh = col_v1.number_input("Max VCP Range (%)", value=2.5, step=0.5, help="Maximum percentage range over lookback to be considered a squeeze")
    vcp_lookback = col_v2.number_input("VCP Lookback (Bars)", value=5, step=1)
    risk_low = col_v3.number_input("Max Low Risk (%)", value=15.0, step=1.0, help="Maximum distance above 50SMA to be considered Low Risk")

    st.markdown("---")
    
    col_v_btn, col_v_note = st.columns([1, 2])
    run_vcp_btn = col_v_btn.button("🔍 Run VCP+Minervini Scan", type="primary", use_container_width=True)
    
    if run_vcp_btn:
        with st.spinner("Running Minervini VCP scan across all NSE stocks..."):
            from vcp_minervini import VCPConfig, MinerviniVCPAnalyzer
            from data_fetcher import get_all_nse_symbols, get_stock_sector
            import yfinance as yf
            import pandas as pd
            
            # Use all NSE symbols (capped at 1800)
            raw_symbols = get_all_nse_symbols()
            symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
            cfg = VCPConfig(
                vcp_thresh=vcp_thresh,
                vcp_lookback=int(vcp_lookback),
                risk_low=risk_low
            )
            
            # Download benchmark for RS Proxy calculation
            try:
                benchmark_df = yf.download("^NSEI", period="2y", interval="1d", progress=False, threads=False, timeout=15)
            except Exception:
                benchmark_df = None
                
            # Batch download data for speed, then run analyzer per-stock
            vcp_results = []
            chunk_size = 100
            sym_chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]
            progress_bar = st.progress(0, text="Downloading data...")
            
            for c_idx, chunk in enumerate(sym_chunks):
                progress_bar.progress(
                    (c_idx + 1) / len(sym_chunks),
                    text=f"Processing chunk {c_idx+1}/{len(sym_chunks)} ({len(vcp_results)} setups found)..."
                )
                try:
                    bulk_df = yf.download(tickers=chunk, period="2y", interval="1d", progress=False, threads=False, timeout=15)
                    if bulk_df.empty:
                        continue
                    
                    for sym in chunk:
                        try:
                            # Extract single-stock data from bulk download
                            if isinstance(bulk_df.columns, pd.MultiIndex):
                                all_tkrs = bulk_df.columns.get_level_values(1).unique().tolist()
                                matched_t = next((t for t in all_tkrs if t.upper() == sym.upper()), None)
                                if not matched_t:
                                    continue
                                t_df = bulk_df.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                            else:
                                t_df = bulk_df.dropna(subset=['Close'])
                            
                            if t_df.empty or len(t_df) < 250:
                                continue
                            
                            # Run the Minervini analyzer with pre-fetched data
                            analyzer = MinerviniVCPAnalyzer(sym, cfg, benchmark_df=benchmark_df)
                            analyzer.df = t_df.copy()
                            analyzer._moving_averages()
                            analyzer._trend_template()
                            analyzer._buy_risk()
                            analyzer._pressure()
                            analyzer._relative_price_strength()
                            analyzer._vcp()
                            analyzer._entry_signals()
                            
                            last = analyzer.df.iloc[-1]
                            result = {
                                "symbol": sym,
                                "date": analyzer.df.index[-1].strftime("%Y-%m-%d"),
                                "close": round(float(last["Close"]), 2),
                                "Pressure": last["pressure_txt"],
                                "Risk (50d)": last["risk_status"],
                                "Trend (TPR)": last["tpr_txt"],
                                "RS Proxy": round(float(last["rpr_proxy"]), 1) if pd.notna(last["rpr_proxy"]) else None,
                                "VCP (5d)": last["vcp_txt"],
                                "VCP range %": round(float(last["vcp_range_pct"]), 2) if pd.notna(last["vcp_range_pct"]) else None,
                                "VCP (10d)": last.get("vcp10_txt", "Normal"),
                                "VCP 10d range %": round(float(last["vcp10_range_pct"]), 2) if "vcp10_range_pct" in last and pd.notna(last["vcp10_range_pct"]) else None,
                                "VCP (15d)": last.get("vcp15_txt", "Normal"),
                                "VCP 15d range %": round(float(last["vcp15_range_pct"]), 2) if "vcp15_range_pct" in last and pd.notna(last["vcp15_range_pct"]) else None,
                                "Entry Signal": last["entry_signal"],
                            }
                            
                            result["Sector"] = get_stock_sector(sym)
                            vcp_results.append(result)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            progress_bar.empty()
            
            if vcp_results:
                # Add score and rank
                vcp_df = pd.DataFrame(vcp_results)
                rs_proxy = pd.to_numeric(vcp_df.get('RS Proxy', 50), errors='coerce').fillna(50)
                vcp_range = pd.to_numeric(vcp_df.get('VCP range %', 100), errors='coerce').fillna(100)
                vcp_df['Score'] = rs_proxy - (vcp_range * 5)
                vcp_df = vcp_df.sort_values(by='Score', ascending=False)
                vcp_df.insert(0, 'Rank', range(1, len(vcp_df) + 1))
                st.session_state.vcp_minervini_results = vcp_df.to_dict('records')
                
                # Save to database
                try:
                    today_str = get_market_date()
                    database.save_vcp_minervini_scan(today_str, st.session_state.vcp_minervini_results)
                except Exception as e:
                    print(f"Error saving VCP+Minervini scan: {e}")
            else:
                st.session_state.vcp_minervini_results = []
            
            st.rerun()

    # Display results
    if st.session_state.get('vcp_minervini_results'):
        import pandas as pd
        v_df = pd.DataFrame(st.session_state.vcp_minervini_results)
        
        vcp_count = len(v_df)
        st.success(f"Found {vcp_count} VCP+Minervini setups!")
        
        # CSV Download
        col_btn, _ = st.columns([2, 8])
        with col_btn:
            vcp_csv = v_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ Download CSV",
                data=vcp_csv,
                file_name="vcp_minervini_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        col1, col2 = st.columns(2)
        with col1:
            show_buyable = st.checkbox("Show Buyable Only (Buying Pressure, Low Risk, PASSED Trend)", value=False)
        with col2:
            show_squeeze = st.checkbox("Show 'Squeeze' / Entry Signals Only", value=True)

        # Reorder columns for display
        display_cols = ['Rank', 'Score', 'symbol', 'Sector', 'close', 'Entry Signal', 'Trend (TPR)', 
                       'Pressure', 'Risk (50d)', 'RS Proxy', 'VCP (5d)', 'VCP range %', 'VCP (10d)', 'VCP 10d range %', 'VCP (15d)', 'VCP 15d range %', 'date']
        available_cols = [c for c in display_cols if c in v_df.columns]
        v_df = v_df[available_cols]
        
        if show_buyable:
            if 'Pressure' in v_df.columns:
                v_df = v_df[v_df['Pressure'].str.contains('Buying', case=False, na=False)]
            if 'Risk (50d)' in v_df.columns:
                v_df = v_df[v_df['Risk (50d)'].str.contains('Low Risk', case=False, na=False)]
            if 'Trend (TPR)' in v_df.columns:
                v_df = v_df[v_df['Trend (TPR)'].str.contains('PASSED', case=False, na=False)]
                
        if show_squeeze:
            if 'Entry Signal' in v_df.columns:
                mask = v_df['Entry Signal'].isin(["BREAKOUT", "EARLY ENTRY"])
                if 'VCP (5d)' in v_df.columns: mask = mask | (v_df['VCP (5d)'] == 'SQUEEZE')
                if 'VCP (10d)' in v_df.columns: mask = mask | (v_df['VCP (10d)'] == 'SQUEEZE')
                if 'VCP (15d)' in v_df.columns: mask = mask | (v_df['VCP (15d)'] == 'SQUEEZE')
                v_df = v_df[mask]
        
        if 'symbol' in v_df.columns:
            v_df['symbol'] = v_df['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            
        st.dataframe(v_df, use_container_width=True, column_config={
            "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)"),
            "date": st.column_config.TextColumn("Date"),
            "close": st.column_config.NumberColumn("CMP", format="%.2f"),
            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            "RS Proxy": st.column_config.NumberColumn("RS Proxy (vs Nifty)", format="%.1f"),
            "VCP range %": st.column_config.NumberColumn("VCP Range 5d %", format="%.2f"),
            "VCP 10d range %": st.column_config.NumberColumn("VCP Range 10d %", format="%.2f"),
            "VCP 15d range %": st.column_config.NumberColumn("VCP Range 15d %", format="%.2f"),
            "Entry Signal": st.column_config.TextColumn("Entry Signal"),
            "Trend (TPR)": st.column_config.TextColumn("Trend Template"),
        })
    else:
        st.info("💡 Click 'Run VCP+Minervini Scan' to find setups.")
