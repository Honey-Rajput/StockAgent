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
    st.markdown("### 📉 Dan Zanger Breakout Scanner")
    st.markdown("Identifies stocks meeting Dan Zanger's criteria: Uptrend stack, prior massive run, shallow base, and high-volume breakout.")
    
    st.markdown("#### Scanner Parameters (Tweak if 0 results)")
    col_z1, col_z2, col_z3 = st.columns(3)
    z_min_run_pct = col_z1.number_input("Min Prior Run (%)", value=25.0, step=5.0)
    z_hft_run_pct = col_z2.number_input("High-Tight Flag Run (%)", value=90.0, step=10.0)
    z_max_base_depth = col_z3.number_input("Max Base Depth (%)", value=25.0, step=5.0)
    
    col_z4, col_z5, col_z6 = st.columns(3)
    z_vol_mult = col_z4.number_input("Breakout Vol Multiplier", value=2.0, step=0.5)
    z_base_lookback = col_z5.number_input("Base Lookback (Bars)", value=15, step=5)
    z_hft_max_base = col_z6.number_input("HFT Max Base Depth (%)", value=20.0, step=5.0)

    st.markdown("---")
    z_require_uptrend = st.checkbox("Require Strict Uptrend (Close > 50MA > 150MA > 200MA)", value=True, help="Uncheck this if you want to find setups even when the stock is in a broad market correction/downtrend.")
    
    st.markdown("---")
    
    def on_zanger_tf_change():
        import database
        zanger_dates = database.get_zanger_scan_dates()
        if zanger_dates:
            try:
                st.session_state.zanger_results = database.get_cached_zanger(zanger_dates[0], timeframe=st.session_state.zanger_tf)
            except Exception:
                st.session_state.zanger_results = []
        else:
            st.session_state.zanger_results = []

    col_z7, _ = st.columns([2, 8])
    with col_z7:
        zanger_tf = st.selectbox("Scan Timeframe", ["Daily", "Weekly"], index=0, key="zanger_tf", on_change=on_zanger_tf_change)
        
    st.markdown("---")
    
    col_btn, col_note = st.columns([1, 2])
    run_zanger_btn = col_btn.button("🔍 Run Dan Zanger Scan", type="primary", use_container_width=True)
    
    if run_zanger_btn:
        with st.spinner("Running Dan Zanger scan..."):
            import yfinance as yf
            from zanger_scanner import ZangerConfig, scan_zanger, get_latest_signal, rank_signals
            
            # Universe is hardcoded to Top 1000 NSE stocks
            from data_fetcher import get_top1000_nse_symbols
            zanger_candidates = get_top1000_nse_symbols()
            
            zanger_results = []
            chunk_size = 50
            chunks = [zanger_candidates[i:i+chunk_size] for i in range(0, len(zanger_candidates), chunk_size)]
            cfg = ZangerConfig(
                min_run_pct=float(z_min_run_pct),
                hft_run_pct=float(z_hft_run_pct),
                max_base_depth_pct=float(z_max_base_depth),
                breakout_vol_mult=float(z_vol_mult),
                base_lookback=int(z_base_lookback),
                hft_max_base_depth_pct=float(z_hft_max_base),
                require_uptrend=z_require_uptrend
            )
            
            yf_interval = "1wk" if zanger_tf == "Weekly" else "1d"
            yf_period = "10y" if zanger_tf == "Weekly" else "1100d"
            
            if zanger_tf == "Weekly":
                cfg.ma_fast = 10     # 50 days = 10 weeks
                cfg.ma_slow = 30     # 150 days = 30 weeks
                cfg.ma_slowest = 40  # 200 days = 40 weeks
                cfg.base_lookback = max(3, cfg.base_lookback // 5)
            
            for chunk in chunks:
                tkrs = [f"{s}.NS" for s in chunk]
                try:
                    df_zanger = yf.download(tickers=tkrs, period=yf_period, interval=yf_interval, progress=False, threads=False, timeout=15)
                    if not df_zanger.empty:
                        for sym in chunk:
                            try:
                                if isinstance(df_zanger.columns, pd.MultiIndex):
                                    all_tkrs = df_zanger.columns.get_level_values(1).unique().tolist()
                                    matched_t = next((t for t in all_tkrs if t.upper() == f"{sym}.NS".upper()), None)
                                    if not matched_t:
                                        continue
                                    t_df = df_zanger.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                                else:
                                    t_df = df_zanger.dropna(subset=['Close'])
                                    
                                if not t_df.empty and len(t_df) > cfg.ma_slowest + 5:
                                    t_df = t_df.reset_index()
                                    if 'Date' not in t_df.columns:
                                        t_df.rename(columns={t_df.columns[0]: 'Date'}, inplace=True)
                                    t_df['Date'] = pd.to_datetime(t_df['Date'])
                                    t_df.set_index('Date', inplace=True)
                                    
                                    res_df = scan_zanger(t_df, cfg)
                                    latest = get_latest_signal(res_df)
                                    
                                    if latest.get("zanger_signal", False):
                                        from data_fetcher import get_stock_sector
                                        latest["symbol"] = sym
                                        latest["sector"] = get_stock_sector(sym)
                                        zanger_results.append(latest)
                            except Exception as e:
                                pass
                except Exception as e:
                    pass
            
            if len(zanger_results) > 0:
                import pandas as pd
                hits_df = pd.DataFrame(zanger_results)
                ranked_df = rank_signals(hits_df, cfg)
                # Convert back to dicts for session_state to be consistent
                st.session_state.zanger_results = ranked_df.to_dict('records')
                try:
                    database.save_zanger_scan(get_market_date(), zanger_tf, st.session_state.zanger_results)
                except Exception as e:
                    print(f"Error saving Dan Zanger scan: {e}")
                st.success(f"Dan Zanger Scan Complete! Found {len(zanger_results)} setups ({zanger_tf}).")
            else:
                st.session_state.zanger_results = []
                st.info("No Dan Zanger setups found today.")
                
    if st.session_state.get('zanger_results') is not None:
        if len(st.session_state.zanger_results) > 0:
            import pandas as pd
            z_df = pd.DataFrame(st.session_state.zanger_results)
            # Reorder columns to put rank and symbol first
            cols = list(z_df.columns)
            if 'date' in cols:
                # keep as string, Streamlit will be told it's text
                z_df['date'] = z_df['date'].astype(str).str[:10]
            
            # Clean up unwanted columns (like company_name if sector is there)
            if 'company_name' in cols and 'sector' in cols:
                cols.remove('company_name')
                z_df = z_df.drop(columns=['company_name'])
                
            if 'rank' in cols and 'symbol' in cols:
                cols.insert(0, cols.pop(cols.index('rank')))
                cols.insert(1, cols.pop(cols.index('symbol')))
                if 'sector' in cols:
                    cols.insert(2, cols.pop(cols.index('sector')))
                if 'score' in cols:
                    cols.insert(3, cols.pop(cols.index('score')))
                if 'confidence_level' in cols:
                    cols.insert(4, cols.pop(cols.index('confidence_level')))
                if 'breakout_status' in cols:
                    cols.insert(5, cols.pop(cols.index('breakout_status')))
                if 'target_price' in cols:
                    risk_idx = cols.index('risk_pct') if 'risk_pct' in cols else len(cols)
                    cols.insert(risk_idx + 1, cols.pop(cols.index('target_price')))
                z_df = z_df[cols]
                
            if 'symbol' in z_df.columns:
                z_df['symbol'] = z_df['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            
            st.dataframe(z_df, use_container_width=True, column_config={
                "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)"),
                "date": st.column_config.TextColumn("Date"),
                "score": st.column_config.NumberColumn("Score (Out of 100)", format="%.1f"),
                "confidence_level": st.column_config.TextColumn("Confidence Level"),
                "breakout_status": st.column_config.TextColumn("Breakout Status")
            })
        else:
            st.info("No Dan Zanger setups found.")
    else:
        st.info("💡 Click 'Run Dan Zanger Scan' to find breakouts.")
