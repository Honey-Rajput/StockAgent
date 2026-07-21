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
    st.markdown("### 📅 Historical Scan Database")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Browse the archive of all historical stock scans saved in Neon PostgreSQL. Retrieve and analyze past breakouts, pullbacks, and mean-reversion trade setups.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load all unique dates from PostgreSQL database
    available_dates = database.get_available_scan_dates()
    
    if not available_dates:
        st.warning("⚠️ No historical scans have been recorded in the database yet. Run the scanner to save today's results first!")
    else:
        # Date selection column
        date_sel_col1, date_sel_col2 = st.columns([3, 5])
        selected_date_str = date_sel_col1.selectbox(
            "Select Historical Scan Session Date:",
            options=available_dates,
            index=0,
            key="history_date_select",
            help="Choose a date from completed historical scanner sessions."
        )
        
        # Display logs summary for the chosen day
        day_log = database.has_scanned_today(selected_date_str)
        if day_log:
            date_sel_col2.markdown(
                f"""
                <div class="glass-card" style="padding: 10px 18px; display: inline-block; background: rgba(41, 182, 246, 0.05); border: 1px solid rgba(41, 182, 246, 0.15);">
                    <span style="font-size: 0.82rem; color: #94a3b8; font-weight:600; text-transform: uppercase;">Session Log Summary</span>
                    <p style="margin: 4px 0 0 0; font-size: 0.95rem; color: #e2e8f0;">
                        <b>Total Scanned:</b> {day_log.get('total_scanned', 'N/A')} stocks | 
                        <b>VDU Breakouts:</b> <span style="color:#00e676; font-weight:600;">{day_log.get('breakouts_found', 0)}</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Nested sub-tabs inside History tab
        sub_vcp_minervini, sub_breakout, sub_gapup, sub_above_ma, sub_support_ma, sub_crossover_ma, sub_wt, sub_vp = st.tabs([
            "🏆 VCP+Minervini",
            "📊 VDU Breakouts",
            "🚀 Gap-Ups",
            "📈 Above 20 & 50 SMA",
            "🛡️ 65 SMA Support",
            "🔄 MA Crossovers",
            "🌊 Wave Trend",
            "📊 Volume Profile"
        ])
        
        with sub_vcp_minervini:
            h_vcp_minervini = database.get_cached_vcp_minervini(selected_date_str)
            if not h_vcp_minervini:
                st.info(f"ℹ️ No VCP+Minervini setups were recorded on {selected_date_str}.")
            else:
                st.markdown(f"**🏆 VCP+Minervini on {selected_date_str} ({len(h_vcp_minervini)})**")
                df_vcp = pd.DataFrame(h_vcp_minervini)
                if not df_vcp.empty:
                    df_vcp = df_vcp.sort_values(by="rs_rating", ascending=False)
                    st.dataframe(df_vcp.style.format({"rs_rating": "{:.1f}", "vcp_range_pct": "{:.1f}%", "vcp10_range_pct": "{:.1f}%", "vcp15_range_pct": "{:.1f}%"}), use_container_width=True, hide_index=True)
        
        # 1. Historical Breakouts
        with sub_breakout:
            h_breakouts = database.get_cached_breakouts(selected_date_str)
            if not h_breakouts:
                st.info(f"ℹ️ No VDU Breakouts were recorded on {selected_date_str}.")
            else:
                sorted_hb = sorted(h_breakouts, key=lambda x: x.get('signal_strength', 0.0), reverse=True)
                st.markdown(f"**📊 VDU Breakouts on {selected_date_str} ({len(sorted_hb)})**")
                render_unified_strategy_table(sorted_hb, "vdu_breakout", f"hist_bo_{selected_date_str}")
                    
                    
        # 2. Historical Gap-Ups
        with sub_gapup:
            h_gapups = database.get_cached_gapups(selected_date_str)
            if not h_gapups:
                st.info(f"ℹ️ No Gap-Ups were recorded on {selected_date_str}.")
            else:
                sorted_hgu = sorted(h_gapups, key=lambda x: x.get('gap_pct', 0.0), reverse=True)
                st.markdown(f"**🚀 Gap-Up Setups on {selected_date_str} ({len(sorted_hgu)})**")
                render_unified_strategy_table(sorted_hgu, "gapup", f"hist_gu_{selected_date_str}")
                    
        # 4. Historical Above 20 & 50 SMA
        with sub_above_ma:
            h_above_ma = database.get_cached_trend_setups(selected_date_str, 'above_ma')
            if not h_above_ma:
                st.info(f"ℹ️ No Above SMA trend setups were recorded on {selected_date_str}.")
            else:
                sorted_ham = sorted(h_above_ma, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
                st.markdown(f"**📈 Above 20 & 50 SMA on {selected_date_str} ({len(sorted_ham)})**")
                render_unified_strategy_table(sorted_ham, "above_ma", f"hist_above_{selected_date_str}")
                    
        # 5. Historical 65 SMA Support
        with sub_support_ma:
            h_support_ma = database.get_cached_trend_setups(selected_date_str, 'support_ma')
            if not h_support_ma:
                st.info(f"ℹ️ No 65 SMA Pullback setups were recorded on {selected_date_str}.")
            else:
                sorted_hsm = sorted(h_support_ma, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
                st.markdown(f"**🛡️ 65 SMA Support Pullbacks on {selected_date_str} ({len(sorted_hsm)})**")
                render_unified_strategy_table(sorted_hsm, "support_ma", f"hist_support_{selected_date_str}")
                    
        # 6. Historical MA Crossovers
        with sub_crossover_ma:
            h_crossovers = database.get_cached_trend_setups(selected_date_str, 'crossover_ma')
            if not h_crossovers:
                st.info(f"ℹ️ No MA Crossover breakouts were recorded on {selected_date_str}.")
            else:
                sorted_hco = sorted(h_crossovers, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
                st.markdown(f"**🔄 MA Crossovers on {selected_date_str} ({len(sorted_hco)})**")
                render_unified_strategy_table(sorted_hco, "crossover_ma", f"hist_cross_{selected_date_str}")
                    
        # 7. Historical WaveTrend
        with sub_wt:
            h_wt = database.get_cached_wt_cross(selected_date_str)
            if not h_wt:
                st.info(f"ℹ️ No WaveTrend oversold buy signals were recorded on {selected_date_str}.")
            else:
                sorted_hwt = sorted(h_wt, key=lambda x: float(x.get('wt_value') or 0.0))
                st.markdown(f"**🌊 WaveTrend Signals on {selected_date_str} ({len(sorted_hwt)})**")
                render_unified_strategy_table(sorted_hwt, "wavetrend", f"hist_wt_{selected_date_str}")
