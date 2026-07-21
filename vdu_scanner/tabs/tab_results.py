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
    try:
        # 1. Premium Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        if scan_data:
            total_scanned = st.session_state.total_scanned
            # If total_scanned is 0 or less than actual breakout count, use breakout count as floor
            # (scan_logs may not have the entry for this date)
            if total_scanned < len(scan_data):
                total_scanned = len(scan_data)
            flagged_count = len(scan_data)
            top_score = max(r['signal_strength'] for r in scan_data)
            avg_vol_ratio = sum(r['volume_ratio'] for r in scan_data) / flagged_count
        else:
            total_scanned = st.session_state.total_scanned or 0
            flagged_count = 0
            top_score = 0.0
            avg_vol_ratio = 0.0
            
        m1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Total Stocks Scanned</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{total_scanned}</h3></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Breakouts Identified</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{flagged_count}</h3></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Highest Signal Score</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{top_score:.1f} <span style="font-size: 1.1rem; color: #94a3b8;">pts</span></h3></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Volume Ratio</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{avg_vol_ratio:.2f}x</h3></div>', unsafe_allow_html=True)
        # Show which date the data is from (critical transparency for cached results)
        _scan_date_display = st.session_state.get('scan_results_date') or st.session_state.get('last_scanned', '')
        if _scan_date_display and scan_data:
            st.info(f"📅 **Showing scan results from: {_scan_date_display}** — Refresh auto-loaded these from the database. Click **'Run Scanner'** in the sidebar to get today's fresh results.")

        # 2. Main Scan Table
        # NOTE: Use len(scan_data) not total_scanned==0 because DB-loaded results
        # have scan_data populated but total_scanned may be 0 from scan_logs.
        if scan_data is None and st.session_state.get('total_scanned', 0) == 0:
            st.info("💡 Get started by configuring your universe in the sidebar and clicking '**Run Scanner**'.")
        elif not scan_data or len(scan_data) == 0:
            st.info("ℹ️ No VDU breakouts found today matching these criteria. Try lowering the thresholds in the sidebar (e.g. Min Volume Ratio or Min Price Change) and re-running.")
        else:
            # Sort results descending by score
            sorted_scan = sorted(scan_data, key=lambda x: x['signal_strength'], reverse=True)
            
            # Download Results Option - safely convert date fields
            def _safe_date(v):
                if v is None:
                    return ""
                try:
                    if pd.isnull(v):
                        return ""
                except (TypeError, ValueError):
                    pass
                if hasattr(v, 'strftime'):
                    return v.strftime("%Y-%m-%d")
                return str(v)

            export_rows = []
            for r in sorted_scan:
                export_rows.append({
                    "Symbol": r['symbol'],
                    "Sector": get_stock_sector(r['symbol']),
                    "CMP (₹)": r['cmp'],
                    "Setup": r.get('setup_type', 'VDU Breakout'),
                    "Day Change %": r.get('day_change_pct', 0.0),
                    "Today Volume": r.get('today_volume', 0),
                    "Dry Avg Volume": r.get('dry_avg_vol', 0),
                    "Volume Ratio": r.get('volume_ratio', 0.0),
                    "Dry Days": r.get('dry_days_count', 0),
                    "Dry Spikes": r.get('dry_spikes', 0),
                    "Market Cap (Cr)": round(r.get('market_cap_cr', 3000.0), 1),
                    "Signal Strength": r.get('signal_strength', 0.0),
                    "Above 50 DMA": r.get('above_50dma', False),
                    "Above 200 DMA": r.get('above_200dma', False),
                    "Dry Start Date": _safe_date(r.get('dry_start_date')),
                    "Dry End Date": _safe_date(r.get('dry_end_date')),
                    "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
                })
            export_df = pd.DataFrame(export_rows)
            csv_data = export_df.to_csv(index=False).encode('utf-8-sig')
            
            st.download_button(
                label="📥 Download Scan Results (CSV)",
                data=csv_data,
                file_name=f"vdu_scan_results_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_scan_top_btn"
            )
            
            st.markdown("---")
            # Render the unified Trade Execution Matrix
            st.markdown("### 📊 Active VDU Breakout Trade Execution Sheet")
            render_unified_strategy_table(sorted_scan, "vdu_breakout", "vdu_tab")
    except Exception as _tab1_err:
        st.error(f"❌ Error rendering scan results: {_tab1_err}")
        st.exception(_tab1_err)
