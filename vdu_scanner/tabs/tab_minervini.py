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
    st.markdown("### 🏆 Mark Minervini Stage-2 Trend Template")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Scan for institutional Stage-2 uptrend breakout setups using the legendary Mark Minervini Trend Template. We prioritize <b style=\"color:#00e676;\">Early Stage-2</b> candidates (within 20% of their 200 SMA support) to capture high-velocity breakouts with tight risk protection.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Mode selector for Minervini tab
    min_mode = st.radio(
        "Minervini Session Mode:",
        ["🟢 View Today's Minervini Setups", "📅 Browse Historical Minervini Scans"],
        horizontal=True,
        help="View live setups from today's scanner run, or select any past scan date to view historical Stage-2 setups.",
        key="min_session_mode_selector"
    )
    
    if min_mode == "🟢 View Today's Minervini Setups":
        minervini_data = st.session_state.minervini_results
        
        if minervini_data is None:
            st.info("💡 Run the scanner from the sidebar to identify stocks matching the Mark Minervini Stage-2 Trend Template.")
        elif len(minervini_data) == 0:
            st.info("ℹ️ No stocks found today matching the Minervini Stage-2 Trend Template. Run scans on a broader universe like Nifty 500 or ALL NSE!")
        else:
            # A. Premium Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            
            m_total = len(minervini_data)
            early_list = [r for r in minervini_data if r.get('is_early', True)]
            early_count = len(early_list)
            extended_count = m_total - early_count
            
            avg_run_200 = sum(r['run_up_200'] for r in minervini_data) / m_total
            avg_run_52w = sum(r['run_up_52w'] for r in minervini_data) / m_total
            
            m_col1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Stage-2 Trend Stocks</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{m_total}</h3></div>', unsafe_allow_html=True)
            m_col2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🏆 Early Stage-2 (Safe)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{early_count}</h3></div>', unsafe_allow_html=True)
            m_col3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Run Up (200 SMA)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">+{avg_run_200:.1f}%</h3></div>', unsafe_allow_html=True)
            m_col4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Run Up (52w Low)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">+{avg_run_52w:.1f}%</h3></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Interactive filter
            f_min1, f_min2 = st.columns([1, 2])
            show_early_only = f_min1.checkbox("🏆 Show Early Stage-2 Only (Accumulation Zone)", value=False, key="min_filter_early_only")
            
            display_data = early_list if show_early_only else minervini_data
            
            if len(display_data) == 0:
                st.warning("⚠️ No stocks match the active filters in this template view.")
            else:
                # Sort by remaining target percentage descending
                sorted_minervini = sorted(display_data, key=lambda x: x.get('target_price', 0.0) - x.get('cmp', 0.0), reverse=True)
                
                # Premium CSV download option
                export_min = []
                for r in sorted_minervini:
                    rem_pct = ((r['target_price'] - r['cmp']) / r['cmp'] * 100) if r['cmp'] > 0 else 0.0
                    export_min.append({
                        "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                                "CMP (₹)": r['cmp'],
                        "Day Change %": r['day_change_pct'],
                        "Run Up from 200 SMA %": r['run_up_200'],
                        "Run Up from 52w Low %": r['run_up_52w'],
                        "Stage Category": "Early Stage-2" if r['is_early'] else "Extended Stage-2",
                        "Suggested Buy (₹)": r['buy_price'],
                        "Suggested Stop Loss (₹)": r['exit_price'],
                        "Suggested Target (₹)": r['target_price'],
                        "Remaining Target Potential %": round(rem_pct, 2),
                        "Confidence Rating": r['confidence'],
                        "Actionable Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
                    })
                export_m_df = pd.DataFrame(export_min)
                csv_m_data = export_m_df.to_csv(index=False).encode('utf-8-sig')
                
                st.download_button(
                    label="📥 Download Minervini Template Results (CSV)",
                    data=csv_m_data,
                    file_name=f"minervini_stage2_setups_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="minervini_download_csv_btn"
                )
                
                st.markdown("---")
                st.markdown("### 🏆 Active Mark Minervini Stage-2 Trade Execution Sheet")
                render_unified_strategy_table(sorted_minervini, "minervini", "minervini_tab")
    else:
        # Browse historical scans
        available_dates = database.get_available_scan_dates()
        if not available_dates:
            st.warning("⚠️ No historical scans have been recorded in the database yet. Run the scanner to save today's results first!")
        else:
            h_date = st.selectbox(
                "Select Historical Minervini Scan Date:",
                options=available_dates,
                index=0,
                key="min_hist_date_select",
                help="Choose a date from completed historical scanner sessions."
            )
            
            h_minervini = ensure_minervini_fields(database.get_cached_trend_setups(h_date, 'minervini'))
            if not h_minervini:
                st.info(f"ℹ️ No Minervini Stage-2 trend setups were recorded on {h_date}.")
            else:
                st.markdown(f"### 🏆 Historical Minervini Stage-2 setups on {h_date} ({len(h_minervini)})")
                
                # A. Premium Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                m_total = len(h_minervini)
                early_list = [r for r in h_minervini if r.get('is_early', True)]
                early_count = len(early_list)
                extended_count = m_total - early_count
                
                avg_run_200 = sum(r['run_up_200'] for r in h_minervini) / m_total
                avg_run_52w = sum(r['run_up_52w'] for r in h_minervini) / m_total
                
                m_col1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Stage-2 Trend Stocks</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{m_total}</h3></div>', unsafe_allow_html=True)
                m_col2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🏆 Early Stage-2 (Safe)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{early_count}</h3></div>', unsafe_allow_html=True)
                m_col3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Run Up (200 SMA)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">+{avg_run_200:.1f}%</h3></div>', unsafe_allow_html=True)
                m_col4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Run Up (52w Low)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">+{avg_run_52w:.1f}%</h3></div>', unsafe_allow_html=True)
                
                st.markdown("---")
                
                show_early_only_h = st.checkbox("🏆 Show Early Stage-2 Only (Accumulation Zone)", value=False, key="min_filter_early_only_h")
                display_data_h = early_list if show_early_only_h else h_minervini
                
                if len(display_data_h) == 0:
                    st.warning("⚠️ No stocks match the active filters in this historical view.")
                else:
                    sorted_minervini_h = sorted(display_data_h, key=lambda x: x.get('target_price', 0.0) - x.get('cmp', 0.0), reverse=True)
                    
                    # Premium CSV download option
                    export_min_h = []
                    for r in sorted_minervini_h:
                        rem_pct = ((r['target_price'] - r['cmp']) / r['cmp'] * 100) if r['cmp'] > 0 else 0.0
                        export_min_h.append({
                            "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                                        "CMP (₹)": r['cmp'],
                            "Day Change %": r['day_change_pct'],
                            "Run Up from 200 SMA %": r['run_up_200'],
                            "Run Up from 52w Low %": r['run_up_52w'],
                            "Stage Category": "Early Stage-2" if r['is_early'] else "Extended Stage-2",
                            "Suggested Buy (₹)": r['buy_price'],
                            "Suggested Stop Loss (₹)": r['exit_price'],
                            "Suggested Target (₹)": r['target_price'],
                            "Remaining Target Potential %": round(rem_pct, 2),
                            "Confidence Rating": r['confidence'],
                            "Actionable Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
                        })
                    export_m_df_h = pd.DataFrame(export_min_h)
                    csv_m_data_h = export_m_df_h.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label="📥 Download Historical Minervini Template Results (CSV)",
                        data=csv_m_data_h,
                        file_name=f"minervini_stage2_setups_hist_{h_date}.csv",
                        mime="text/csv",
                        key="minervini_download_csv_btn_h"
                    )
                    
                    st.markdown("---")
                    st.markdown(f"### 🏆 Historical Mark Minervini Stage-2 Trade Execution Sheet ({h_date})")
                    render_unified_strategy_table(sorted_minervini_h, "minervini", f"minervini_tab_hist_{h_date}")
                    
    st.markdown("<br>", unsafe_allow_html=True)
    # Trend Template Rules explanation
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 How Mark Minervini Stage-2 Trend Template Works"):
        st.markdown(r"""
        **Mark Minervini Stage-2 Trend Template Rules**
        
        To qualify as an institutional Stage-2 stock, an asset must meet all of the following rules:
        
        1. **Price is Above 150 EMA and 200 SMA:** Confirms a structural long-term uptrend.
        2. **150-day EMA is Above the 200-day SMA:** Confirms standard momentum alignment.
        - **Early Stage-2 Accumulation:** Stocks trading $\le 20\%$ above their rising 200 SMA are in standard buying zones. They offer maximum upside potential with high-probability breakout rates.
        - **Extended / Overbought:** Stocks trading $> 20\%$ above the 200 SMA are mathematically overextended. They are prone to mean-reversion pullbacks and carry a high failure rate for new breakouts.
        3. **200-day SMA is Rising:** Confirms the institutional floor is actively tilting upwards.
        4. **50-day SMA is Above 150 EMA and 200 SMA:** Short-term momentum is supportive of rapid moves.
        5. **Current Price is Above the 50-day SMA:** Confirms standard breakouts are in active trading.
        6. **Price is at least 30% Above its 52-Week Low:** Confirms a durable turnaround and trend reversal.
        7. **Price is Within 25% of its 52-Week High:** Confirms standard strength and dynamic demand.
        
        **Strategy & Risk Management:**
        - **Early Stage-2 Accumulation:** Stocks trading $\le 20\%$ above their rising 200 SMA are in standard buying zones. They offer maximum upside potential with high-probability breakout rates.
        - **Stop Loss:** Set tightly underneath the 200 SMA support floor to keep risk below 4–5%.
        - **Swing Target:** Projected standard target is the 52-week high or +25% momentum swing target, prioritizing early entrants with large remaining potential.
        """, unsafe_allow_html=False)
