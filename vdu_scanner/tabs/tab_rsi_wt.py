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
        st.markdown("### 🎯 RSI Oversold Scanner")
        st.markdown(
            "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-8px; line-height:1.5;'>"
            "Finds stocks where <b style='color:#ef4444;'>RSI is oversold</b> — "
            "the strongest mean-reversion candidates. "
            "Data is fetched from <b style='color:#29b6f6;'>existing database scans</b>. "
            "<span style='color:#ffa000; font-weight:600;'>Requires Support Bounce scan to be run on the same day.</span>"
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        rw_today_str = get_market_date(for_display=True)

        # Settings
        rw_col1, _ = st.columns(2)
        with rw_col1:
            rw_rsi_thresh = st.slider("RSI Threshold (Oversold)", min_value=20.0, max_value=45.0, value=35.0, step=1.0, key="rw_rsi_thresh_v2")

        # Load from DB cache instantly
        st.session_state.rsi_oversold_results = database.get_rsi_oversold(rw_today_str, rsi_threshold=rw_rsi_thresh)
        rw_data = st.session_state.get('rsi_oversold_results', [])

        # Metrics
        rw_m1, rw_m2, rw_m3 = st.columns(3)
        rw_count = len(rw_data) if rw_data else 0
        rw_avg_rsi = (sum(r['rsi'] for r in rw_data) / rw_count) if rw_count > 0 else 0.0
        rw_high_conf = len([r for r in rw_data if r.get('confidence') == 'High']) if rw_data else 0

        rw_m1.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🎯 Oversold Setups</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{rw_count}</h3></div>', unsafe_allow_html=True)
        rw_m2.markdown(f'<div class="glass-card metric-glow-red"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">📉 Avg RSI</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ef4444;">{rw_avg_rsi:.1f}</h3></div>', unsafe_allow_html=True)
        rw_m3.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🔥 High Confidence</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{rw_high_conf}</h3></div>', unsafe_allow_html=True)

        if not rw_data:
            st.info("No RSI Oversold results found. Run **🛡️ Support** scan first to populate data, or increase the RSI threshold.")
        else:
            # Build table
            rw_rows_html = []
            for idx, r in enumerate(rw_data, 1):
                sym = r.get('symbol', '')
                cmp = r.get('cmp', 0.0)
                chg = r.get('day_change_pct', 0.0)
                rsi = r.get('rsi', 0.0)
                sup_price = r.get('support_price', 0.0)
                touches = r.get('support_touches', 0)
                dist = r.get('distance_to_support_pct', 0.0)
                sc = r.get('score', 0.0)
                vol = r.get('volume', 0)
                conf = r.get('confidence', 'Low')

                # Color coding
                chg_color = "#00e676" if chg >= 0 else "#ef4444"
                chg_sign = "+" if chg >= 0 else ""
                rsi_color = "#ef4444" if rsi <= 30 else "#ffa000" if rsi <= 35 else "#94a3b8"
                conf_color = "#00e676" if conf == "High" else "#ffa000" if conf == "Medium" else "#ef4444"
                touch_color = "#00e676" if touches >= 3 else "#ffa000" if touches >= 2 else "#94a3b8"

                conf_badge = f'<span style="background:rgba({("0,230,118" if conf=="High" else "255,160,0" if conf=="Medium" else "239,68,68")},0.12); color:{conf_color}; padding:2px 6px; border-radius:4px; font-size:0.75rem; font-weight:bold; border:1px solid {conf_color};">{conf}</span>'

                # Score color
                sc_color = "#00e676" if sc >= 60 else "#ffa000" if sc >= 35 else "#ce93d8"

                row = (
                    f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">'
                    f'<td style="padding:10px 12px; color:#64748b;">{idx}</td>'
                    f'<td style="padding:10px 12px; font-weight:bold;">'
                    f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" style="color:#29b6f6; text-decoration:none;">{sym}</a>'
                    f'</td>'
                    f'<td style="padding:10px 12px; color:#e2e8f0;">₹{cmp:,.2f}</td>'
                    f'<td style="padding:10px 12px; color:{chg_color}; font-weight:600;">{chg_sign}{chg:.2f}%</td>'
                    f'<td style="padding:10px 12px; color:{rsi_color}; font-weight:bold;">{rsi:.1f}</td>'
                    f'<td style="padding:10px 12px; color:#00e676; font-weight:600;">₹{sup_price:,.2f}</td>'
                    f'<td style="padding:10px 12px; color:{touch_color}; font-weight:bold; text-align:center;">{touches}</td>'
                    f'<td style="padding:10px 12px;">{conf_badge}</td>'
                    f'<td style="padding:10px 12px; color:{sc_color}; font-weight:600;">{sc:.1f}</td>'
                    f'<td style="padding:10px 12px; color:#94a3b8; text-align:right;">{vol:,}</td>'
                    f'</tr>'
                )
                rw_rows_html.append(row)

            rw_table_body = "".join(rw_rows_html)

            st.markdown(
                f'<div class="glass-card" style="padding:18px; margin-bottom:22px; border:1px solid rgba(206,147,216,0.3); background:rgba(9,13,22,0.55); border-radius:12px;">'
                f'<h3 style="margin-top:0; color:#ce93d8; font-size:1.15rem; display:flex; align-items:center; gap:8px; font-family:Outfit,sans-serif;">'
                f'🎯 RSI Oversold Setups — {rw_today_str}'
                f'</h3>'
                f'<p style="font-size:0.82rem; color:#94a3b8; margin-top:-8px; margin-bottom:15px; font-family:Outfit,sans-serif;">'
                f'Stocks with RSI in oversold zone. '
                f'<b style="color:#00e676;">Higher score = deeper oversold + stronger historical support floor.</b> '
                f'Wait for a green candle confirmation before entering.'
                f'</p>'
                f'<div style="overflow-x:auto;">'
                f'<table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.85rem; color:#cbd5e1; font-family:Outfit,sans-serif;">'
                f'<thead>'
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#38bdf8; font-weight:bold; background:rgba(206,147,216,0.06); font-size:0.78rem; text-transform:uppercase;">'
                f'<th style="padding:8px 12px;">#</th>'
                f'<th style="padding:8px 12px;">Symbol</th>'
                f'<th style="padding:8px 12px;">CMP</th>'
                f'<th style="padding:8px 12px;">Change</th>'
                f'<th style="padding:8px 12px; color:#ef4444;">RSI</th>'
                f'<th style="padding:8px 12px; color:#00e676;">Support</th>'
                f'<th style="padding:8px 12px; text-align:center;">Touches</th>'
                f'<th style="padding:8px 12px;">Confidence</th>'
                f'<th style="padding:8px 12px; color:#ce93d8;">Score</th>'
                f'<th style="padding:8px 12px; text-align:right;">Volume</th>'
                f'</tr>'
                f'</thead>'
                f'<tbody>{rw_table_body}</tbody>'
                f'</table>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Quick Trade Board
            render_quick_trade_board(rw_data, key_prefix="rsi_oversold")

            # CSV Download
            df_rw = pd.DataFrame([{
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'RSI': round(r['rsi'], 2),
                'Support Level': r.get('support_price', 0),
                'Touches': r.get('support_touches', 0),
                'Distance %': r.get('distance_to_support_pct', 0),
                'Score': r.get('score', 0),
                'Confidence': r.get('confidence', ''),
                'Volume': r.get('volume', 0),
            } for r in rw_data])

            csv_rw = df_rw.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download RSI Oversold Results (CSV)",
                data=csv_rw,
                file_name=f"RSI_Oversold_{rw_today_str}.csv",
                mime="text/csv",
                key="dl_rsi_csv_v2"
            )

            # Explainer
            with st.expander("ℹ️ How RSI Oversold Scanner Works", expanded=False):
                st.markdown("""
                **This scanner finds high-conviction buy setups based on extreme RSI levels:**

                1. **📉 RSI Oversold** — RSI(14) ≤ 35 indicates selling pressure exhaustion. 
                2. **🔗 Database Retrieval** — Fetches directly from the Support Bounce scan. This is a database-only operation, so it's instantly updated when you move the slider.

                **Scoring Formula:**
                - **RSI depth**: `(35 - RSI) × 2` — deeper oversold = higher score
                - **Support touches**: `touches × 10` — more historical bounces = stronger floor

                **Confidence Levels:**
                - **🟢 High**: RSI ≤ 30 + Support touches ≥ 3
                - **🟡 Medium**: RSI ≤ chosen threshold
                - **🔴 Low**: Weak support touches or shallow oversold


                **How to trade:**
                - **Entry:** Wait for a green candle bounce confirmation
                - **Stop Loss:** 3-5% below the support level
                - **Target:** 15-25% upside (double oversold = stronger mean reversion)
                - **Best setups:** High confidence + Score > 50
                """)

    except Exception as e:
        st.error(f"Error rendering RSI+Wave tab: {e}")
