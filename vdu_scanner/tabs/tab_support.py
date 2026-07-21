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
        st.markdown("### 🛡️ Support Bounce Scanner — Historical Support + RSI Oversold")
        st.markdown(
            "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-8px; line-height:1.5;'>"
            "Finds stocks sitting <b style='color:#00e676;'>near historical support levels</b> (where price previously bounced) "
            "with <b style='color:#ef4444;'>RSI in oversold territory</b> (≤ 35). "
            "Multiple touches at a support level = stronger floor. "
            "<span style='color:#ffa000; font-weight:600;'>Filters: Price ≥ ₹100 | Market Cap ≥ ₹2000 Cr</span>"
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        today_str = get_market_date(for_display=True)

        # Settings
        sup_col1, sup_col2, sup_col3 = st.columns(3)
        with sup_col1:
            sup_rsi_threshold = st.slider("RSI Threshold (Oversold)", min_value=20.0, max_value=45.0, value=35.0, step=1.0, key="sup_rsi_thresh")
        with sup_col2:
            sup_proximity = st.slider("Max Distance to Support %", min_value=1.0, max_value=8.0, value=3.0, step=0.5, key="sup_proximity")
        with sup_col3:
            sup_index = st.selectbox("Stock Universe", ["NIFTY 500", "ALL NSE"], key="sup_universe")

        # Scan button
        if st.button("🔍 Run Support Bounce Scan", key="run_support_scan", type="primary"):
            from scanner import scan_support_rsi
            from data_fetcher import get_index_stocks, get_all_nse_symbols

            with st.spinner("Scanning for stocks at support with oversold RSI..."):
                if sup_index == "ALL NSE":
                    raw_symbols = get_all_nse_symbols()
                else:
                    raw_symbols = get_index_stocks(sup_index)

                symbols_to_scan = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
                support_results = []
                chunk_size = 50
                chunks = [symbols_to_scan[i:i+chunk_size] for i in range(0, len(symbols_to_scan), chunk_size)]

                progress_bar = st.progress(0.0, text="Starting scan...")
                for c_idx, chunk in enumerate(chunks):
                    progress_bar.progress((c_idx + 1) / len(chunks), text=f"Scanning chunk {c_idx+1}/{len(chunks)}... Found {len(support_results)} matches")
                    try:
                        bulk_df = yf.download(tickers=chunk, period="1y", interval="1d", progress=False, threads=False, timeout=15)
                        if bulk_df is not None and not bulk_df.empty:
                            for sym_ns in chunk:
                                try:
                                    sym = sym_ns.replace('.NS', '')
                                    if isinstance(bulk_df.columns, pd.MultiIndex):
                                        all_tkrs = bulk_df.columns.get_level_values(1).unique().tolist()
                                        matched_t = next((t for t in all_tkrs if t.upper() == sym_ns.upper()), None)
                                        if not matched_t:
                                            continue
                                        df_sym = bulk_df.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                                    else:
                                        if len(chunk) == 1:
                                            df_sym = bulk_df.dropna(subset=['Close'])
                                        else:
                                            continue

                                    if not df_sym.empty and len(df_sym) >= 100:
                                        df_sym = df_sym.reset_index()
                                        df_sym.rename(columns={df_sym.columns[0]: 'Date'}, inplace=True)
                                        res = scan_support_rsi(sym, df_sym, market_cap=0.0,
                                                               rsi_threshold=sup_rsi_threshold,
                                                               support_proximity_pct=sup_proximity)
                                        if res is not None:
                                            support_results.append(res)
                                except Exception:
                                    pass
                    except Exception as chunk_ex:
                        print(f"Support scan chunk {c_idx} error: {chunk_ex}")

                progress_bar.empty()

                # Sort by score descending
                support_results.sort(key=lambda x: x.get('score', 0), reverse=True)

                # Save to database
                try:
                    database.save_support_rsi_only(today_str, support_results)
                    st.toast(f"✅ Saved {len(support_results)} support bounce results to database!", icon="💾")
                except Exception as db_ex:
                    print(f"Failed to save support RSI results: {db_ex}")

                st.session_state.support_rsi_results = support_results
                st.session_state.support_rsi_scan_date = today_str

        # Load from DB cache if not in session
        if 'support_rsi_results' not in st.session_state or not st.session_state.support_rsi_results:
            cached = database.get_cached_support_rsi(today_str)
            if cached:
                st.session_state.support_rsi_results = cached
                st.session_state.support_rsi_scan_date = today_str
            else:
                # Fallback: load the most recent scan from any date
                latest_results, latest_date = database.get_latest_support_rsi()
                if latest_results:
                    st.session_state.support_rsi_results = latest_results
                    st.session_state.support_rsi_scan_date = latest_date

        sup_data = st.session_state.get('support_rsi_results', [])
        sup_scan_date = st.session_state.get('support_rsi_scan_date', today_str)

        # Show info if displaying older results
        if sup_data and sup_scan_date != today_str:
            st.info(f"📅 Showing last scan results from **{sup_scan_date}**. Click '🔍 Run Support Bounce Scan' to refresh with today's data.")

        # Metrics
        sup_m1, sup_m2, sup_m3, sup_m4 = st.columns(4)
        sup_count = len(sup_data) if sup_data else 0
        sup_avg_rsi = (sum(r['rsi'] for r in sup_data) / sup_count) if sup_count > 0 else 0.0
        sup_high_conf = len([r for r in sup_data if r.get('confidence') == 'High']) if sup_data else 0
        sup_max_touches = max(r.get('support_touches', 0) for r in sup_data) if sup_data else 0

        sup_m1.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🛡️ Stocks at Support</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{sup_count}</h3></div>', unsafe_allow_html=True)
        sup_m2.markdown(f'<div class="glass-card metric-glow-red"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">📉 Avg RSI</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ef4444;">{sup_avg_rsi:.1f}</h3></div>', unsafe_allow_html=True)
        sup_m3.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">⭐ High Confidence</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{sup_high_conf}</h3></div>', unsafe_allow_html=True)
        sup_m4.markdown(f'<div class="glass-card metric-glow-purple"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🔁 Max Support Touches</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ce93d8;">{sup_max_touches}</h3></div>', unsafe_allow_html=True)

        if not sup_data:
            st.info(f"No support bounce results found. Click '🔍 Run Support Bounce Scan' above to scan.")

        else:
            # Build table
            rows_html = []
            for idx, r in enumerate(sup_data, 1):
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

                # SMA badges
                sma_badges = []
                if r.get('above_200sma'): sma_badges.append('<span style="background:rgba(0,230,118,0.12); color:#00e676; padding:1px 5px; border-radius:3px; font-size:0.72rem; font-weight:600;">200</span>')
                if r.get('above_50sma'): sma_badges.append('<span style="background:rgba(41,182,246,0.12); color:#29b6f6; padding:1px 5px; border-radius:3px; font-size:0.72rem; font-weight:600;">50</span>')
                if r.get('above_20sma'): sma_badges.append('<span style="background:rgba(206,147,216,0.12); color:#ce93d8; padding:1px 5px; border-radius:3px; font-size:0.72rem; font-weight:600;">20</span>')
                sma_html = " ".join(sma_badges) if sma_badges else '<span style="color:#64748b; font-size:0.72rem;">—</span>'

                conf_badge = f'<span style="background:rgba({("0,230,118" if conf=="High" else "255,160,0" if conf=="Medium" else "239,68,68")},0.12); color:{conf_color}; padding:2px 6px; border-radius:4px; font-size:0.75rem; font-weight:bold; border:1px solid {conf_color};">{conf}</span>'

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
                    f'<td style="padding:10px 12px; color:#ffa000;">{dist:.1f}%</td>'
                    f'<td style="padding:10px 12px;">{conf_badge}</td>'
                    f'<td style="padding:10px 12px; color:#ce93d8; font-weight:600;">{sc:.1f}</td>'
                    f'<td style="padding:10px 12px;">{sma_html}</td>'
                    f'<td style="padding:10px 12px; color:#94a3b8; text-align:right;">{vol:,}</td>'
                    f'</tr>'
                )
                rows_html.append(row)

            table_body = "".join(rows_html)

            st.markdown(
                f'<div class="glass-card" style="padding:18px; margin-bottom:22px; border:1px solid rgba(0,230,118,0.2); background:rgba(9,13,22,0.55); border-radius:12px;">'
                f'<h3 style="margin-top:0; color:#00e676; font-size:1.15rem; display:flex; align-items:center; gap:8px; font-family:Outfit,sans-serif;">'
                f'🛡️ Support Bounce + RSI Oversold — {sup_scan_date}'
                f'</h3>'
                f'<p style="font-size:0.82rem; color:#94a3b8; margin-top:-8px; margin-bottom:15px; font-family:Outfit,sans-serif;">'
                f'Stocks near multi-touch historical support zones with oversold RSI. <b>More touches = stronger support floor.</b> '
                f'Wait for a green candle bounce confirmation before entering.'
                f'</p>'
                f'<div style="overflow-x:auto;">'
                f'<table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.85rem; color:#cbd5e1; font-family:Outfit,sans-serif;">'
                f'<thead>'
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#38bdf8; font-weight:bold; background:rgba(0,230,118,0.04); font-size:0.78rem; text-transform:uppercase;">'
                f'<th style="padding:8px 12px;">#</th>'
                f'<th style="padding:8px 12px;">Symbol</th>'
                f'<th style="padding:8px 12px;">CMP</th>'
                f'<th style="padding:8px 12px;">Change</th>'
                f'<th style="padding:8px 12px; color:#ef4444;">RSI</th>'
                f'<th style="padding:8px 12px; color:#00e676;">Support Level</th>'
                f'<th style="padding:8px 12px; text-align:center;">Touches</th>'
                f'<th style="padding:8px 12px; color:#ffa000;">Distance</th>'
                f'<th style="padding:8px 12px;">Confidence</th>'
                f'<th style="padding:8px 12px; color:#ce93d8;">Score</th>'
                f'<th style="padding:8px 12px;">Above SMA</th>'
                f'<th style="padding:8px 12px; text-align:right;">Volume</th>'
                f'</tr>'
                f'</thead>'
                f'<tbody>{table_body}</tbody>'
                f'</table>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Quick Trade Board
            render_quick_trade_board(sup_data, key_prefix="support_bounce")

            # CSV Download
            df_support = pd.DataFrame([{
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'RSI': round(r['rsi'], 2),
                'CCI': round(r.get('cci', 0.0), 2),
                'Support Level': r['support_price'],
                'Touches': r['support_touches'],
                'Distance %': r['distance_to_support_pct'],
                'Score': r['score'],
                'Confidence': r.get('confidence'),
                'Buy Price': r.get('buy_price'),
                'Exit Price': r.get('exit_price'),
                'Target Price': r.get('target_price'),
                'Above 20 SMA': r.get('above_20sma'),
                'Above 50 SMA': r.get('above_50sma'),
                'Above 200 SMA': r.get('above_200sma'),
                'Volume': r['volume'],
                'Recommendation': r.get('recommendation'),
            } for r in sup_data])

            csv_sup = df_support.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Support Bounce Results (CSV)",
                data=csv_sup,
                file_name=f"Support_RSI_{today_str}.csv",
                mime="text/csv",
                key="dl_support_csv"
            )

            # Explainer
            with st.expander("ℹ️ How Support Bounce Scanner Works", expanded=False):
                st.markdown("""
                **This scanner finds stocks at historical support with oversold momentum:**

                1. **🛡️ Support Detection** — Scans the last ~1 year of daily price data to find **swing lows** (pivot points where price bounced up). Nearby swing lows are clustered into **support zones**. More touches at the same level = stronger support.

                2. **📉 RSI Oversold Filter** — Only shows stocks where RSI(14) is ≤ 35 (oversold territory), meaning selling pressure may be exhausted.

                3. **📏 Proximity Check** — Current price must be within the configured distance % of a support zone (default 3%).

                **Scoring:**
                - **Touches × 15** — More historical bounces = stronger support
                - **RSI depth × 2** — Deeper oversold = more upside potential
                - **Proximity bonus × 5** — Closer to support = better entry

                **How to trade:**
                - **Entry:** Near the support level — wait for a green bounce candle
                - **Stop Loss:** 3% below the support level
                - **Target:** 10-15% upside (mean reversion to fair value)
                - **Higher confidence:** 3+ touches + RSI < 30
                """)

    except Exception as e:
        st.error(f"Error rendering Support Bounce tab: {e}")
