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
    st.markdown("### 🤖 Technical Chart Pattern Recognition with AI")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Inspect daily candle charts with Euri / Groq AI technical analysts and save/cache findings in Neon PostgreSQL database.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Fetch available symbols for analyzer
    w_db = watchlist.load_watchlist()
    available_tickers = []
    if not w_db.empty:
        available_tickers.extend(list(w_db['symbol'].unique()))
    if st.session_state.scan_results:
        available_tickers.extend([r['symbol'] for r in st.session_state.scan_results])
    
    # Unique sorted values
    available_tickers = list(set([s.upper() for s in available_tickers]))
    available_tickers.sort()

    col_s1, col_s2 = st.columns([3, 1])
    
    # Initialize selector defaults from session state if set by the dashboard load click
    options_list = [""] + available_tickers + ["Custom Ticker (Type Manual)"]
    if st.session_state.ai_selected_stock not in options_list:
        if st.session_state.ai_selected_stock:
            st.session_state.ai_custom_sym_input = st.session_state.ai_selected_stock
            st.session_state.ai_selected_stock = "Custom Ticker (Type Manual)"
        else:
            st.session_state.ai_selected_stock = ""
            
    ai_selection = col_s1.selectbox(
        "Select Stock to Analyze:",
        options=options_list,
        key="ai_selected_stock"
    )

    custom_ai_sym = ""
    if ai_selection == "Custom Ticker (Type Manual)":
        default_val = st.session_state.get("ai_custom_sym_input", "")
        custom_ai_sym = col_s2.text_input(
            "Enter Ticker Name (e.g. INFIBEAM):", 
            value=default_val,
            key="ai_custom_sym_input"
        ).strip().upper()

    ticker_to_analyze = custom_ai_sym if ai_selection == "Custom Ticker (Type Manual)" else ai_selection

    if ticker_to_analyze:
        st.markdown(f"#### 🔍 Ready to Analyze: **{ticker_to_analyze}**")
        
        # Action button to trigger scan
        btn_analyze = st.button("🤖 Analyze Pattern with AI", key="run_ai_analysis_btn")
        
        # Get today's date in IST
        today_date_str = get_market_date()
        display_date_str = get_market_date(for_display=True)
        
        # Check cache first (always check cache automatically to show today's output immediately!)
        cached_result = database.get_pattern_by_date(ticker_to_analyze, display_date_str)
        
        if cached_result or btn_analyze:
            # We either load from cache or run live!
            analysis_dict = None
            loaded_from_db = False
            
            if cached_result:
                analysis_dict = cached_result
                loaded_from_db = True
            elif btn_analyze:
                # Run live scan
                with st.spinner(f"Downloading historical data & querying AI Technical Analyst for {ticker_to_analyze}..."):
                    df_historical = fetch_ohlcv(ticker_to_analyze)
                    if df_historical is None or df_historical.empty:
                        st.error(f"❌ Failed to download historical data for {ticker_to_analyze} via yfinance.")
                    else:
                        analysis_dict = ai_detector.detect_chart_pattern(ticker_to_analyze, df_historical)
                        
                        if analysis_dict and analysis_dict.get("pattern_name") != "Error":
                            analysis_dict['analyzed_date'] = today_date_str
                            # Create small snapshot string of last 5 days close prices
                            subset_5d = df_historical.iloc[-5:]
                            snap_list = [f"{row['Date'].strftime('%m-%d')}:{row['Close']:.0f}" for _, row in subset_5d.iterrows()]
                            snap_str = ",".join(snap_list)
                            
                            # Cache in Postgres Neon db
                            database.save_pattern(
                                symbol=ticker_to_analyze,
                                pattern_name=analysis_dict['pattern_name'],
                                confidence=analysis_dict['confidence'],
                                direction=analysis_dict['direction'],
                                analysis_text=analysis_dict['analysis_text'],
                                price_data_snapshot=snap_str,
                                date_str=today_date_str
                            )
                            st.toast(f"💾 Analysis cached in Neon PostgreSQL for today!", icon="✅")
            
            if analysis_dict:
                if analysis_dict.get("pattern_name") == "Error":
                    st.error(f"❌ Analysis failed: {analysis_dict['analysis_text']}")
                else:
                    # Retrieve df_historical if not already loaded (e.g. on Cache Hit)
                    if 'df_historical' not in locals() or df_historical is None or df_historical.empty:
                        df_historical = fetch_ohlcv(ticker_to_analyze)
                        
                    # Run mathematical pattern scanner locally to display the "Mathematical Charting Proof"
                    from ai_detector import run_algorithmic_pattern_scan
                    algo_res = run_algorithmic_pattern_scan(df_historical)
                    algo_pat = algo_res["pattern"]
                    algo_det = algo_res["details"]
                    
                    # Display results beautifully
                    if loaded_from_db:
                        st.markdown("<p style='color: #00e676; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px;'>⚡ Cache Hit: Loaded instantly from PostgreSQL Database (Neon)</p>", unsafe_allow_html=True)
                    else:
                        model_name = analysis_dict.get('model_used', 'gpt-4.1-mini (Euri)')
                        st.markdown(f"<p style='color: #29b6f6; font-size: 0.85rem; font-weight: 600; margin-bottom: 15px;'>🤖 Live Analysis: Computed via {model_name} Technical Analyst</p>", unsafe_allow_html=True)
                    
                    # Columns for pattern metrics
                    c_det1, c_det2 = st.columns([1, 2])
                    
                    with c_det1:
                        # Color coding direction
                        d_val = analysis_dict['direction'].strip().capitalize()
                        if d_val == "Bullish":
                            dir_badge_html = '<span class="custom-badge badge-green">▲ Bullish</span>'
                        elif d_val == "Bearish":
                            dir_badge_html = '<span class="custom-badge badge-red">▼ Bearish</span>'
                        else:
                            dir_badge_html = '<span class="custom-badge badge-blue">■ Neutral</span>'
                            
                        # Color coding confidence
                        c_val = analysis_dict['confidence'].strip().capitalize()
                        if c_val == "High":
                            conf_badge_html = '<span class="custom-badge badge-amber">★ High Confidence</span>'
                        elif c_val == "Medium":
                            conf_badge_html = '<span class="custom-badge badge-blue">☆ Medium Confidence</span>'
                        else:
                            conf_badge_html = '<span class="custom-badge badge-grey">☆ Low/None</span>'
                            
                        st.markdown(f"""
                        <div class="glass-card">
                            <h4 style="margin-top:0; color:#29b6f6;">AI Assessment</h4>
                            <div style="margin: 14px 0;"><span style="color:#94a3b8; font-size:0.85rem;">Pattern Detected:</span><br><b style="font-size:1.25rem; color:#ffa000;">{analysis_dict['pattern_name']}</b></div>
                            <div style="margin: 14px 0;"><span style="color:#94a3b8; font-size:0.85rem;">Market Direction:</span><br>{dir_badge_html}</div>
                            <div style="margin: 14px 0;"><span style="color:#94a3b8; font-size:0.85rem;">Model Confidence:</span><br>{conf_badge_html}</div>
                            <div style="margin: 10px 0; font-size: 0.85rem; color:#64748b;">Scan Date: {analysis_dict['analyzed_date']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Render the local Mathematical verified pattern scan card under c_det1!
                        if algo_pat != "None":
                            border_style = "border: 1px solid rgba(0, 230, 118, 0.25);"
                            bg_style = "background: rgba(0, 230, 118, 0.04);"
                            verified_badge = '<span class="custom-badge badge-green" style="font-size:0.75rem; border-radius:4px; font-weight:bold; background:rgba(0,230,118,0.1); border:1px solid rgba(0,230,118,0.3); color:#00e676;">✓ Mathematically Verified</span>'
                        else:
                            border_style = "border: 1px solid rgba(255,255,255,0.05);"
                            bg_style = "background: rgba(30, 41, 59, 0.2);"
                            verified_badge = '<span class="custom-badge badge-grey" style="font-size:0.75rem; border-radius:4px; font-weight:bold; background:rgba(148,163,184,0.1); color:#94a3b8;">■ Consolidation / No Match</span>'
                            
                        st.markdown(f"""
                        <div class="glass-card" style="margin-top:12px; {border_style} {bg_style}">
                            <h4 style="margin-top:0; color:#00e676;">🎯 Mathematical Pattern Proof</h4>
                            <div style="margin: 8px 0;"><span style="color:#94a3b8; font-size:0.8rem;">Pattern Scan:</span><br><b style="font-size:1.15rem; color:#ffa000;">{algo_pat}</b></div>
                            <div style="margin: 8px 0;">{verified_badge}</div>
                            <p style="margin: 8px 0 0 0; font-size:0.82rem; color:#cbd5e1; line-height:1.4; font-style:italic;">{algo_det}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with c_det2:
                        st.markdown(f"""
                        <div class="glass-card" style="height: 100%;">
                            <h4 style="margin-top:0; color:#ffa000;">Technical Analyst Remarks</h4>
                            <p style="font-size: 1.05rem; line-height: 1.6; color: #e2e8f0; margin-top: 15px;">
                                "{analysis_dict['analysis_text']}"
                            </p>
                            <br>
                            <div style="padding: 10px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.04); font-size:0.85rem; color:#94a3b8;">
                                💡 <b>Technical Tip:</b> Technical patterns provide high-probability outcomes when aligned with volume. Always verify breakout levels before initiating trades.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Fetch indicators locally and render the unified Technical Indicators Dashboard & checklist!
                    if df_historical is not None and not df_historical.empty:
                        rich_payload = compute_rich_analysis(df_historical, ticker_to_analyze, "AI Chart Pattern Analysis", "The chart pattern aligns with underlying volume momentum.")
                        cmp_val = float(df_historical['Close'].iloc[-1])
                        dummy_ai = {
                            "symbol": ticker_to_analyze,
                            "cmp": cmp_val,
                            "buy_price": cmp_val,
                            "exit_price": cmp_val * 0.93,
                            "target_price": cmp_val * 1.15,
                            "confidence": analysis_dict['confidence'],
                            "recommendation": rich_payload
                        }
                        render_trading_setup_card(dummy_ai, "ai_tab_card", 0)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Candlestick chart for the last 30 trading days
                    # Load historical data for plotting
                    if df_historical is not None and not df_historical.empty:
                        df_chart_30d = df_historical.iloc[-30:].copy()
                        
                        fig_ai = go.Figure(
                            data=[
                                go.Candlestick(
                                    x=df_chart_30d['Date'],
                                    open=df_chart_30d['Open'],
                                    high=df_chart_30d['High'],
                                    low=df_chart_30d['Low'],
                                    close=df_chart_30d['Close'],
                                    increasing_line_color="#00e676",
                                    decreasing_line_color="#ef4444",
                                    name="Price"
                                )
                            ]
                        )
                        fig_ai.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="#090d16",
                            paper_bgcolor="#090d16",
                            margin=dict(l=30, r=30, t=30, b=30),
                            xaxis=dict(
                                rangeslider=dict(visible=False),
                                gridcolor="rgba(255,255,255,0.04)",
                                rangebreaks=[dict(bounds=["sat", "mon"])]
                            ),
                            yaxis=dict(
                                gridcolor="rgba(255,255,255,0.04)",
                                title="Price (₹)"
                            ),
                            font=dict(family="Outfit, sans-serif"),
                            height=350,
                            title={
                                'text': f"🔍 Last 30 Trading Days Price History for {ticker_to_analyze}",
                                'font': {'size': 14, 'family': 'Outfit', 'color': '#29b6f6'}
                            }
                        )
                        st.plotly_chart(fig_ai, width="stretch")

    # ==========================================================================
    # BATCH AI DASHBOARD FOR FLAGGED STOCKS
    # ==========================================================================
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Scanned Breakouts & Squeezes AI Pattern Dashboard")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-10px;'>Batch-analyze classical chart patterns recognized by AI for all breakout and contraction setups flagged in today's scans.</p>", unsafe_allow_html=True)
    
    # Collate active flagged stocks from scanner results
    active_flagged_symbols = []
    symbol_origins = {}
    
    if st.session_state.scan_results:
        for r in st.session_state.scan_results:
            sym = r['symbol'].upper()
            active_flagged_symbols.append(sym)
            symbol_origins[sym] = "📊 Breakout"
            
            sym = r['symbol'].upper()
            if sym not in symbol_origins:
                active_flagged_symbols.append(sym)
                
    active_flagged_symbols = list(set(active_flagged_symbols))
    active_flagged_symbols.sort()
    
    if not active_flagged_symbols:
        st.info("💡 Run a market scan first from the sidebar to find breakout or contraction setups and dynamically batch-analyze them with AI here!")
    else:
        # Load cached patterns from database for all active flagged symbols
        today_str = get_market_date(for_display=True)
        
        flagged_db_records = {}
        all_today_patterns = database.get_all_patterns_by_date(today_str)
        for s in active_flagged_symbols:
            rec = all_today_patterns.get(s)
            if rec:
                flagged_db_records[s] = rec
        # Count stats
        scanned_count = len(flagged_db_records)
        unscanned_count = len(active_flagged_symbols) - scanned_count
        
        # Display small dashboard summary
        d_c1, d_c2, d_c3 = st.columns(3)
        d_c1.markdown(f'<div class="glass-card"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Flagged Candidates</p><h3 style="font-size:1.6rem; margin:5px 0 0 0; color:#29b6f6;">{len(active_flagged_symbols)}</h3></div>', unsafe_allow_html=True)
        d_c2.markdown(f'<div class="glass-card"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">AI Analyzed Today</p><h3 style="font-size:1.6rem; margin:5px 0 0 0; color:#00e676;">{scanned_count}</h3></div>', unsafe_allow_html=True)
        d_c3.markdown(f'<div class="glass-card"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Pending AI Scan</p><h3 style="font-size:1.6rem; margin:5px 0 0 0; color:#ffa000;">{unscanned_count}</h3></div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Check background thread status
        is_background_scanning = any(t.name == "AI_Background_Scan" for t in threading.enumerate())

        if is_background_scanning:
            st.markdown(
                f"""
                <div class="glass-card" style="padding: 18px; border: 1px solid rgba(41, 182, 246, 0.35); background: rgba(41, 182, 246, 0.05); border-radius: 12px; margin-bottom: 22px;">
                    <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
                        <div style="font-size: 2.2rem; animation: pulse 2s infinite; color: #29b6f6; display: flex; align-items: center;">⚡</div>
                        <div style="flex: 1; min-width: 250px;">
                            <span style="font-weight: 700; color: #29b6f6; font-size: 1.1rem; display: block; margin-bottom: 4px;">🤖 AI Pattern Recognition Active in Background</span>
                            <span style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.4;">
                                Streamlit is analyzing <b>{unscanned_count} pending stocks</b> using parallel daemon threads in the backend. 
                                Feel free to monitor other tabs, update your watchlists, or examine charts in the meantime!
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Add dynamic refresh button
            if st.button("🔄 Refresh progressive AI results", key="refresh_ai_background_scan_results", width="stretch"):
                st.rerun()
        else:
            # Batch Scan Control Buttons
            btn_cols = st.columns(2)
            btn_batch_scan = False
            btn_force_batch_scan = False
            
            if unscanned_count > 0:
                btn_batch_scan = btn_cols[0].button(f"🤖 Trigger Background AI Scan ({unscanned_count} Pending)", key="batch_ai_scan_action_btn", width="stretch")
                
            if len(active_flagged_symbols) > 0:
                btn_force_batch_scan = btn_cols[1].button(f"🔄 Force Re-scan All ({len(active_flagged_symbols)} Flagged Candidates)", key="force_batch_ai_scan_action_btn", width="stretch", disabled=not is_admin)
                
            if btn_batch_scan or btn_force_batch_scan:
                to_scan_list = []
                for sym in active_flagged_symbols:
                    if btn_force_batch_scan or (sym not in flagged_db_records):
                        to_scan_list.append(sym)
                
                if to_scan_list:
                    try:
                        run_background_ai_scan(to_scan_list, today_str, force=btn_force_batch_scan)
                        st.toast(f"🚀 AI pattern analysis started in the background for {len(to_scan_list)} stocks!", icon="🤖")
                        st.rerun()
                    except Exception as launch_err:
                        st.error(f"❌ Failed to launch background AI scan: {launch_err}")
                
        # Interactive filters for the dashboard list
        st.markdown("#### 🔍 Filter Patterns Identified")
        f_cols = st.columns(3)
        
        unique_patterns = ["All"]
        for s, rec in flagged_db_records.items():
            pat = rec['pattern_name'].strip()
            if pat not in unique_patterns and pat != "None" and pat != "Error":
                unique_patterns.append(pat)
                
        filter_pattern = f_cols[0].selectbox("Filter by Pattern Shape:", options=unique_patterns, key="dash_filter_pat")
        filter_direction = f_cols[1].selectbox("Filter by AI Direction:", options=["All", "Bullish", "Bearish", "Neutral"], key="dash_filter_dir")
        filter_status = f_cols[2].selectbox("Filter by Analysis Status:", options=["All", "AI Scanned Only", "Not Scanned Only"], key="dash_filter_status")
        
        # Display Flagged Stocks list
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 AI Chart Pattern Summary")
        
        tb_cols = st.columns([1.2, 1.2, 2.0, 1.2, 1.2, 2.2, 1.0])
        tb_cols[0].markdown("**Symbol**")
        tb_cols[1].markdown("**Scanner Type**")
        tb_cols[2].markdown("**Pattern Shape**")
        tb_cols[3].markdown("**Direction**")
        tb_cols[4].markdown("**Confidence**")
        tb_cols[5].markdown("**AI Technical Remarks**")
        tb_cols[6].markdown("**Actions**")
        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        displayed_rows = 0
        for sym in active_flagged_symbols:
            rec = flagged_db_records.get(sym)
            
            # Apply filters
            if filter_status == "AI Scanned Only" and not rec:
                continue
            if filter_status == "Not Scanned Only" and rec:
                continue
                
            if rec:
                pat_name = rec['pattern_name'].strip()
                dir_val = rec['direction'].strip().capitalize()
                conf_val = rec['confidence'].strip().capitalize()
                text_val = rec['analysis_text']
                
                if filter_pattern != "All" and pat_name != filter_pattern:
                    continue
                if filter_direction != "All" and dir_val != filter_direction:
                    continue
            else:
                pat_name = "None/Pending"
                dir_val = "Pending"
                conf_val = "Pending"
                text_val = "Stock has not been analyzed by AI technical analyst yet. Click batch scan above to compute."
                
                if filter_pattern != "All":
                    continue
                if filter_direction != "All":
                    continue
                    
            displayed_rows += 1
            
            row_cols = st.columns([1.2, 1.2, 2.0, 1.2, 1.2, 2.2, 1.0])
            
            # Symbol & Origin styling
            tv_sym = sym.replace('.NS', '')
            row_cols[0].markdown(f"<a href='https://in.tradingview.com/chart/?symbol=NSE:{tv_sym}' target='_blank' rel='noopener noreferrer' style='color: #29b6f6; font-weight: bold; text-decoration: none;'>{sym}</a>", unsafe_allow_html=True)
            
            origin = symbol_origins.get(sym, "📊 Breakout")
            origin_color = "#29b6f6" if "Breakout" in origin else "#ab47bc"
            row_cols[1].markdown(f"<span style='color:{origin_color}; font-weight:600;'>{origin}</span>", unsafe_allow_html=True)
            
            # Pattern Shape
            if rec:
                row_cols[2].markdown(f"<b style='color:#ffa000;'>{pat_name}</b>", unsafe_allow_html=True)
                
                # Direction badge
                if dir_val == "Bullish":
                    d_badge = '<span class="custom-badge badge-green">▲ Bullish</span>'
                elif dir_val == "Bearish":
                    d_badge = '<span class="custom-badge badge-red">▼ Bearish</span>'
                else:
                    d_badge = '<span class="custom-badge badge-blue">■ Neutral</span>'
                    
                # Confidence badge
                if conf_val == "High":
                    c_badge = '<span class="custom-badge badge-amber">★ High</span>'
                elif conf_val == "Medium":
                    c_badge = '<span class="custom-badge badge-blue">☆ Medium</span>'
                else:
                    c_badge = '<span class="custom-badge badge-grey">☆ Low</span>'
            else:
                row_cols[2].markdown("<span style='color:#64748b;'>⏳ Not Scanned</span>", unsafe_allow_html=True)
                d_badge = '<span class="custom-badge badge-grey">⏳ Pending</span>'
                c_badge = '<span class="custom-badge badge-grey">⏳ Pending</span>'
                
            row_cols[3].markdown(d_badge, unsafe_allow_html=True)
            row_cols[4].markdown(c_badge, unsafe_allow_html=True)
            
            # Shortened remarks snippet
            remarks_snippet = text_val[:80] + "..." if len(text_val) > 80 else text_val
            row_cols[5].markdown(f"<span style='font-size:0.85rem; color:#94a3b8;'>\"{remarks_snippet}\"</span>", unsafe_allow_html=True)
            
            # Action button to select this ticker inside selector
            action_key = f"dash_load_{sym}_{displayed_rows}"
            
            def set_ai_selection(s=sym):
                st.session_state.ai_selected_stock = s
                
            if row_cols[6].button("🔍 View", key=action_key, width="stretch", on_click=set_ai_selection):
                st.toast(f"🔍 Loading detailed charts & AI context for {sym}...")
                
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.03);'>", unsafe_allow_html=True)
            
        if displayed_rows == 0:
            st.info("ℹ️ No stocks match the active filters in this dashboard.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 📋 Recent AI Patterns Scanned")
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8; margin-top:-10px;'>A real-time dashboard of technical patterns identified by other scans saved on Neon PostgreSQL.</p>", unsafe_allow_html=True)
    
    recent_records = database.get_recent_patterns(limit=10)
    if not recent_records:
        st.info("ℹ️ No technical patterns have been analyzed or saved in the database yet. Select a stock above and run the AI scanner to cache the first result!")
    else:
        # Sort and build dashboard columns
        head_cols = st.columns([1.5, 2.5, 1.5, 1.5, 2.0, 1.5])
        head_cols[0].markdown("**Symbol**")
        head_cols[1].markdown("**Pattern Identified**")
        head_cols[2].markdown("**Direction**")
        head_cols[3].markdown("**Confidence**")
        head_cols[4].markdown("**Analyzed Date**")
        head_cols[5].markdown("**Fetch Cache**")
        st.markdown("<hr style='margin: 8px 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
        
        for idx, rec in enumerate(recent_records):
            row_cols = st.columns([1.5, 2.5, 1.5, 1.5, 2.0, 1.5])
            tv_sym = rec['symbol'].replace('.NS', '')
            row_cols[0].markdown(f"<a href='https://in.tradingview.com/chart/?symbol=NSE:{tv_sym}' target='_blank' rel='noopener noreferrer' style='color: #29b6f6; font-weight: bold; text-decoration: none;'>{rec['symbol']}</a>", unsafe_allow_html=True)
            row_cols[1].markdown(f"<span style='color:#ffa000; font-weight:500;'>{rec['pattern_name']}</span>", unsafe_allow_html=True)
            
            # Direction styling
            d_lower = rec['direction'].strip().lower()
            if d_lower == "bullish":
                d_badge = '<span class="custom-badge badge-green">▲ Bullish</span>'
            elif d_lower == "bearish":
                d_badge = '<span class="custom-badge badge-red">▼ Bearish</span>'
            else:
                d_badge = '<span class="custom-badge badge-blue">■ Neutral</span>'
                
            # Confidence styling
            c_lower = rec['confidence'].strip().lower()
            if c_lower == "high":
                c_badge = '<span class="custom-badge badge-amber">★ High</span>'
            elif c_lower == "medium":
                c_badge = '<span class="custom-badge badge-blue">☆ Medium</span>'
            else:
                c_badge = '<span class="custom-badge badge-grey">☆ Low</span>'
                
            row_cols[2].markdown(d_badge, unsafe_allow_html=True)
            row_cols[3].markdown(c_badge, unsafe_allow_html=True)
            row_cols[4].markdown(f"<span style='font-size:0.85rem; color:#94a3b8;'>{rec['analyzed_date']}</span>", unsafe_allow_html=True)
            
            # Action button to load this symbol's cached analysis
            def set_cached_ai_selection(s=rec['symbol']):
                st.session_state.ai_selected_stock = s
                
            if row_cols[5].button("⚡ Load", key=f"load_rec_{rec['symbol']}_{idx}", width="stretch", on_click=set_cached_ai_selection):
                st.toast(f"Loading cached analysis for {rec['symbol']}!")
                
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.03);'>", unsafe_allow_html=True)
