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
    # Mode selector for analysis target
    search_mode = st.radio(
        "Choose Analysis Target Mode:",
        ["🔍 Select from Scanned Breakouts", "✏️ Search Any Ticker (Custom Assessment)"],
        horizontal=True,
        key="detail_search_mode_radio",
        help="Analyze scanned breakouts from the current scanner run, or enter any stock ticker name for real-time custom technical assessment."
    )
    
    detail_data = None
    
    if search_mode == "🔍 Select from Scanned Breakouts":
        if not scan_data or len(scan_data) == 0:
            st.info("💡 No scan results available. Run a scanner from the sidebar first, or switch to Custom Ticker mode to search any stock manually.")
        else:
            symbols_flagged = [r['symbol'] for r in scan_data]
            selected_sym = st.selectbox(
                "Select Scanned Stock for Detailed Charting:",
                options=symbols_flagged,
                index=0,
                help="Choose a stock from current scan output"
            )
            detail_data = next((r for r in scan_data if r['symbol'] == selected_sym), None)
    else:
        # Custom search mode
        custom_input = st.text_input(
            "Enter NSE Ticker Name (e.g. SBIN, RELIANCE, INFIBEAM, TATASTEEL):",
            value="",
            key="detail_custom_ticker_input",
            help="Type any active NSE ticker. We will download its real-time quotes, calculate indicators, and generate custom recommendations."
        ).strip().upper()
        
        if custom_input:
            with st.spinner(f"Fetching quotes and calculating technical indicators for {custom_input}..."):
                df_custom = fetch_ohlcv(custom_input)
                if df_custom is None or df_custom.empty:
                    st.error(f"❌ Failed to retrieve historical data for '{custom_input}'. Please check the ticker name and try again.")
                else:
                    cmp_val = float(df_custom['Close'].iloc[-1])
                    buy_price = round(cmp_val, 2)
                    min_5d_low = float(df_custom['Low'].iloc[-5:].min()) if len(df_custom) >= 5 else cmp_val
                    exit_price = round(min(buy_price * 0.95, min_5d_low * 0.98), 2)
                    target_price = round(buy_price * 1.15, 2)
                    
                    rich_payload = compute_rich_analysis(
                        df_custom, 
                        custom_input, 
                        "Custom Technical Assessment", 
                        f"Custom Technical entry on dynamic indicators confluence. Buy around ₹{buy_price:.2f} with stop loss ₹{exit_price:.2f} and target swing price ₹{target_price:.2f} (+15%)."
                    )
                    
                    yesterday_close = float(df_custom['Close'].iloc[-2]) if len(df_custom) >= 2 else cmp_val
                    day_change_pct = ((cmp_val - yesterday_close) / yesterday_close * 100) if yesterday_close > 0 else 0.0
                    
                    dry_avg_vol = float(df_custom['Volume'].mean())
                    today_volume = float(df_custom['Volume'].iloc[-1])
                    volume_ratio = today_volume / dry_avg_vol if dry_avg_vol > 0 else 1.0
                    
                    detail_data = {
                        "symbol": custom_input,
                        "company_name": get_company_name(custom_input),
                        "cmp": cmp_val,
                        "day_change_pct": round(day_change_pct, 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "buy_price": buy_price,
                        "exit_price": exit_price,
                        "target_price": target_price,
                        "confidence": "Medium-High Assessment",
                        "recommendation": rich_payload,
                        "df": df_custom,
                        "dry_start_date": df_custom['Date'].iloc[-min(30, len(df_custom))],
                        "dry_end_date": df_custom['Date'].iloc[-1],
                        "dry_days_count": 0,
                        "dry_avg_vol": dry_avg_vol,
                        "today_volume": int(today_volume),
                        "signal_strength": 65.0,
                        "above_50dma": cmp_val > (df_custom['Close'].rolling(window=50).mean().iloc[-1] if len(df_custom) >= 50 else cmp_val)
                    }
        
    if detail_data:
        selected_sym = detail_data['symbol']
        # Lazy-load historical OHLCV data for charting if loaded from daily database cache
        if 'df' not in detail_data or detail_data['df'] is None or detail_data['df'].empty:
            with st.spinner(f"Lazy-loading historical candle data for {selected_sym}..."):
                detail_data['df'] = fetch_ohlcv(selected_sym)
        
        df = detail_data['df']
        if df is None or df.empty:
            st.warning(f"⚠️ Could not load historical chart data for {selected_sym}. Please verify your connection or choose another stock.")
        else:
            try:
                if df is not None and 'MA50' not in df.columns:
                    df['MA50'] = df['Close'].rolling(window=50).mean()
                if df is not None:
                    if 'high_52w' not in detail_data or detail_data.get('high_52w') is None:
                        detail_data['high_52w'] = float(df['High'].max())
                    if 'low_52w' not in detail_data or detail_data.get('low_52w') is None:
                        detail_data['low_52w'] = float(df['Low'].min())
                today_date = df['Date'].iloc[-1]
                dry_start_date = detail_data.get('dry_start_date', df['Date'].iloc[-min(30, len(df))] if len(df) > 0 else today_date)
                dry_end_date = detail_data.get('dry_end_date', today_date)
                dry_days_count = detail_data.get('dry_days_count', 0)
                dry_avg_vol = detail_data.get('dry_avg_vol', df['Volume'].mean() if len(df) > 0 else 0)
                volume_ratio = detail_data.get('volume_ratio', 1.0)
                signal_strength = detail_data.get('signal_strength', 50.0)
                above_50dma = detail_data.get('above_50dma', False)
                today_volume = detail_data.get('today_volume', int(df['Volume'].iloc[-1]) if len(df) > 0 else 0)

                # Calculate dry zone return
                try:
                    dry_start_mask = df['Date'] >= pd.to_datetime(dry_start_date)
                    dry_end_mask = df['Date'] <= pd.to_datetime(dry_end_date)
                    dry_df = df[dry_start_mask & dry_end_mask]
                    if not dry_df.empty:
                        dry_start_price = dry_df.iloc[0]['Close']
                        dry_end_price = dry_df.iloc[-1]['Close']
                        dry_zone_return = ((dry_end_price - dry_start_price) / dry_start_price) * 100
                    else:
                        dry_zone_return = 0.0
                except Exception:
                    dry_zone_return = 0.0

                # Limit chart data to max ~7 months (150 trading days) for better visibility
                df = df.tail(150)

                # A. Dual subplot layout
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"📈 {selected_sym} Candlestick Chart & 50 DMA", f"📊 Volume Analysis")
                )

                # Top Candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=df['Date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name="Price",
                        increasing_line_color="#00e676",
                        decreasing_line_color="#ef4444"
                    ),
                    row=1, col=1
                )

                # Top 50 DMA trace
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df['MA50'],
                        name="50 DMA",
                        line=dict(color="#ab47bc", width=2, dash="dash"),
                        mode="lines"
                    ),
                    row=1, col=1
                )

                # Bottom volume color builder
                bar_colors = []
                for _, row in df.iterrows():
                    row_date = row['Date']
                    if row_date == today_date:
                        bar_colors.append("#00e676") # Breakout surge
                    elif dry_start_date <= row_date <= dry_end_date:
                        bar_colors.append("#475569") # Dry volume zone
                    else:
                        bar_colors.append("#1e3a8a") # Normal blue volume

                fig.add_trace(
                    go.Bar(
                        x=df['Date'],
                        y=df['Volume'],
                        name="Volume",
                        marker_color=bar_colors,
                        showlegend=False
                    ),
                    row=2, col=1
                )

                # Prevent extreme volume outliers from squishing the volume bars
                fig.update_yaxes(range=[0, df['Volume'].quantile(0.99) * 1.5], row=2, col=1)

                # Shade the dry zone region on the candlestick subplot
                fig.add_vrect(
                    x0=dry_start_date,
                    x1=dry_end_date,
                    fillcolor="rgba(255, 160, 0, 0.08)",
                    opacity=0.6,
                    layer="below",
                    line_width=1,
                    line_color="rgba(255,160,0,0.15)",
                    annotation_text="📭 Dry Zone (Consolidation)",
                    annotation_position="top left",
                    annotation_font=dict(color="#ffa000", size=11, family="Outfit"),
                    row=1, col=1
                )

                # Draw breakout arrow annotation on today's price action
                fig.add_annotation(
                    x=today_date,
                    y=detail_data['cmp'],
                    text="🚀 Breakout",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2,
                    arrowcolor="#00e676",
                    ax=-50,
                    ay=-40,
                    font=dict(color="#00e676", size=12, family="Outfit", weight="bold"),
                    bgcolor="rgba(0, 230, 118, 0.08)",
                    bordercolor="rgba(0,230,118,0.3)",
                    borderwidth=1,
                    borderpad=4,
                    row=1, col=1
                )

                # Visual templates update
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="#090d16",
                    paper_bgcolor="#090d16",
                    margin=dict(l=40, r=40, t=40, b=40),
                    xaxis=dict(
                        rangeslider=dict(visible=False),
                        gridcolor="rgba(255,255,255,0.04)",
                        rangebreaks=[dict(bounds=["sat", "mon"])]
                    ),
                    xaxis2=dict(
                        gridcolor="rgba(255,255,255,0.04)"
                    ),
                    yaxis=dict(
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Price (₹)"
                    ),
                    yaxis2=dict(
                        gridcolor="rgba(255,255,255,0.04)",
                        title="Volume"
                    ),
                    font=dict(family="Outfit, sans-serif"),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=600
                )

                st.plotly_chart(fig, width="stretch")

                st.markdown("---")

                # B. 3-column detailed metric cards
                c1, c2, c3 = st.columns(3)

                # Column 1
                c1.markdown(f"""
                <div class="glass-card">
                    <h4 style="margin-top:0; color:#29b6f6; font-size:1.1rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:8px;">📈 Price Action Details</h4>
                    <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Current Price:</span><br><b style="font-size:1.3rem;">₹{detail_data['cmp']:.2f}</b></div>
                    <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Price Change today:</span><br>{get_day_change_badge_html(detail_data['day_change_pct'])}</div>
                    <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">120d Period High / Low:</span><br><b>₹{detail_data['high_52w']:.2f}</b> / <b>₹{detail_data['low_52w']:.2f}</b></div>
                </div>
                """, unsafe_allow_html=True)

                # Column 2
                c2.markdown(f"""
                <div class="glass-card">
                    <h4 style="margin-top:0; color:#00e676; font-size:1.1rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:8px;">📭 Dry Zone Volume Metrics</h4>
                    <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Volume Ratio:</span><br><b style="font-size:1.3rem; color:#00e676;">{volume_ratio:.2f}x</b> (vs Dry Average)</div>
                    <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Dry zone Duration:</span><br><b>{dry_days_count}</b> trading days</div>
                    <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Dry zone Return:</span><br><b style="color:{'#00e676' if dry_zone_return >= 0 else '#ef4444'};">{dry_zone_return:+.2f}%</b></div>
                    <div style="margin: 12px 0;"><span style="color:#94a3b8; font-size:0.9rem;">Dry average / today's volume:</span><br><b>{int(dry_avg_vol):,}</b> / <b>{today_volume:,}</b></div>
                </div>
                """, unsafe_allow_html=True)

                # Column 3: Custom Plotly Gauge Chart for strength
                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=signal_strength,
                        title={'text': "Signal Score Rating", 'font': {'size': 15, 'color': '#ffa000', 'family': 'Outfit'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                            'bar': {'color': "#ffa000"},
                            'bgcolor': "rgba(255,255,255,0.03)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(255,255,255,0.08)",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(148, 163, 184, 0.08)'},
                                {'range': [50, 70], 'color': 'rgba(41, 182, 246, 0.12)'},
                                {'range': [70, 100], 'color': 'rgba(255, 160, 0, 0.16)'}
                            ]
                        }
                    )
                )
                gauge_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#e2e8f0", 'family': "Outfit"},
                    height=180,
                    margin=dict(l=15, r=15, t=30, b=10)
                )

                with c3:
                    st.plotly_chart(gauge_fig, width="stretch")

                    # DMA Flag badge
                    dma_status = above_50dma
                    dma_badge = '<span class="custom-badge badge-green">▲ ABOVE 50 DMA</span>' if dma_status else '<span class="custom-badge badge-red">▼ BELOW 50 DMA</span>'

                    st.markdown(
                        f"""
                        <div style='text-align:center; padding:12px; background:rgba(17, 24, 39, 0.4); border-radius:10px; border:1px solid rgba(255,255,255,0.05); margin-top:-10px;'>
                            <b>DMA Trend Filter:</b><br>{dma_badge}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                    # Render the gorgeous Technical Indicators dashboard and checklists!
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_trading_setup_card(detail_data, "detail_tab_setup", 0)
            except Exception as chart_err:
                st.error(f"❌ Error rendering charts for {selected_sym}: {chart_err}")
