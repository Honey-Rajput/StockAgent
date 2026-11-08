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
try:
    from export_pva_matrix import classify_pva, generate_pva_excel_bytes
except ImportError:
    from vdu_scanner.export_pva_matrix import classify_pva, generate_pva_excel_bytes


# ─────────────────────────────────────────────────────────────────────────────
# BLAST SCORE: Composite pre-breakout scoring (0-100)
# Weights: Multi-TF alignment 40 + Volume surge 25 + Momentum accel 20 + RSI zone 15
# ─────────────────────────────────────────────────────────────────────────────
def compute_blast_score(r: dict) -> dict:
    """
    Returns a dict with 'blast_score' (0–100) and 'blast_reasons' list.
    High score = stock primed to shoot up in coming days.
    """
    d = r.get('daily', {})
    w = r.get('weekly', {})
    m = r.get('monthly', {})
    score = 0
    reasons = []

    # ── 1. Multi-Timeframe Alignment (max 40 pts) ────────────────────────────
    daily_bull  = (d.get('major', 0) == 1 and d.get('mid', 0) == 1 and d.get('minor', 0) == 1)
    weekly_bull = (w.get('major', 0) == 1 and w.get('mid', 0) == 1 and w.get('minor', 0) == 1)
    monthly_bull= (m.get('major', 0) == 1 and m.get('mid', 0) == 1)

    if daily_bull and weekly_bull and monthly_bull:
        score += 40
        reasons.append("✅ Triple Alignment: Daily + Weekly + Monthly Bullish")
    elif daily_bull and weekly_bull:
        score += 30
        reasons.append("✅ Dual Alignment: Daily + Weekly Bullish Stack")
    elif daily_bull and monthly_bull:
        score += 25
        reasons.append("🟢 Daily + Monthly Macro Bullish Stack")
    elif weekly_bull and monthly_bull:
        score += 20
        reasons.append("🟢 Weekly + Monthly Macro Base Bullish")
    elif daily_bull:
        score += 12
        reasons.append("🟡 Daily Bull Trend Active")
    elif weekly_bull:
        score += 10

    # Partial credit: minor turning up while major/mid already up (Early Entry / Pullback)
    if d.get('major', 0) == 1 and d.get('mid', 0) == 1 and d.get('minor', 0) == 0:
        score += 8
        reasons.append("🔵 Daily Minor Just Turning Up (High-Probability Pullback Entry)")

    # ── 2. Volume Surge (max 25 pts) ────────────────────────────────────────
    vol_ratio = r.get('vol_ratio_5d', 1.0)
    if vol_ratio >= 2.5:
        score += 25
        reasons.append(f"🔥 Massive Volume Expansion ({vol_ratio:.1f}x 20-day avg)")
    elif vol_ratio >= 1.8:
        score += 18
        reasons.append(f"📈 Strong Volume Expansion ({vol_ratio:.1f}x avg)")
    elif vol_ratio >= 1.3:
        score += 10
        reasons.append(f"📊 Volume Rising ({vol_ratio:.1f}x avg)")
    elif vol_ratio < 0.7:
        score -= 5
        reasons.append(f"💧 Volume Dry-Up ({vol_ratio:.1f}x) — Setup in Consolidation")

    # ── 3. Momentum Acceleration (max 20 pts) ───────────────────────────────
    d_major_val = abs(d.get('major_val', 0))
    w_major_val = abs(w.get('major_val', 0))

    if d_major_val > 2.0 and w_major_val > 1.5:
        score += 20
        reasons.append(f"🚀 Powerful RWI Trend Velocity (D:{d_major_val:.1f}, W:{w_major_val:.1f})")
    elif d_major_val > 1.5:
        score += 12
        reasons.append(f"📈 Solid Daily RWI Momentum ({d_major_val:.1f})")
    elif d_major_val > 1.0:
        score += 6

    # ── 4. RSI Zone (max 15 pts) ─────────────────────────────────────────────
    d_rsi = d.get('rsi', 50)
    w_rsi = w.get('rsi', 50)
    if 55 <= d_rsi <= 72 and 50 <= w_rsi <= 75:
        score += 15
        reasons.append(f"✅ RSI Sweet Spot (D:{d_rsi:.0f}, W:{w_rsi:.0f}) — High Momentum, Not Overbought")
    elif 50 <= d_rsi <= 78:
        score += 8
        reasons.append(f"🟡 RSI Positive Zone (D:{d_rsi:.0f})")
    elif d_rsi > 80:
        score -= 10
        reasons.append(f"⚠️ RSI Overbought ({d_rsi:.0f}) — Potential Near-term Exhaustion")
    elif d_rsi < 40:
        score -= 6
        reasons.append(f"⚠️ RSI Subdued ({d_rsi:.0f})")

    # ── 5. Bonus: Breakout Proximity ─────────────────────────────────────────
    if r.get('near_breakout', False):
        score += 8
        reasons.append("💎 Hovering within 3% of 20-Day High (Breakout Imminent)")

    # ── 6. Penalty: Higher Timeframe Headwinds ────────────────────────────────
    if m.get('major', 0) == -1:
        score -= 12
        reasons.append("❌ Monthly Major Downtrend — Counter-Trend Headwind")
    if w.get('major', 0) == -1 and d.get('major', 0) == 1:
        score -= 6
        reasons.append("⚠️ Weekly Downtrend — Short-Term Counter Trend Only")

    score = max(0, min(100, score))

    if score >= 75:
        tier = "🔥 BLAST"
    elif score >= 55:
        tier = "🚀 Strong"
    elif score >= 35:
        tier = "📈 Moderate"
    else:
        tier = "⚪ Weak"

    return {
        'blast_score': score,
        'blast_tier': tier,
        'blast_reasons': reasons
    }


def render():
    scan_data = st.session_state.get('scan_results', [])
    is_admin = st.session_state.get('is_admin', False)
    st.markdown("### 🚥 VPA Trend Indicator — Volume Price Analysis")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-8px;'>"
        "Scans all NSE stocks across <b>Daily, Weekly & Monthly</b> timeframes using Random Walk Index (RWI). "
        "The <b>🔥 Blast Watchlist</b> highlights stocks primed for an imminent explosive move (Multi-Timeframe Trend Stacking + Volume Expansion + Healthy RSI)."
        "</p>",
        unsafe_allow_html=True
    )
    
    # Pick up background scan results if available
    if not st.session_state.get('vpa_results') and st.session_state.ALL_TAB_SCAN_STATUS["vpa_results"] is not None:
        st.session_state.vpa_results = st.session_state.ALL_TAB_SCAN_STATUS["vpa_results"]

    col1, col2 = st.columns([3, 7])
    with col1:
        run_vpa_btn = st.button("🚀 Run Advanced VPA Scan", width="stretch", disabled=not is_admin)
    
    if run_vpa_btn:
        st.session_state.vpa_results = []
        with st.spinner("Initializing Ultra-Fast VPA Scan on ALL NSE Stocks..."):
            try:
                from data_fetcher import get_all_nse_symbols
                import yfinance as yf
                from concurrent.futures import ThreadPoolExecutor
                import time
                
                raw_symbols = get_all_nse_symbols()
                all_symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
                
                st.info(f"Phase 1: Downloading 5 years of history for {len(all_symbols)} stocks...")
                prog = st.progress(0)
                status = st.empty()
                
                chunk_size = 300
                sym_chunks = [all_symbols[i:i + chunk_size] for i in range(0, len(all_symbols), chunk_size)]
                
                valid_data = {}
                price_filtered = []
                
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
                
                st.info("Phase 2: Calculating VPA Trends & Blast Scores...")
                status.empty()
                prog.progress(1.0)
                
                vpa_list = []
                for sym in price_filtered:
                    df = valid_data[sym]
                    clean_sym = sym.replace('.NS', '')
                    vpa_res = scan_vpa_trend(clean_sym, df)
                    if vpa_res is not None:
                        vol_20 = df['Volume'].tail(20).mean()
                        vol_5  = df['Volume'].tail(5).mean()
                        vpa_res['vol_ratio_5d'] = round(vol_5 / vol_20, 2) if vol_20 > 0 else 1.0
                        high_20 = df['Close'].tail(20).max()
                        cmp = df['Close'].iloc[-1]
                        vpa_res['near_breakout'] = bool((high_20 - cmp) / cmp <= 0.03) if cmp > 0 else False
                        vpa_res['market_cap_cr'] = 0
                        blast = compute_blast_score(vpa_res)
                        vpa_res.update(blast)
                        vpa_list.append(vpa_res)
                            
                prog.empty()
                status.empty()
                st.session_state.vpa_results = vpa_list
                try:
                    today_ist_str = get_market_date()
                    database.save_vpa_only(today_ist_str, vpa_list)
                except Exception as e:
                    print(f"Failed to cache custom VPA scan: {e}")
                st.success(f"VPA Scan complete! Found {len(vpa_list)} stocks. Blast Watchlist ready!")
                
            except Exception as e:
                st.error(f"Scan failed: {e}")
                
    if not st.session_state.get('vpa_results'):
        if st.session_state.ALL_TAB_SCAN_STATUS["is_running"]:
            _bg_status = st.session_state.ALL_TAB_SCAN_STATUS["status_text"]
            _bg_progress = st.session_state.ALL_TAB_SCAN_STATUS["progress"]
            st.markdown(f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px;">
                <h4 style="color:#00e5ff; margin:0 0 10px 0;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">VPA results will appear here when ready!</p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """, unsafe_allow_html=True)
            st.progress(_bg_progress)
            if st.button("🔄 Refresh Scanner Status", key="refresh_bg_vpa_status_btn"):
                st.rerun()
                
        st.info("No VPA data available. Click 'Run Advanced VPA Scan' to process.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # DATA ENRICHMENT — compute blast scores if missing (loaded from DB cache)
    # ─────────────────────────────────────────────────────────────────────────
    vpa_data = st.session_state.vpa_results
    for r in vpa_data:
        if 'vol_ratio_5d' not in r:
            r['vol_ratio_5d'] = r.get('vol_ratio_5d', 1.0)
        if 'blast_score' not in r:
            r['near_breakout'] = r.get('near_breakout', False)
            blast = compute_blast_score(r)
            r.update(blast)
        if 'pva' not in r:
            r['pva'] = classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def get_action_signal_text(short, mid, max_t, max_val, rsi=0, cci=0):
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

    def trend_badge(t_val):
        if t_val == 1:
            return "<span style='color:#00e676; font-weight:bold;'>▲ Up</span>"
        elif t_val == -1:
            return "<span style='color:#ef4444; font-weight:bold;'>▼ Dn</span>"
        return "<span style='color:#fbbf24; font-weight:bold;'>◼ Neu</span>"

    def action_badge(short, mid, max_t, max_val, rsi=0, cci=0):
        text = get_action_signal_text(short, mid, max_t, max_val, rsi=rsi, cci=cci)
        if "Perfect Buy" in text:
            return f"<span style='color:#00e676; font-weight:bold;'>🟢 {text}</span>"
        elif "Counter Trend Buy" in text:
            return f"<span style='color:#4ade80; font-weight:bold;'>🟢 {text}</span>"
        elif "Early Breakout" in text:
            return f"<span style='color:#3b82f6; font-weight:bold;'>🔵 {text}</span>"
        elif "Pullback" in text:
            return f"<span style='color:#fbbf24; font-weight:bold;'>🟡 {text}</span>"
        elif "Warning" in text:
            return f"<span style='color:#f97316; font-weight:bold;'>🟠 {text}</span>"
        elif "Avoid" in text:
            return f"<span style='color:#ef4444; font-weight:bold;'>🔴 {text}</span>"
        elif "Parabolic" in text or "Overextended" in text:
            return f"<span style='color:#d946ef; font-weight:bold;'>🟣 {text}</span>"
        return f"<span style='color:#9ca3af; font-weight:bold;'>⚪ {text}</span>"

    def blast_tier_color(tier):
        if "BLAST" in tier:
            return "#ff4444"
        elif "Strong" in tier:
            return "#ff9900"
        elif "Moderate" in tier:
            return "#29b6f6"
        return "#64748b"

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: 🔥 BLAST WATCHLIST
    # ─────────────────────────────────────────────────────────────────────────
    blast_stocks = [r for r in vpa_data if r.get('blast_score', 0) >= 55]
    blast_stocks = sorted(blast_stocks, key=lambda x: x.get('blast_score', 0), reverse=True)

    st.markdown("---")
    st.markdown(
        f"<h3 style='color:#ff4444; margin-bottom:4px;'>🔥 Blast Watchlist — Stocks Prime to Shoot in Coming Days</h3>"
        f"<p style='font-size:0.85rem; color:#94a3b8; margin-top:0;'>"
        f"Selected using multi-timeframe trend alignment, institutional volume surge, and explosive RWI momentum. "
        f"<b style='color:#ff9900;'>{len(blast_stocks)} stocks</b> qualify with Blast Score ≥ 55/100."
        f"</p>",
        unsafe_allow_html=True
    )

    if not blast_stocks:
        st.info("🔍 No blast setups meeting threshold (≥55) in the current dataset. Run a fresh VPA scan to evaluate latest candles.")
    else:
        bc1, bc2, bc3, bc4 = st.columns(4)
        blast_tier = [r for r in blast_stocks if r.get('blast_score', 0) >= 75]
        strong_tier = [r for r in blast_stocks if 55 <= r.get('blast_score', 0) < 75]
        avg_blast = sum(r.get('blast_score', 0) for r in blast_stocks) / len(blast_stocks)
        bc1.markdown(f'<div class="glass-card" style="padding:12px 16px;"><p style="font-size:0.8rem; color:#94a3b8; margin:0;">🔥 High-Probability BLAST</p><h3 style="font-size:1.6rem; margin:4px 0 0 0; color:#ff4444;">{len(blast_tier)}</h3></div>', unsafe_allow_html=True)
        bc2.markdown(f'<div class="glass-card" style="padding:12px 16px;"><p style="font-size:0.8rem; color:#94a3b8; margin:0;">🚀 Strong Conviction</p><h3 style="font-size:1.6rem; margin:4px 0 0 0; color:#ff9900;">{len(strong_tier)}</h3></div>', unsafe_allow_html=True)
        bc3.markdown(f'<div class="glass-card" style="padding:12px 16px;"><p style="font-size:0.8rem; color:#94a3b8; margin:0;">📊 Avg Blast Score</p><h3 style="font-size:1.6rem; margin:4px 0 0 0; color:#29b6f6;">{avg_blast:.0f}/100</h3></div>', unsafe_allow_html=True)
        bc4.markdown(f'<div class="glass-card" style="padding:12px 16px;"><p style="font-size:0.8rem; color:#94a3b8; margin:0;">📈 Total VPA Universe</p><h3 style="font-size:1.6rem; margin:4px 0 0 0; color:#00e676;">{len(vpa_data)}</h3></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        blast_rows = []
        for i, r in enumerate(blast_stocks[:50], 1):
            d = r['daily']
            w = r['weekly']
            m = r['monthly']
            sym = r['symbol']
            tv_link = f"https://in.tradingview.com/chart/?symbol=NSE:{sym.replace('.NS','')}"
            tier = r.get('blast_tier', '')
            tier_color = blast_tier_color(tier)
            reasons_html = "<br>".join(r.get('blast_reasons', [])[:4])
            vol_ratio = r.get('vol_ratio_5d', 1.0)

            blast_rows.append(f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.05); background:{'rgba(255,68,68,0.04)' if r.get('blast_score',0)>=75 else 'transparent'};">
<td style="padding:10px 8px; font-weight:bold; color:#94a3b8;">#{i}</td>
<td style="padding:10px 8px;"><a href="{tv_link}" target="_blank" style="color:#00e5ff; font-weight:bold; text-decoration:none;">{sym}</a></td>
<td style="padding:10px 8px; color:#e2e8f0;">₹{r['cmp']:,.2f}</td>
<td style="padding:10px 8px;">{get_day_change_badge_html(r['day_change_pct'])}</td>
<td style="padding:10px 8px; font-weight:bold; font-size:1.1rem; color:{tier_color};">{r.get('blast_score',0)}</td>
<td style="padding:10px 8px; color:{tier_color}; font-weight:bold;">{tier}</td>
<td style="padding:10px 8px;">{trend_badge(d['major'])} / {trend_badge(w['major'])} / {trend_badge(m['major'])}</td>
<td style="padding:10px 8px;">{trend_badge(d['mid'])} / {trend_badge(w['mid'])} / {trend_badge(m['mid'])}</td>
<td style="padding:10px 8px; color:{'#00e676' if vol_ratio>=1.5 else '#fbbf24' if vol_ratio>=1.0 else '#ef4444'};">{vol_ratio:.1f}x</td>
<td style="padding:10px 8px; font-size:0.78rem; color:#94a3b8; max-width:240px;">{reasons_html}</td>
</tr>""")

        blast_html = f"""<div style="overflow-x:auto; margin-top:10px;">
<table style="width:100%; border-collapse:collapse; font-size:0.9rem;">
<thead>
<tr style="background:rgba(255,68,68,0.08); border-bottom:2px solid rgba(255,68,68,0.3);">
<th style="padding:10px 8px; color:#ff4444;">#</th>
<th style="padding:10px 8px; color:#ff4444;">Symbol</th>
<th style="padding:10px 8px; color:#ff4444;">CMP</th>
<th style="padding:10px 8px; color:#ff4444;">Chg%</th>
<th style="padding:10px 8px; color:#ff4444;">Blast Score</th>
<th style="padding:10px 8px; color:#ff4444;">Tier</th>
<th style="padding:10px 8px; color:#ff4444;">Major (D/W/M)</th>
<th style="padding:10px 8px; color:#ff4444;">Mid (D/W/M)</th>
<th style="padding:10px 8px; color:#ff4444;">Vol Ratio</th>
<th style="padding:10px 8px; color:#ff4444;">Why It May Blast</th>
</tr>
</thead>
<tbody>
{"".join(blast_rows)}
</tbody>
</table>
</div>"""
        st.markdown(blast_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        blast_csv_data = []
        for r in blast_stocks:
            d = r['daily']; w = r['weekly']; m = r['monthly']
            blast_csv_data.append({
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Blast Score': r.get('blast_score', 0),
                'Tier': r.get('blast_tier', ''),
                'D Major': r['daily']['major'], 'D Mid': r['daily']['mid'], 'D Minor': r['daily']['minor'],
                'W Major': r['weekly']['major'], 'W Mid': r['weekly']['mid'], 'W Minor': r['weekly']['minor'],
                'M Major': r['monthly']['major'], 'M Mid': r['monthly']['mid'],
                'D RSI': r['daily'].get('rsi', 0), 'W RSI': r['weekly'].get('rsi', 0),
                'Vol Ratio 5d': r.get('vol_ratio_5d', 1.0),
                'Near Breakout': r.get('near_breakout', False),
                'Reasons': ' | '.join(r.get('blast_reasons', [])),
                'VPA Score': r.get('score', 0),
            })
        st.download_button(
            label="📥 Download Blast Watchlist (CSV)",
            data=pd.DataFrame(blast_csv_data).to_csv(index=False).encode('utf-8-sig'),
            file_name="vpa_blast_watchlist.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: 🔬 6-CONDITION PRICE + VOLUME MATRIX (AUTOMATIC INTERPRETATION)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<h3 style='color:#38bdf8; margin-bottom:4px;'>🔬 6-Condition Price + Volume Matrix (Wyckoff / PVA Analysis)</h3>"
        f"<p style='font-size:0.85rem; color:#94a3b8; margin-top:0;'>"
        f"Simultaneously correlates <b>Price Movement</b> and <b>Volume Movement</b> to classify stocks into 6 precise market regimes with automated interpretations."
        f"</p>",
        unsafe_allow_html=True
    )

    with st.expander("📖 View 6 Market Conditions Reference Matrix & Trading Interpretations", expanded=False):
        st.markdown("""
| Volume Movement | Price Movement | Interpretation | Signal | Actionable Bias |
| :--- | :--- | :--- | :--- | :--- |
| **Increasing** | **Increasing** | Strong buying interest; possible accumulation or breakout | <span style="color:#00e676; font-weight:bold;">🟢 Bullish</span> | Look for momentum long entries / add to winners |
| **Increasing** | **Stable** | Buyers and sellers competing; possible accumulation | <span style="color:#38bdf8; font-weight:bold;">🔵 Watch</span> | Prime absorption zone; alert on upside break |
| **Decreasing** | **Increasing** | Price rising with weaker participation; rally may lack strength | <span style="color:#fbbf24; font-weight:bold;">🟡 Caution</span> | Tighten trailing stop; avoid aggressive fresh buys |
| **Decreasing** | **Stable** | Low participation; consolidation or lack of interest | <span style="color:#94a3b8; font-weight:bold;">⚪ Neutral</span> | Sideways churn; wait for volume trigger |
| **Decreasing** | **Decreasing** | Selling pressure may be weakening; possible selling exhaustion | <span style="color:#c084fc; font-weight:bold;">🟣 Watch reversal</span> | Watch for bottom reversal / spring setup |
| **Increasing** | **Decreasing** | Strong selling pressure; possible distribution or breakdown | <span style="color:#ef4444; font-weight:bold;">🔴 Bearish</span> | High risk of breakdown; exit or short |
""", unsafe_allow_html=True)

    # Calculate counts for all 6 conditions
    pva_counts = {
        'Bullish': 0,
        'Watch': 0,
        'Caution': 0,
        'Neutral': 0,
        'Watch reversal': 0,
        'Bearish': 0,
    }
    for r in vpa_data:
        pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
        sig = pva['signal']
        pva_counts[sig] = pva_counts.get(sig, 0) + 1

    # 6 Metric KPI cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #00e676;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🟢 Bullish</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#00e676;">{pva_counts.get("Bullish", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▲ Price ▲</span></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #38bdf8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🔵 Watch</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#38bdf8;">{pva_counts.get("Watch", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▲ Price ◼</span></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #fbbf24;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🟡 Caution</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#fbbf24;">{pva_counts.get("Caution", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▼ Price ▲</span></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #94a3b8;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">⚪ Neutral</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#94a3b8;">{pva_counts.get("Neutral", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▼ Price ◼</span></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #c084fc;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🟣 Reversal</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#c084fc;">{pva_counts.get("Watch reversal", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▼ Price ▼</span></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="glass-card" style="padding:10px 12px; border-left:3px solid #ef4444;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">🔴 Bearish</p><h3 style="font-size:1.4rem; margin:2px 0 0 0; color:#ef4444;">{pva_counts.get("Bearish", 0)}</h3><span style="font-size:0.68rem; color:#64748b;">Vol ▲ Price ▼</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter controls for matrix
    f_col1, f_col2, f_col3 = st.columns([3, 3, 4])
    with f_col1:
        signal_filter_options = [
            f"All ({len(vpa_data)})",
            f"🟢 Bullish ({pva_counts.get('Bullish', 0)})",
            f"🔵 Watch ({pva_counts.get('Watch', 0)})",
            f"🟡 Caution ({pva_counts.get('Caution', 0)})",
            f"⚪ Neutral ({pva_counts.get('Neutral', 0)})",
            f"🟣 Watch reversal ({pva_counts.get('Watch reversal', 0)})",
            f"🔴 Bearish ({pva_counts.get('Bearish', 0)})"
        ]
        chosen_filter = st.selectbox("Filter by Condition", signal_filter_options, key="pva_signal_filter_sel")
    with f_col2:
        pva_sort = st.selectbox("Sort Matrix By", ["Blast Score ↓", "Volume Ratio ↓", "Change % ↓", "CMP ↑"], key="pva_matrix_sort_sel")
    with f_col3:
        pva_search = st.text_input("🔍 Search Stock Symbol", "", key="pva_sym_search_input", placeholder="e.g. RELIANCE, TCS...")

    # Filter logic
    selected_sig_name = None
    if "Bullish" in chosen_filter:
        selected_sig_name = "Bullish"
    elif "Watch reversal" in chosen_filter:
        selected_sig_name = "Watch reversal"
    elif "Watch" in chosen_filter:
        selected_sig_name = "Watch"
    elif "Caution" in chosen_filter:
        selected_sig_name = "Caution"
    elif "Neutral" in chosen_filter:
        selected_sig_name = "Neutral"
    elif "Bearish" in chosen_filter:
        selected_sig_name = "Bearish"

    filtered_pva_stocks = []
    for r in vpa_data:
        pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
        if selected_sig_name and pva['signal'] != selected_sig_name:
            continue
        if pva_search and pva_search.strip().upper() not in r.get('symbol', '').upper():
            continue
        filtered_pva_stocks.append(r)

    # Sort
    if pva_sort == "Blast Score ↓":
        filtered_pva_stocks.sort(key=lambda x: -x.get('blast_score', 0))
    elif pva_sort == "Volume Ratio ↓":
        filtered_pva_stocks.sort(key=lambda x: -x.get('vol_ratio_5d', 1.0))
    elif pva_sort == "Change % ↓":
        filtered_pva_stocks.sort(key=lambda x: -x.get('day_change_pct', 0.0))
    elif pva_sort == "CMP ↑":
        filtered_pva_stocks.sort(key=lambda x: x.get('cmp', 0.0))

    # Render PVA Table
    pva_rows = []
    for idx, r in enumerate(filtered_pva_stocks[:150], start=1):
        pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
        sym = r.get('symbol', '')
        tv_link = f"https://in.tradingview.com/chart/?symbol=NSE:{sym.replace('.NS','')}"
        vol_mov_badge = "<span style='color:#00e676; font-weight:600;'>▲ Increasing</span>" if pva['vol_movement'] == "Increasing" else "<span style='color:#94a3b8; font-weight:600;'>▼ Decreasing</span>"
        if pva['price_movement'] == "Increasing":
            price_mov_badge = "<span style='color:#00e676; font-weight:600;'>▲ Increasing</span>"
        elif pva['price_movement'] == "Decreasing":
            price_mov_badge = "<span style='color:#ef4444; font-weight:600;'>▼ Decreasing</span>"
        else:
            price_mov_badge = "<span style='color:#38bdf8; font-weight:600;'>◼ Stable</span>"

        sig_badge = f"<span style='background:rgba(255,255,255,0.06); color:{pva['color']}; border:1px solid {pva['color']}; padding:3px 8px; border-radius:4px; font-weight:bold;'>{pva['signal']}</span>"
        blast_badge = f"<span style='color:{blast_tier_color(r.get('blast_tier',''))}; font-weight:bold;'>{r.get('blast_score', 0)} ({r.get('blast_tier','')})</span>"

        pva_rows.append(f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:9px 8px; color:#94a3b8; font-weight:bold;">#{idx}</td>
<td style="padding:9px 8px;"><a href="{tv_link}" target="_blank" style="color:#00e5ff; font-weight:bold; text-decoration:none;">{sym}</a></td>
<td style="padding:9px 8px; color:#e2e8f0;">₹{r.get('cmp', 0.0):,.2f}</td>
<td style="padding:9px 8px;">{get_day_change_badge_html(r.get('day_change_pct', 0.0))}</td>
<td style="padding:9px 8px; color:{'#00e676' if r.get('vol_ratio_5d',1.0)>=1.0 else '#94a3b8'};">{r.get('vol_ratio_5d', 1.0):.2f}x</td>
<td style="padding:9px 8px;">{vol_mov_badge}</td>
<td style="padding:9px 8px;">{price_mov_badge}</td>
<td style="padding:9px 8px;">{sig_badge}</td>
<td style="padding:9px 8px; font-size:0.83rem; color:#cbd5e1; max-width:320px;">{pva['interpretation']}</td>
<td style="padding:9px 8px;">{blast_badge}</td>
</tr>""")

    pva_table_html = f"""<div style="overflow-x:auto; margin-top:8px;">
<table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
<thead>
<tr style="background:rgba(56,189,248,0.08); border-bottom:2px solid rgba(56,189,248,0.3);">
<th style="padding:10px 8px; color:#38bdf8;">#</th>
<th style="padding:10px 8px; color:#38bdf8;">Symbol</th>
<th style="padding:10px 8px; color:#38bdf8;">CMP</th>
<th style="padding:10px 8px; color:#38bdf8;">Chg%</th>
<th style="padding:10px 8px; color:#38bdf8;">Vol Ratio</th>
<th style="padding:10px 8px; color:#38bdf8;">Volume Movement</th>
<th style="padding:10px 8px; color:#38bdf8;">Price Movement</th>
<th style="padding:10px 8px; color:#38bdf8;">Signal</th>
<th style="padding:10px 8px; color:#38bdf8;">Automatic Interpretation</th>
<th style="padding:10px 8px; color:#38bdf8;">Blast Score</th>
</tr>
</thead>
<tbody>
{"".join(pva_rows)}
</tbody>
</table>
</div>"""
    st.markdown(pva_table_html, unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.78rem; color:#64748b; margin-top:6px;'>Showing {len(filtered_pva_stocks[:150])} of {len(filtered_pva_stocks)} matching stocks</p>", unsafe_allow_html=True)

    # Excel and CSV Download buttons
    st.markdown("<br>", unsafe_allow_html=True)
    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        try:
            excel_bytes = generate_pva_excel_bytes(vpa_data)
            st.download_button(
                label="📥 Download 6-Condition Matrix (Excel .xlsx)",
                data=excel_bytes,
                file_name=f"price_volume_matrix_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Includes both the Price_Volume_Matrix sheet and the Market_Conditions_Legend sheet, ready for Google Sheets or Microsoft Excel."
            )
        except Exception as e:
            st.error(f"Excel export error: {e}")

    with dl_c2:
        csv_rows = []
        for r in vpa_data:
            pva = r.get('pva') or classify_pva(r.get('day_change_pct', 0.0), r.get('vol_ratio_5d', 1.0), r.get('vol_rising'))
            csv_rows.append({
                'Symbol': r.get('symbol', ''),
                'Company Name': r.get('company_name', ''),
                'CMP': r.get('cmp', 0.0),
                'Change %': round(r.get('day_change_pct', 0.0), 2),
                'Volume Ratio': round(r.get('vol_ratio_5d', 1.0), 2),
                'Volume Movement': pva['vol_movement'],
                'Price Movement': pva['price_movement'],
                'Signal': pva['signal'],
                'Automatic Interpretation': pva['interpretation'],
                'Blast Score': r.get('blast_score', 0),
                'Blast Tier': r.get('blast_tier', '')
            })
        st.download_button(
            label="📥 Download Matrix for Google Sheets (CSV)",
            data=pd.DataFrame(csv_rows).to_csv(index=False).encode('utf-8-sig'),
            file_name=f"price_volume_matrix_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Plain CSV compatible with Google Sheets import"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: FULL VPA TABLE (all stocks)
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Full VPA Results — All Stocks (Multi-Timeframe)")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 4])
    with ctrl1:
        selected_tf = st.selectbox("Timeframe", ["Daily", "Weekly", "Monthly"], key="vpa_tf_select")
    with ctrl2:
        only_buy_signals = st.checkbox("🟢 Buy Signals Only", value=False)
    with ctrl3:
        min_blast = st.slider("Min Blast Score", min_value=0, max_value=100, value=0, step=5,
                               help="Filter to show only stocks with Blast Score ≥ this value")

    sort_col, _ = st.columns([3, 7])
    with sort_col:
        sort_by = st.selectbox("Sort By", ["Blast Score ↓", "VPA Score ↓", "Volume Ratio ↓", "CMP ↑", "Change % ↓"], key="vpa_sort_select")

    sort_map = {
        "Blast Score ↓": lambda x: -x.get('blast_score', 0),
        "VPA Score ↓":   lambda x: -x.get('score', 0),
        "Volume Ratio ↓":lambda x: -x.get('vol_ratio_5d', 1.0),
        "CMP ↑":         lambda x: x.get('cmp', 0),
        "Change % ↓":    lambda x: -x.get('day_change_pct', 0),
    }
    sorted_vpa = sorted(vpa_data, key=sort_map.get(sort_by, sort_map["Blast Score ↓"]))

    daily_export, weekly_export, monthly_export = [], [], []
    html_rows = []
    rank = 0

    for r in sorted_vpa:
        d = r['daily']; w = r['weekly']; m = r['monthly']

        if selected_tf == "Weekly":
            tf_data = w
        elif selected_tf == "Monthly":
            tf_data = m
        else:
            tf_data = d

        tf_sig = get_signal(tf_data['minor'], tf_data['mid'], tf_data['major'], tf_data.get('major_val', 0))
        if only_buy_signals and tf_sig != "Buy":
            continue

        blast_score = r.get('blast_score', 0)
        if blast_score < min_blast:
            continue

        rank += 1
        sym = r['symbol']
        tv_link = f"https://in.tradingview.com/chart/?symbol=NSE:{sym.replace('.NS','')}"
        tier_color = blast_tier_color(r.get('blast_tier', ''))

        html_rows.append(f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:9px 8px; color:#94a3b8; font-weight:bold;">#{rank}</td>
<td style="padding:9px 8px;"><a href="{tv_link}" target="_blank" style="color:#00e5ff; font-weight:bold; text-decoration:none;">{sym}</a></td>
<td style="padding:9px 8px; color:#e2e8f0;">₹{r['cmp']:,.2f}</td>
<td style="padding:9px 8px;">{get_day_change_badge_html(r['day_change_pct'])}</td>
<td style="padding:9px 8px; font-weight:bold; color:{tier_color};">{blast_score}</td>
<td style="padding:9px 8px; border-left:1px solid rgba(255,255,255,0.08);">{trend_badge(tf_data['minor'])}</td>
<td style="padding:9px 8px;">{trend_badge(tf_data['mid'])}</td>
<td style="padding:9px 8px;">{trend_badge(tf_data['major'])}</td>
<td style="padding:9px 8px; border-left:1px solid rgba(255,255,255,0.08);">{action_badge(tf_data['minor'], tf_data['mid'], tf_data['major'], tf_data.get('major_val',0), rsi=tf_data.get('rsi',0), cci=tf_data.get('cci',0))}</td>
<td style="padding:9px 8px; color:#a3e635; font-weight:bold;">{r.get('score', 0)}</td>
<td style="padding:9px 8px; color:{'#00e676' if r.get('vol_ratio_5d',1)>=1.5 else '#fbbf24'};">{r.get('vol_ratio_5d',1.0):.1f}x</td>
<td style="padding:9px 8px;">{r.get('confidence','N/A')}</td>
</tr>""")

        for tf_key, tf_d, csv_list in [("daily", d, daily_export), ("weekly", w, weekly_export), ("monthly", m, monthly_export)]:
            csv_list.append({
                'Rank': rank, 'Symbol': sym, 'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                'Major Trend': "Up" if tf_d['major']==1 else ("Down" if tf_d['major']==-1 else "Neutral"),
                'Mid Trend': "Up" if tf_d['mid']==1 else ("Down" if tf_d['mid']==-1 else "Neutral"),
                'Minor Trend': "Up" if tf_d['minor']==1 else ("Down" if tf_d['minor']==-1 else "Neutral"),
                'RSI': tf_d.get('rsi', 0.0), 'CCI': tf_d.get('cci', 0.0),
                'Action': get_action_signal_text(tf_d['minor'], tf_d['mid'], tf_d['major'], tf_d.get('major_val',0), rsi=tf_d.get('rsi',0), cci=tf_d.get('cci',0)),
                'Signal': get_signal(tf_d['minor'], tf_d['mid'], tf_d['major'], tf_d.get('major_val',0)),
                'Blast Score': blast_score, 'Vol Ratio 5d': r.get('vol_ratio_5d', 1.0),
                'VPA Score': r.get('score', 0), 'Confidence': r.get('confidence', 'N/A')
            })

    st.markdown("<br>", unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("📥 Daily VPA (CSV)", data=pd.DataFrame(daily_export).to_csv(index=False).encode('utf-8-sig'),
                           file_name="vpa_daily_trend.csv", mime="text/csv", width="stretch")
    with dl2:
        st.download_button("📥 Weekly VPA (CSV)", data=pd.DataFrame(weekly_export).to_csv(index=False).encode('utf-8-sig'),
                           file_name="vpa_weekly_trend.csv", mime="text/csv", width="stretch")
    with dl3:
        st.download_button("📥 Monthly VPA (CSV)", data=pd.DataFrame(monthly_export).to_csv(index=False).encode('utf-8-sig'),
                           file_name="vpa_monthly_trend.csv", mime="text/csv", width="stretch")

    st.caption(f"Showing **{rank}** stocks | {selected_tf} Timeframe | Min Blast Score: {min_blast}")

    if not html_rows:
        st.info("No stocks match the current filters.")
    else:
        table_html = f"""<div style="overflow-x:auto; margin-top:10px;">
<table style="width:100%; text-align:left; border-collapse:collapse; font-size:0.88rem;">
<thead>
<tr style="background:rgba(255,255,255,0.05); border-bottom:1px solid rgba(255,255,255,0.1);">
<th style="padding:9px 8px;">Rank</th>
<th style="padding:9px 8px;">Symbol</th>
<th style="padding:9px 8px;">CMP</th>
<th style="padding:9px 8px;">Chg%</th>
<th style="padding:9px 8px; color:#ff9900;">🔥 Blast</th>
<th style="padding:9px 8px; border-left:1px solid rgba(255,255,255,0.08);">Short</th>
<th style="padding:9px 8px;">Mid</th>
<th style="padding:9px 8px;">Major</th>
<th style="padding:9px 8px; border-left:1px solid rgba(255,255,255,0.08);">Action / Signal</th>
<th style="padding:9px 8px;">VPA Score</th>
<th style="padding:9px 8px;">Vol Ratio</th>
<th style="padding:9px 8px;">Confidence</th>
</tr>
</thead>
<tbody>
{"".join(html_rows)}
</tbody>
</table>
</div>"""
        st.markdown(table_html, unsafe_allow_html=True)
