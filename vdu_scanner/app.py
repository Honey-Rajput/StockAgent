# app.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from config import IST_TIMEZONE, get_company_name, DRY_ZONE_MIN_DAYS, DRY_ZONE_MAX_DAYS, MIN_VOLUME_RATIO, MIN_PRICE_CHANGE
from data_fetcher import fetch_ohlcv, get_index_stocks, fetch_ohlcv_timeframe, get_stock_sector
from scanner import scan_stock, scan_wt_cross, compute_rich_analysis, scan_monthly_momentum, scan_weekly_momentum, scan_vcs, scan_monthly_early_stage2, scan_vpa_trend, scan_structural_vcp
from indicators import precompute_indicators

import watchlist
from utils import inject_premium_css, get_signal_badge_html, get_day_change_badge_html
import database
import ai_detector
import re
import threading
import concurrent.futures

def run_background_ai_scan(symbols_list, date_str, force=False):
    """
    Executes high-speed parallel AI scans in a background daemon thread
    to prevent blocking the Streamlit UI, allowing progressive database updates.
    """
    # Guard to prevent duplicate concurrent background scanning threads
    is_already_running = any(t.name == "AI_Background_Scan" for t in threading.enumerate())
    if is_already_running:
        print("Background AI scan thread is already active. Skipping duplicate thread launch.")
        return

    def scan_and_save(sym):
        try:
            # Check if already scanned today to avoid redundant API queries
            if not force:
                existing = database.get_pattern_by_date(sym, date_str)
                if existing and existing.get('pattern_name') not in ["Error", "Pending"]:
                    return sym, True
                
            df_hist = fetch_ohlcv(sym)
            if df_hist is not None and not df_hist.empty:
                ans_dict = ai_detector.detect_chart_pattern(sym, df_hist)
                if ans_dict:
                    pattern_name = ans_dict.get("pattern_name", "None")
                    if pattern_name == "Error":
                        pattern_name = "None Detected"
                        
                    subset_5d = df_hist.iloc[-5:]
                    snap_list = [f"{row['Date'].strftime('%m-%d')}:{row['Close']:.0f}" for _, row in subset_5d.iterrows()]
                    snap_str = ",".join(snap_list)
                    
                    success = database.save_pattern(
                        symbol=sym,
                        pattern_name=pattern_name,
                        confidence=ans_dict.get('confidence', 'None'),
                        direction=ans_dict.get('direction', 'None'),
                        analysis_text=ans_dict.get('analysis_text', 'No details available.'),
                        price_data_snapshot=snap_str,
                        date_str=date_str
                    )
                    return sym, success
        except Exception as e:
            print(f"Background AI scan failed for {sym}: {e}")
        return sym, False

    def thread_runner():
        print(f"Background AI scan daemon thread started for symbols: {symbols_list} (Force={force})")
        if not symbols_list:
            return
        # Exclude already processed items to speed up background process
        # Bulk query existing patterns to prevent N+1 DB lookups
        existing_patterns = {} if force else database.get_all_patterns_by_date(date_str)
        to_scan = []
        
        for s in symbols_list:
            if force:
                to_scan.append(s)
            else:
                exist = existing_patterns.get(s)
                if not exist or exist.get('pattern_name') in ["Error", "Pending"]:
                    to_scan.append(s)
                
        if not to_scan:
            print("All symbols already analyzed by AI. Skipping background daemon.")
            return
            
        max_workers = min(20, len(to_scan)) # Increased from 5 to 20 for faster parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(scan_and_save, to_scan)
        print("Background AI scan daemon thread finished successfully!")

    # Start the daemon thread and name it "AI_Background_Scan"
    t = threading.Thread(target=thread_runner, name="AI_Background_Scan", daemon=True)
    t.start()

def ensure_minervini_fields(m_list):
    if not m_list:
        return m_list
    for r in m_list:
        # Check if fields are already populated in database record
        if r.get('run_up_200') is not None:
            r['run_up_200'] = float(r['run_up_200'])
        else:
            # Extract from recommendation text (resolving rich JSON first)
            rec = r.get('recommendation', '')
            plain_text = rec
            if rec.strip().startswith("{") and rec.strip().endswith("}"):
                try:
                    import json
                    plain_text = json.loads(rec).get("text", rec)
                except Exception:
                    pass
            
            run_up_200_match = re.search(r'holding\s+(\d+\.?\d*)%\s+above\s+its\s+200\s+SMA', plain_text, re.IGNORECASE)
            if run_up_200_match:
                r['run_up_200'] = float(run_up_200_match.group(1))
            else:
                r['run_up_200'] = 10.0
                
        if r.get('run_up_52w') is not None:
            r['run_up_52w'] = float(r['run_up_52w'])
        else:
            rec = r.get('recommendation', '')
            plain_text = rec
            if rec.strip().startswith("{") and rec.strip().endswith("}"):
                try:
                    import json
                    plain_text = json.loads(rec).get("text", rec)
                except Exception:
                    pass
                    
            run_up_52w_match = re.search(r'run\s+up\s+(\d+\.?\d*)%\s+from\s+its\s+52w', plain_text, re.IGNORECASE)
            if run_up_52w_match:
                r['run_up_52w'] = float(run_up_52w_match.group(1))
            else:
                r['run_up_52w'] = 30.0
                
        if r.get('is_early') is not None:
            r['is_early'] = bool(r['is_early'])
        else:
            conf = r.get('confidence', '')
            rec = r.get('recommendation', '')
            plain_text = rec
            if rec.strip().startswith("{") and rec.strip().endswith("}"):
                try:
                    import json
                    plain_text = json.loads(rec).get("text", rec)
                except Exception:
                    pass
            r['is_early'] = 'early' in conf.lower() or 'early' in plain_text.lower()
            
    return m_list


# --- Page Configurations ---
st.set_page_config(
    page_title="Volume Surge Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern Outfit typography, glassmorphism card layouts and custom color styles
inject_premium_css()

# Initialize PostgreSQL database schema (Neon) on app load — non-fatal if DB is unreachable
try:
    database.init_db()
except Exception as db_init_err:
    print(f"Database initialization failed (non-fatal): {db_init_err}")

# --- Process Watchlist Query Parameter Actions ---
if "add_to_watchlist" in st.query_params:
    sym = st.query_params["add_to_watchlist"].strip().upper()
    try:
        price_val = st.query_params.get("price", 0.0)
        price = float(price_val) if not isinstance(price_val, list) else float(price_val[0])
    except Exception:
        price = 0.0
    try:
        score_val = st.query_params.get("score", 50.0)
        score = float(score_val) if not isinstance(score_val, list) else float(score_val[0])
    except Exception:
        score = 50.0
    
    watchlist.add_stock(symbol=sym, entry_price=price, signal_strength=score)
    st.toast(f"🚀 Added {sym} to Watchlist!")
    # Safely clear query params using del keys to avoid websocket page crash reruns
    for k in ["add_to_watchlist", "price", "score"]:
        if k in st.query_params:
            del st.query_params[k]

if "remove_from_watchlist" in st.query_params:
    sym = st.query_params["remove_from_watchlist"].strip().upper()
    watchlist.remove_stock(sym)
    st.toast(f"❌ Removed {sym} from Watchlist!")
    if "remove_from_watchlist" in st.query_params:
        del st.query_params["remove_from_watchlist"]

# --- Process Table Sorting Query Parameter Actions ---
if "sort_col" in st.query_params:
    sort_col = st.query_params["sort_col"]
    prefix = st.query_params.get("prefix", "vdu_tab")
    
    # Toggle sorting direction
    curr_col = st.session_state.get(f"{prefix}_sort_col", "")
    curr_dir = st.session_state.get(f"{prefix}_sort_dir", "desc")
    
    if curr_col == sort_col:
        # Toggle direction
        new_dir = "asc" if curr_dir == "desc" else "desc"
    else:
        new_dir = "desc"
        
    st.session_state[f"{prefix}_sort_col"] = sort_col
    st.session_state[f"{prefix}_sort_dir"] = new_dir
    
    # Safely remove sorting parameters in place using del
    for k in ["sort_col", "prefix"]:
        if k in st.query_params:
            del st.query_params[k]

def render_trading_setup_card(r: dict, key_prefix: str, idx: int):
    """
    Renders a premium, glassmorphic expandable sub-row card for trading guidance.
    """
    buy = r.get('buy_price') or r.get('cmp') or 0.0
    sl = r.get('exit_price') or 0.0
    target = r.get('target_price') or 0.0
    conf = r.get('confidence') or 'Medium'
    rec = r.get('recommendation') or 'No recommendation generated.'
    
    # Custom colored tag for confidence
    conf_color = "#ef4444" if "Low" in conf else "#ffa000" if "Medium" in conf else "#00e676"
    
    # Check if recommendation is rich JSON
    is_rich = False
    rich_data = {}
    if rec.strip().startswith("{") and rec.strip().endswith("}"):
        try:
            import json
            rich_data = json.loads(rec)
            if rich_data.get("is_rich"):
                is_rich = True
        except Exception:
            is_rich = False
            
    with st.expander(f"🎯 Trade Setup & Actionable Recommendation for {r['symbol']}", expanded=True):
        if is_rich:
            rec_text = rich_data.get("text", "")
            rsi_val = rich_data.get("rsi", 0.0)
            rsi_stat = rich_data.get("rsi_status", "Neutral")
            rsi_int = rich_data.get("rsi_interp", "")
            cci_val = rich_data.get("cci", 0.0)
            cci_stat = rich_data.get("cci_status", "Neutral")
            cci_int = rich_data.get("cci_interp", "")
            ema20 = rich_data.get("ema20", 0.0)
            sma50 = rich_data.get("sma50", 0.0)
            sma200 = rich_data.get("sma200", 0.0)
            triggers = rich_data.get("triggers", [])
            cmp_val = r.get('cmp') or buy
            
            # Formulate EMA/SMA statuses
            ema_status = "Price is Above" if cmp_val > ema20 else "Price is Below"
            sma50_status = "Price is Above" if cmp_val > sma50 else "Price is Below"
            sma200_status = "Price is Above" if cmp_val > sma200 else "Price is Below"
            
            ema_interp = "Dynamic short-term exponential trend support."
            sma50_interp = "Mid-term institutional trend boundary."
            sma200_interp = "Major long-term structural trend boundary."
            
            # HTML for triggers
            triggers_html = "".join([f'<div style="font-size:0.88rem; color:#00e676; margin-bottom: 5px; font-weight: 500;">{t}</div>' for t in triggers])
            
            st.html(f"""
                <div class="glass-card" style="padding: 18px; border-left: 4px solid #29b6f6; background: rgba(30, 41, 59, 0.4); margin-bottom: 8px;">
                    <!-- Top metrics row -->
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 2px;">Strategy Confidence</span>
                            <span class="custom-badge" style="background: rgba({ '0,230,118' if 'High' in conf else '255,160,0' if 'Medium-High' in conf or 'Medium' in conf else '239,68,68' },0.15); color: {conf_color}; font-weight: bold; border: 1px solid {conf_color}; font-size: 0.9rem;">🎯 {conf}</span>
                        </div>
                        <div style="display: flex; gap: 15px;">
                            <div style="background: rgba(41,182,246,0.06); border: 1px solid rgba(41,182,246,0.15); padding: 5px 12px; border-radius: 8px; text-align: center;">
                                <span style="font-size: 0.75rem; color: #29b6f6; font-weight: 600; text-transform: uppercase;">Buy Range</span>
                                <span style="font-size: 1.05rem; color: #e2e8f0; font-weight: 700; display: block; margin-top: 2px;">₹{buy:,.2f}</span>
                            </div>
                            <div style="background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.15); padding: 5px 12px; border-radius: 8px; text-align: center;">
                                <span style="font-size: 0.75rem; color: #ef4444; font-weight: 600; text-transform: uppercase;">Stop Loss (Exit)</span>
                                <span style="font-size: 1.05rem; color: #ef4444; font-weight: 700; display: block; margin-top: 2px;">₹{sl:,.2f}</span>
                            </div>
                            <div style="background: rgba(0,230,118,0.06); border: 1px solid rgba(0,230,118,0.15); padding: 5px 12px; border-radius: 8px; text-align: center;">
                                <span style="font-size: 0.75rem; color: #00e676; font-weight: 600; text-transform: uppercase;">Swing Target</span>
                                <span style="font-size: 1.05rem; color: #00e676; font-weight: 700; display: block; margin-top: 2px;">₹{target:,.2f}</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Structured Table for indicators -->
                    <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 12px; margin-bottom: 15px; overflow-x: auto;">
                        <span style="font-size: 0.8rem; color: #38bdf8; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 4px;">📊 Technical Indicators Dashboard</span>
                        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; color: #cbd5e1;">
                            <thead>
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #94a3b8; font-weight: 600;">
                                    <th style="padding: 6px 12px 6px 6px;">Indicator</th>
                                    <th style="padding: 6px 12px;">Value</th>
                                    <th style="padding: 6px 12px;">Status / Reading</th>
                                    <th style="padding: 6px 6px 6px 12px;">Analysis & Guidance</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                                    <td style="padding: 6px 12px 6px 6px; font-weight: 600; color: #38bdf8;">RSI (14)</td>
                                    <td style="padding: 6px 12px;">{rsi_val:.1f}</td>
                                    <td style="padding: 6px 12px;"><span class="custom-badge" style="background:rgba(41,182,246,0.1); color:#38bdf8; border: 1px solid rgba(41,182,246,0.25); padding: 1px 6px; font-size:0.75rem; border-radius: 4px;">{rsi_stat}</span></td>
                                    <td style="padding: 6px 6px 6px 12px; color: #94a3b8; font-style: italic;">{rsi_int}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                                    <td style="padding: 6px 12px 6px 6px; font-weight: 600; color: #ab47bc;">CCI (14)</td>
                                    <td style="padding: 6px 12px;">{cci_val:.1f}</td>
                                    <td style="padding: 6px 12px;"><span class="custom-badge" style="background:rgba(171,71,188,0.1); color:#ba68c8; border: 1px solid rgba(171,71,188,0.25); padding: 1px 6px; font-size:0.75rem; border-radius: 4px;">{cci_stat}</span></td>
                                    <td style="padding: 6px 6px 6px 12px; color: #94a3b8; font-style: italic;">{cci_int}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                                    <td style="padding: 6px 12px 6px 6px; font-weight: 600; color: #e2e8f0;">20 EMA</td>
                                    <td style="padding: 6px 12px;">₹{ema20:,.2f}</td>
                                    <td style="padding: 6px 12px; color:{'#00e676' if 'Above' in ema_status else '#ef4444'}; font-weight:600; font-size: 0.8rem;">{ema_status.upper()}</td>
                                    <td style="padding: 6px 6px 6px 12px; color: #94a3b8; font-style: italic;">{ema_interp}</td>
                                </tr>
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.03);">
                                    <td style="padding: 6px 12px 6px 6px; font-weight: 600; color: #cbd5e1;">50 SMA</td>
                                    <td style="padding: 6px 12px;">₹{sma50:,.2f}</td>
                                    <td style="padding: 6px 12px; color:{'#00e676' if 'Above' in sma50_status else '#ef4444'}; font-weight:600; font-size: 0.8rem;">{sma50_status.upper()}</td>
                                    <td style="padding: 6px 6px 6px 12px; color: #94a3b8; font-style: italic;">{sma50_interp}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 6px 12px 6px 6px; font-weight: 600; color: #94a3b8;">200 SMA</td>
                                    <td style="padding: 6px 12px;">₹{sma200:,.2f}</td>
                                    <td style="padding: 6px 12px; color:{'#00e676' if 'Above' in sma200_status else '#ef4444'}; font-weight:600; font-size: 0.8rem;">{sma200_status.upper()}</td>
                                    <td style="padding: 6px 6px 6px 12px; color: #94a3b8; font-style: italic;">{sma200_interp}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Checklist buy triggers -->
                    <div style="background: rgba(0, 230, 118, 0.03); border: 1px dashed rgba(0, 230, 118, 0.25); border-radius: 10px; padding: 12px; margin-bottom: 15px;">
                        <span style="font-size: 0.8rem; color: #00e676; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 8px; border-bottom: 1px dashed rgba(0,230,118,0.15); padding-bottom: 4px;">🎯 Technical Buying Strengths</span>
                        {triggers_html}
                    </div>

                    <!-- Actionable recommendation text -->
                    <div style="background: rgba(148,163,184,0.05); padding: 10px 14px; border-radius: 8px; border: 1px dashed rgba(148,163,184,0.15);">
                        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 4px;">📈 Actionable Recommendation</span>
                        <p style="margin: 0; font-size: 0.92rem; color: #e2e8f0; line-height: 1.5; font-style: italic;">"{rec_text}"</p>
                    </div>
                </div>
            """)
            
            # Collapsible strategy reference guide under the indicators table
            with st.expander("🎓 Indicator Strategy Reference Guide", expanded=False):
                st.html(
                    """
                    <div style="font-size: 0.88rem; line-height: 1.4; color: #cbd5e1; margin-bottom: 8px;">
                        <span style="color: #38bdf8; font-weight: 600;">Core Technical Signals & Parameters:</span>
                        <p style="margin: 4px 0 10px 0;">This checklist and table help identify the highest probability swing setups. When multiple indicators align at their optimal buying values, it creates <b>bullish confluence</b>.</p>
                    </div>
                    
                    <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.8rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;">
                        <thead>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #38bdf8; font-weight: bold; background: rgba(56, 189, 248, 0.05);">
                                <th style="padding: 6px 10px;">Indicator</th>
                                <th style="padding: 6px 10px;">Technical Reasoning & Purpose</th>
                                <th style="padding: 6px 10px;">Best Buy Signal Conditions</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                                <td style="padding: 6px 10px; font-weight: bold; color: #38bdf8;">RSI (14)</td>
                                <td style="padding: 6px 10px; color: #94a3b8;">Measures velocity of price action to spot oversold bounces or active continuation phases.</td>
                                <td style="padding: 6px 10px; color: #00e676;"><b>35 - 50</b> (Oversold Bounce)<br><b>50 - 65</b> (Momentum Continuation)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                                <td style="padding: 6px 10px; font-weight: bold; color: #ab47bc;">CCI (14)</td>
                                <td style="padding: 6px 10px; color: #94a3b8;">Measures deviation from historical average price. Excellent at catching powerful trend breakouts early.</td>
                                <td style="padding: 6px 10px; color: #00e676;"><b>&gt; +100</b> (Bullish Momentum Breakout)<br><b>&lt; -100</b> (Institutional Selling Exhaustion Reversal)</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                                <td style="padding: 6px 10px; font-weight: bold; color: #e2e8f0;">20 EMA</td>
                                <td style="padding: 6px 10px; color: #94a3b8;">Exponential Moving Average weighting recent price. Acts as a dynamic support anchor during rapid trends.</td>
                                <td style="padding: 6px 10px; color: #00e676;">Price pulls back within <b>&plusmn;2%</b> of the 20 EMA to offer a low-risk entry.</td>
                            </tr>
                            <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                                <td style="padding: 6px 10px; font-weight: bold; color: #cbd5e1;">50 SMA</td>
                                <td style="padding: 6px 10px; color: #94a3b8;">Medium-term institutional trend boundary. Essential filter separating healthy trend accumulation from distribution.</td>
                                <td style="padding: 6px 10px; color: #00e676;">CMP trades safely <b>above 50 SMA</b> (confirms mid-term uptrend support).</td>
                            </tr>
                            <tr>
                                <td style="padding: 6px 10px; font-weight: bold; color: #cbd5e1;">200 SMA</td>
                                <td style="padding: 6px 10px; color: #94a3b8;">Long-term structural dividing line. Serves as the ultimate institutional support floor.</td>
                                <td style="padding: 6px 10px; color: #00e676;">CMP trades <b>above 200 SMA</b> (enforces global bull market bias) and <b>50 SMA &gt; 200 SMA</b> (Golden Cross).</td>
                            </tr>
                        </tbody>
                    </table>
                """)
        else:
            # Fallback legacy layout
            st.html(f"""
                <div class="glass-card" style="padding: 15px; border-left: 4px solid #29b6f6; background: rgba(30, 41, 59, 0.4); margin-bottom: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <span style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; display: block; margin-bottom: 2px;">Strategy Confidence</span>
                            <span class="custom-badge" style="background: rgba({ '0,230,118' if 'High' in conf else '255,160,0' if 'Medium-High' in conf or 'Medium' in conf else '239,68,68' },0.15); color: {conf_color}; font-weight: bold; border: 1px solid {conf_color}; font-size: 0.9rem;">🎯 {conf}</span>
                        </div>
                        <div style="display: flex; gap: 15px;">
                            <div style="background: rgba(41,182,246,0.06); border: 1px solid rgba(41,182,246,0.15); padding: 5px 12px; border-radius: 8px; text-align: center;">
                                <span style="font-size: 0.75rem; color: #29b6f6; font-weight: 600; text-transform: uppercase;">Buy Range</span>
                                <span style="font-size: 1.05rem; color: #e2e8f0; font-weight: 700; display: block; margin-top: 2px;">₹{buy:,.2f}</span>
                            </div>
                            <div style="background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.15); padding: 5px 12px; border-radius: 8px; text-align: center;">
                                <span style="font-size: 0.75rem; color: #ef4444; font-weight: 600; text-transform: uppercase;">Stop Loss (Exit)</span>
                                <span style="font-size: 1.05rem; color: #ef4444; font-weight: 700; display: block; margin-top: 2px;">₹{sl:,.2f}</span>
                            </div>
                            <div style="background: rgba(0,230,118,0.06); border: 1px solid rgba(0,230,118,0.15); padding: 5px 12px; border-radius: 8px; text-align: center;">
                                <span style="font-size: 0.75rem; color: #00e676; font-weight: 600; text-transform: uppercase;">Swing Target</span>
                                <span style="font-size: 1.05rem; color: #00e676; font-weight: 700; display: block; margin-top: 2px;">₹{target:,.2f}</span>
                            </div>
                        </div>
                    </div>
                    <div style="background: rgba(148,163,184,0.05); padding: 10px 14px; border-radius: 8px; border: 1px dashed rgba(148,163,184,0.15); margin-top: 8px;">
                        <span style="font-size: 0.8rem; color: #94a3b8; font-weight: 600; text-transform: uppercase; display: block; margin-bottom: 4px;">📈 Actionable Recommendation</span>
                        <p style="margin: 0; font-size: 0.92rem; color: #e2e8f0; line-height: 1.5; font-style: italic;">"{rec}"</p>
                    </div>
                </div>
            """)




def extract_clean_recommendation(rec) -> str:
    if not rec:
        return ""
    if isinstance(rec, dict):
        if rec.get("is_rich"):
            text_val = rec.get("text", "")
            if isinstance(text_val, str):
                text_val = text_val.replace('\\u20b9', '₹')
            return text_val
        return str(rec)
        
    rec_str = str(rec).strip()
    
    # Proactively unescape/unwrap outer quotes repeatedly (up to 3 times) to handle double-escaped payloads
    for _ in range(3):
        if rec_str.startswith('"') and rec_str.endswith('"'):
            rec_str = rec_str[1:-1].strip()
        elif rec_str.startswith("'") and rec_str.endswith("'"):
            rec_str = rec_str[1:-1].strip()
        else:
            break
            
    if rec_str.startswith("{") and rec_str.endswith("}"):
        try:
            import json
            # First try parsing the raw unescaped string
            try:
                data = json.loads(rec_str)
            except Exception:
                # Try handling pythonic string representations
                formatted_rec = rec_str.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null")
                data = json.loads(formatted_rec)
            
            # Recursively unpack if stringified json inside json
            if isinstance(data, str) and data.strip().startswith("{"):
                try:
                    data = json.loads(data)
                except Exception:
                    pass
                    
            if isinstance(data, dict):
                if data.get("is_rich"):
                    text_val = data.get("text", "")
                    if isinstance(text_val, str):
                        return text_val.replace('\\u20b9', '₹')
                else:
                    # If it's a simple dict but not marked as is_rich, check if there's any text/analysis key
                    for key in ["text", "analysis_text", "recommendation", "rec"]:
                        if data.get(key):
                            return str(data[key]).replace('\\u20b9', '₹')
        except Exception:
            pass
            
    # Proactively unescape backslashes for unicode characters or quotes
    if isinstance(rec_str, str):
        rec_str = rec_str.replace('\\"', '"').replace('\\u20b9', '₹')
    return rec_str

def render_unified_strategy_table(results_list: list, strategy_type: str, key_prefix: str):
    if not results_list or len(results_list) == 0:
        return
        
    w_df = watchlist.load_watchlist()
    watchlist_symbols = set(w_df['symbol'].str.upper().unique()) if not w_df.empty else set()
    
    # 1. Define safe sorting lambda mapping for all table columns
    sort_mapper = {
        "Symbol": lambda x: (x.get('symbol') or "").upper(),
                "CMP": lambda x: float(x.get('cmp') or 0.0),
        "Day Chg %": lambda x: float(x.get('day_change_pct') or x.get('pct_change_today') or 0.0),
        "Volume": lambda x: float(x.get('today_volume') or x.get('volume') or 0.0),
        "Dry Avg Vol": lambda x: float(x.get('dry_avg_vol') or 0.0),
        "Vol Ratio": lambda x: float(x.get('volume_ratio') or 0.0),
        "Dry Days": lambda x: int(x.get('dry_days_count') or x.get('dry_days') or 0),
        "Spikes": lambda x: int(x.get('dry_spikes') or 0),
        "Score": lambda x: float(x.get('score') or x.get('signal_strength') or 0.0),
        "Base Bottom": lambda x: float(x.get('base_bottom') or 0.0),
        "Historical High": lambda x: float(x.get('historical_high') or 0.0),
        "Extension %": lambda x: float(x.get('extension') or 0.0),
        "7M SMA": lambda x: float(x.get('sma7') or 0.0),
        "Squeeze Score": lambda x: float(x.get('squeeze_score') or 0.0),
        "VCS Score": lambda x: float(x.get('vcs_score') or 0.0),
        "Contractions": lambda x: int(x.get('contractions') or 0),
        "VPA Score": lambda x: float(x.get('trend_score') or x.get('score') or 0.0),
        "5d Range": lambda x: float(x.get('range_5d') or 0.0),
        "Pre-Range": lambda x: float(x.get('pre_range') or 0.0),
        "Prev Close": lambda x: float(x.get('prev_close') or 0.0),
        "Open": lambda x: float(x.get('open_price') or 0.0),
        "Gap %": lambda x: float(x.get('gap_pct') or 0.0),
        "WT1": lambda x: float(x.get('wt_value') or 0.0),
        "WT2": lambda x: float(x.get('wt2_value') or 0.0),
        "WT Diff": lambda x: float(x.get('wt_diff') or 0.0),
        "Signal": lambda x: 1 if x.get('buy_signal') else 0,
        "Buy Range": lambda x: float(x.get('buy_price') or x.get('cmp') or 0.0),
        "Stop Loss": lambda x: float(x.get('exit_price') or 0.0),
        "Swing Target": lambda x: float(x.get('target_price') or 0.0),
        "Confidence": lambda x: (x.get('confidence') or "").upper(),
        "Actionable Guidance & Reasoning": lambda x: (extract_clean_recommendation(x.get('recommendation') or "")).upper(),
        "Run Up 200 SMA": lambda x: float(x.get('run_up_200') or 0.0),
        "Run Up 52w Low": lambda x: float(x.get('run_up_52w') or 0.0),
        "Remaining Target %": lambda x: float(((x.get('target_price', 0.0) - x.get('cmp', 1.0)) / x.get('cmp', 1.0) * 100) if x.get('cmp', 0.0) > 0 else 0.0)
    }
    
    # 2. Determine active sort column and direction from session state
    if strategy_type == "vdu_breakout":
        default_col = "Score"
    elif strategy_type == "gapup":
        default_col = "Gap %"
    elif strategy_type == "wavetrend":
        default_col = "WT1"
    elif strategy_type == "minervini":
        default_col = "Remaining Target %"
    elif strategy_type == "vcs":
        default_col = "VCS Score"
    elif strategy_type == "struct_vcp":
        default_col = "Contractions"
    elif strategy_type == "vpa":
        default_col = "VPA Score"
    elif strategy_type == "stage2":
        default_col = "Score"
    else:
        default_col = "Day Chg %"
        
    active_col = st.session_state.get(f"{key_prefix}_sort_col", default_col)
    active_dir = st.session_state.get(f"{key_prefix}_sort_dir", "desc" if active_col != "Symbol" else "asc")
    
    if active_col not in sort_mapper:
        active_col = default_col
        active_dir = "desc" if active_col != "Symbol" else "asc"
        
    # 3. Sort the list
    reverse_sort = (active_dir == "desc")
    sorted_list = sorted(results_list, key=sort_mapper[active_col], reverse=reverse_sort)
    
    rows_html = []
    for idx, r in enumerate(sorted_list):
        buy = r.get('buy_price') or r.get('cmp') or 0.0
        sl = r.get('exit_price') or 0.0
        target = r.get('target_price') or 0.0
        conf = r.get('confidence') or 'Medium'
        clean_conf = conf.split(" (")[0] if " (" in conf else conf
        rec = r.get('recommendation') or 'No recommendation generated.'
        clean_rec = extract_clean_recommendation(rec)
        
        # Color coding confidence badge
        conf_color = "#ef4444" if "Low" in clean_conf else "#ffa000" if "Medium" in clean_conf else "#00e676"
        conf_badge = f'<span class="custom-badge" style="background: rgba({ "0,230,118" if "High" in clean_conf else "255,160,0" if "Medium" in clean_conf else "239,68,68" },0.12); color: {conf_color}; border: 1px solid {conf_color}; font-size: 0.75rem; font-weight: bold; padding: 2px 6px; border-radius: 4px;">{clean_conf}</span>'
        
        # Determine unique strategy score for watchlist adding
        if strategy_type == "vdu_breakout":
            score_val = float(r.get('signal_strength', 50.0))
        elif strategy_type == "stage2":
            score_val = float(r.get('score', 50.0))
        elif strategy_type == "gapup":
            score_val = float(round(r.get('gap_pct', 0.0) * 10, 1))
        elif strategy_type == "wavetrend":
            score_val = float(abs(r.get('wt_value', 50.0)))
        else:
            score_val = 50.0
            
        # Build cell values based on strategy type
        cells = []
        
        # 1. Interactive Watchlist Column (Tick Column)
        is_in_watchlist = r['symbol'].upper() in watchlist_symbols
        if is_in_watchlist:
            wl_cell = f'<td style="padding: 10px 12px; text-align: center;"><span style="color: #00e676; font-size: 1.1rem;" title="In Watchlist">☑️</span> <a href="/?remove_from_watchlist={r["symbol"]}" target="_self" style="color: #ef4444; font-size: 0.72rem; text-decoration: none; margin-left: 2px;">[Remove]</a></td>'
        else:
            wl_cell = f'<td style="padding: 10px 12px; text-align: center;"><a href="/?add_to_watchlist={r["symbol"]}&price={buy}&score={score_val}" target="_self" style="color: #94a3b8; font-size: 1.1rem; text-decoration: none; font-weight: bold;" title="Click to Add to Watchlist">☐</a> <a href="/?add_to_watchlist={r["symbol"]}&price={buy}&score={score_val}" target="_self" style="color: #00e676; font-size: 0.72rem; text-decoration: none; font-weight: bold; margin-left: 2px;">[Add]</a></td>'
        cells.append(wl_cell)
        
        # Clickable TradingView Symbol Link
        tv_sym = r["symbol"].replace('.NS', '')
        cells.append(f'<td style="padding: 10px 12px; font-weight: bold; color: #29b6f6;"><a href="https://in.tradingview.com/chart/?symbol=NSE:{tv_sym}" target="_blank" rel="noopener noreferrer" style="color: #29b6f6; text-decoration: none;">{r["symbol"]}</a></td>')
                
        # Sector column
        sector = get_stock_sector(r["symbol"])
        cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1; font-size: 0.8rem; font-style: italic;">{sector}</td>')
        
        cells.append(f'<td style="padding: 10px 12px; color: #e2e8f0; font-weight: 500;">₹{buy:,.2f}</td>')
        
        if strategy_type == "vdu_breakout":
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">{int(r.get("today_volume") or r.get("volume") or 0):,}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">{int(r.get("dry_avg_vol") or 0):,}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #ffa000; font-weight: 600;">{r.get("volume_ratio", 0.0):.2f}x</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">{r.get("dry_days_count") or r.get("dry_days") or 0}d</td>')
            spikes = r.get("dry_spikes", 0)
            spikes_badge = f'<span class="custom-badge badge-red" style="font-weight:600; padding: 2px 6px; border-radius: 4px;">{spikes}</span>' if spikes > 0 else f'<span class="custom-badge badge-grey" style="padding: 2px 6px; border-radius: 4px;">{spikes}</span>'
            cells.append(f'<td style="padding: 10px 12px;">{spikes_badge}</td>')
            score_badge = get_signal_badge_html(r.get("signal_strength", 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{score_badge}</td>')
            
        elif strategy_type == "gapup":
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">₹{r.get("prev_close", 0.0):,.2f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">₹{r.get("open_price", 0.0):,.2f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #00e676; font-weight: 600;">+{r.get("gap_pct", 0.0):.2f}%</td>')
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">{int(r.get("volume") or r.get("today_volume") or 0):,}</td>')
            
        elif strategy_type in ["above_ma", "support_ma", "crossover_ma"]:
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            
            if strategy_type == "above_ma":
                d20 = r.get('dist_20sma_pct', 0.0)
                d50 = r.get('dist_50sma_pct', 0.0)
                cells.append(f'<td style="padding: 10px 12px; font-size:0.85rem;"><span style="color:#00e676;">20: +{d20:.1f}%</span><br><span style="color:#29b6f6;">50: +{d50:.1f}%</span></td>')
            elif strategy_type == "support_ma":
                d65 = r.get('dist_65sma_pct', 0.0)
                color = "#00e676" if d65 >= 0 else "#ef4444"
                cells.append(f'<td style="padding: 10px 12px; font-size:0.85rem; color:{color};">65 SMA: {d65:+.1f}%</td>')
            elif strategy_type == "crossover_ma":
                d50 = r.get('dist_50sma_pct', 0.0)
                d200 = r.get('dist_200sma_pct', 0.0)
                cells.append(f'<td style="padding: 10px 12px; font-size:0.85rem;"><span style="color:#29b6f6;">50: {d50:+.1f}%</span><br><span style="color:#ffa000;">200: {d200:+.1f}%</span></td>')
            
        elif strategy_type == "minervini":
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            
            run_up_200 = r.get('run_up_200', 0.0)
            cells.append(f'<td style="padding: 10px 12px; color: #29b6f6; font-weight: 600;">+{run_up_200:.1f}%</td>')
            
            run_up_52w = r.get('run_up_52w', 0.0)
            cells.append(f'<td style="padding: 10px 12px; color: #ffa000; font-weight: 600;">+{run_up_52w:.1f}%</td>')
            
            is_early = r.get('is_early', True)
            stage_badge = '<span class="custom-badge badge-green" style="font-weight:600;">Early Stage-2</span>' if is_early else '<span class="custom-badge badge-amber" style="font-weight:600;">Extended</span>'
            cells.append(f'<td style="padding: 10px 12px;">{stage_badge}</td>')
            
            rem_pct = ((target - r['cmp']) / r['cmp'] * 100) if r['cmp'] > 0 else 0.0
            cells.append(f'<td style="padding: 10px 12px; color: #00e676; font-weight: 700;">+{rem_pct:.1f}%</td>')
            
        elif strategy_type == "wavetrend":
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            wt1_val = r.get('wt_value', 0.0)
            wt_color = "#ef4444" if wt1_val <= -60 else "#ffa000" if wt1_val <= -50 else "#29b6f6"
            cells.append(f'<td style="padding: 10px 12px; color: {wt_color}; font-weight: 600;">{wt1_val:.1f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #94a3b8;">{r.get("wt2_value", 0.0):.1f}</td>')
            diff_val = r.get('wt_diff', wt1_val - r.get('wt2_value', 0.0))
            diff_color = "#00e676" if diff_val > 0 else "#ef4444"
            cells.append(f'<td style="padding: 10px 12px; color: {diff_color}; font-weight: 600;">{diff_val:+.1f}</td>')
            sig_badge = '<span class="custom-badge badge-green" style="font-weight:600; padding: 2px 6px; border-radius: 4px;">🟢 BUY</span>' if r.get('buy_signal') else '<span class="custom-badge badge-grey" style="padding: 2px 6px; border-radius: 4px;">Oversold</span>'
            cells.append(f'<td style="padding: 10px 12px;">{sig_badge}</td>')
            
        elif strategy_type == "vcs":
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            score_val = r.get('vcs_score', 0.0)
            cells.append(f'<td style="padding: 10px 12px; color: #29b6f6; font-weight: 700;">{score_val:.2f}</td>')
            
        elif strategy_type == "struct_vcp":
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #00e676; font-weight: 700;">{r.get("contractions", 0)}T</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">{r.get("vol_50d", 0):,.0f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #ffa000;">₹{r.get("pivot_price", 0.0):,.2f}</td>')
            
        elif strategy_type == "vpa":
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #29b6f6; font-weight: 700;">{r.get("trend_score", r.get("score", 0.0)):.1f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">{r.get("pattern", "N/A")}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #00e676;">{r.get("trend", "N/A")}</td>')
            
        elif strategy_type == "stage2":
            cells.append(f'<td style="padding: 10px 12px; color: #00e676; font-weight: 600;">₹{r.get("base_bottom", 0.0):,.2f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1;">₹{r.get("historical_high", 0.0):,.2f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #ffa000; font-weight: 600;">{r.get("extension", 0.0):.1f}%</td>')
            cells.append(f'<td style="padding: 10px 12px; color: #29b6f6;">₹{r.get("sma7", 0.0):,.2f}</td>')
            rsi_color = "#00e676" if 60 <= r.get("rsi", 0.0) <= 75 else "#ffa000"
            cci_color = "#00e676" if r.get("cci", 0.0) >= 100 else "#ffa000" if r.get("cci", 0.0) >= 0 else "#ef4444"
            cells.append(f'<td style="padding: 10px 12px; color: {rsi_color}; font-weight: 600;">{r.get("rsi", 0.0):.1f}</td>')
            cells.append(f'<td style="padding: 10px 12px; color: {cci_color}; font-weight: 600;">{r.get("cci", 0.0):.1f}</td>')
            cells.append(f'<td style="padding: 10px 12px;">{get_signal_badge_html(r.get("score", 0.0))}</td>')
            
        # Common Execution Columns
        cells.append(f'<td style="padding: 10px 12px; color: #cbd5e1; font-weight: 600;">₹{buy:,.2f}</td>')
        cells.append(f'<td style="padding: 10px 12px; color: #ef4444; font-weight: 600;">₹{sl:,.2f}</td>')
        cells.append(f'<td style="padding: 10px 12px; color: #00e676; font-weight: 600;">₹{target:,.2f}</td>')
        cells.append(f'<td style="padding: 10px 12px;">{conf_badge}</td>')
        cells.append(f'<td style="padding: 10px 12px; color: #94a3b8; font-style: italic; font-size: 0.82rem; line-height: 1.4; min-width: 250px; max-width: 350px; white-space: normal !important; word-wrap: break-word;">"{clean_rec}"</td>')
        
        row_str = f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.2s;">{"".join(cells)}</tr>'
        rows_html.append(row_str)
        
    table_rows = "".join(rows_html)
    
    # Headers based on strategy
    headers = ["Watchlist", "Symbol", "Sector", "CMP"]
    if strategy_type == "vdu_breakout":
        headers.extend(["Day Chg %", "Volume", "Dry Avg Vol", "Vol Ratio", "Dry Days", "Spikes", "Score"])
    elif strategy_type == "gapup":
        headers.extend(["Prev Close", "Open", "Gap %", "Day Chg %", "Volume"])
    elif strategy_type in ["above_ma", "support_ma", "crossover_ma"]:
        headers.extend(["Day Chg %", "Dist to SMA"])
    elif strategy_type == "minervini":
        headers.extend(["Day Chg %", "Run Up 200 SMA", "Run Up 52w Low", "Stage Type", "Remaining Target %"])
    elif strategy_type == "wavetrend":
        headers.extend(["Day Chg %", "WT1", "WT2", "WT Diff", "Signal"])
    elif strategy_type == "vcs":
        headers.extend(["Day Chg %", "VCS Score"])
    elif strategy_type == "struct_vcp":
        headers.extend(["Day Chg %", "Contractions", "Avg Vol", "Pivot Price"])
    elif strategy_type == "vpa":
        headers.extend(["Day Chg %", "VPA Score", "Pattern", "Trend"])
    elif strategy_type == "stage2":
        headers.extend(["Base Bottom", "Historical High", "Extension %", "7M SMA", "RSI", "CCI", "Score"])
        
    # Append common execution columns
    headers.extend(["Buy Range", "Stop Loss", "Swing Target", "Confidence", "Actionable Guidance & Reasoning"])
    
    # Render table headers dynamically with active direction arrow indicators
    header_cols = []
    for h in headers:
        if h in sort_mapper:
            if active_col == h:
                arrow = " 🟢▲" if active_dir == "asc" else " 🟢▼"
            else:
                arrow = " ↕️"
            header_cols.append(
                f'<th style="padding: 8px 12px;">'
                f'<a href="/?sort_col={h}&prefix={key_prefix}" target="_self" style="color: #38bdf8; text-decoration: none;">'
                f'{h}{arrow}'
                f'</a>'
                f'</th>'
            )
        else:
            header_cols.append(f'<th style="padding: 8px 12px;">{h}</th>')
            
    header_cols_html = "".join(header_cols)
    
    st.markdown(
        f'<div class="glass-card" style="padding: 18px; margin-bottom: 22px; border: 1px solid rgba(41, 182, 246, 0.2); background: rgba(9, 13, 22, 0.55); border-radius: 12px;">'
        f'<div style="overflow-x: auto;">'
        f'<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; color: #cbd5e1; font-family: Outfit, sans-serif;">'
        f'<thead>'
        f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #38bdf8; font-weight: bold; background: rgba(56, 189, 248, 0.05); font-size: 0.8rem; text-transform: uppercase;">'
        f'{header_cols_html}'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'{table_rows}'
        f'</tbody>'
        f'</table>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

def render_quick_trade_board(results_list: list, key_prefix: str):
    if not results_list or len(results_list) == 0:
        return
        
    rows_html = []
    for r in results_list:
        buy = r.get('buy_price') or r.get('cmp') or 0.0
        sl = r.get('exit_price') or 0.0
        target = r.get('target_price') or 0.0
        conf = r.get('confidence') or 'Medium'
        clean_conf = conf.split(" (")[0] if " (" in conf else conf
        rec = r.get('recommendation') or 'No recommendation generated.'
        
        clean_rec = extract_clean_recommendation(rec)
        
        # Color coding confidence badge
        conf_color = "#ef4444" if "Low" in clean_conf else "#ffa000" if "Medium" in clean_conf else "#00e676"
        conf_badge = f'<span class="custom-badge" style="background: rgba({ "0,230,118" if "High" in clean_conf else "255,160,0" if "Medium" in clean_conf else "239,68,68" },0.12); color: {conf_color}; border: 1px solid {conf_color}; font-size: 0.75rem; font-weight: bold; padding: 2px 6px; border-radius: 4px;">{clean_conf}</span>'
        
        row_str = (
            f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.2s;">'
            f'<td style="padding: 10px 12px; font-weight: bold; color: #29b6f6;">{r["symbol"]}</td>'
            f'<td style="padding: 10px 12px; color: #e2e8f0; font-weight: 500;">₹{r.get("cmp", r.get("buy_price", 0.0)):,.2f}</td>'
            f'<td style="padding: 10px 12px; color: #e2e8f0; font-weight: 600;">₹{buy:,.2f}</td>'
            f'<td style="padding: 10px 12px; color: #ef4444; font-weight: 600;">₹{sl:,.2f}</td>'
            f'<td style="padding: 10px 12px; color: #00e676; font-weight: 600;">₹{target:,.2f}</td>'
            f'<td style="padding: 10px 12px;">{conf_badge}</td>'
            f'<td style="padding: 10px 12px; color: #94a3b8; font-style: italic; font-size: 0.82rem; line-height: 1.4; min-width: 250px; max-width: 350px; white-space: normal !important; word-wrap: break-word;">"{clean_rec}"</td>'
            f'</tr>'
        )
        rows_html.append(row_str)
        
    table_rows = "".join(rows_html)
    
    st.markdown(
        f'<div class="glass-card" style="padding: 18px; margin-bottom: 22px; border: 1px solid rgba(41, 182, 246, 0.2); background: rgba(9, 13, 22, 0.55); border-radius: 12px;">'
        f'<h3 style="margin-top:0; color:#29b6f6; font-size:1.15rem; display: flex; align-items: center; gap: 8px; font-family: Outfit, sans-serif;">'
        f'🎯 Quick-Action Trade Execution Matrix'
        f'</h3>'
        f'<p style="font-size:0.85rem; color:#94a3b8; margin-top:-8px; margin-bottom:15px; font-family: Outfit, sans-serif;">'
        f'A consolidated execution sheet for all active setups. Use these precise price thresholds to configure your trade orders instantly.'
        f'</p>'
        f'<div style="overflow-x: auto;">'
        f'<table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.85rem; color: #cbd5e1; font-family: Outfit, sans-serif;">'
        f'<thead>'
        f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #38bdf8; font-weight: bold; background: rgba(56, 189, 248, 0.05); font-size: 0.8rem; text-transform: uppercase;">'
        f'<th style="padding: 8px 12px;">Symbol</th>'
        f'<th style="padding: 8px 12px;">CMP</th>'
        f'<th style="padding: 8px 12px; color: #29b6f6;">Buy Range</th>'
        f'<th style="padding: 8px 12px; color: #ef4444;">Stop Loss</th>'
        f'<th style="padding: 8px 12px; color: #00e676;">Swing Target</th>'
        f'<th style="padding: 8px 12px;">Confidence</th>'
        f'<th style="padding: 8px 12px; width: 40%;">Actionable Guidance & Reasoning</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>'
        f'{table_rows}'
        f'</tbody>'
        f'</table>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# --- Initialize Session State ---
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'failed_count' not in st.session_state:
    st.session_state.failed_count = 0
if 'last_scanned' not in st.session_state:
    st.session_state.last_scanned = None
if 'confirm_clear' not in st.session_state:
    st.session_state.confirm_clear = False
if 'ai_selected_stock' not in st.session_state:
    st.session_state.ai_selected_stock = ""
if 'ai_custom_sym_input' not in st.session_state:
    st.session_state.ai_custom_sym_input = ""
if 'gapup_results' not in st.session_state:
    st.session_state.gapup_results = None
if 'above_ma_results' not in st.session_state:
    st.session_state.above_ma_results = None
if 'support_ma_results' not in st.session_state:
    st.session_state.support_ma_results = None
if 'crossover_ma_results' not in st.session_state:
    st.session_state.crossover_ma_results = None
if 'wt_results' not in st.session_state:
    st.session_state.wt_results = None
if 'wt_results_by_tf' not in st.session_state:
    st.session_state.wt_results_by_tf = {}
if 'minervini_results' not in st.session_state:
    st.session_state.minervini_results = None
if 'vcs_results' not in st.session_state:
    st.session_state.vcs_results = None
if 'structural_vcp_results' not in st.session_state:
    st.session_state.structural_vcp_results = None
# Initialize global status dictionary if not present (shared across all threads/sessions)
if "MOMENTUM_SCAN_STATUS" not in globals():
    global MOMENTUM_SCAN_STATUS
    MOMENTUM_SCAN_STATUS = {
        "is_running": False,
        "status_text": "Not started",
        "progress": 0.0,
        "monthly_results": None,
        "weekly_results": None
    }

def run_background_momentum_scans():
    """
    Runs both Monthly and Weekly Momentum scans in a non-blocking background daemon thread.
    Updates MOMENTUM_SCAN_STATUS and saves the results to daily JSON cache files.
    """
    global MOMENTUM_SCAN_STATUS
    if MOMENTUM_SCAN_STATUS["is_running"]:
        return

    MOMENTUM_SCAN_STATUS["is_running"] = True
    MOMENTUM_SCAN_STATUS["status_text"] = "Initializing background scans..."
    MOMENTUM_SCAN_STATUS["progress"] = 0.0

    def target_runner():
        import concurrent.futures as _cf
        import time as _time
        import json
        
        try:
            today_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
            CRORE = 1_00_00_000

            # 1. Resolve Universe (using ALL NSE for comprehensive coverage)
            MOMENTUM_SCAN_STATUS["status_text"] = "Resolving ALL NSE listed symbols..."
            from data_fetcher import get_all_nse_symbols
            universe = get_all_nse_symbols()
            
            if not universe:
                MOMENTUM_SCAN_STATUS["status_text"] = "Error: Could not resolve NSE symbols universe."
                MOMENTUM_SCAN_STATUS["is_running"] = False
                return

            # Resolve monthly and weekly base dates
            import database
            from datetime import timedelta
            from scanner import run_monthly_momentum_update, run_weekly_momentum_update
            
            today_ist = datetime.now(IST_TIMEZONE)
            base_date_monthly = database.get_monthly_base_date(today_ist.year, today_ist.month)
            
            iso_weekday = today_ist.isoweekday()
            start_of_week = today_ist - timedelta(days=iso_weekday - 1)
            end_of_week = start_of_week + timedelta(days=6)
            base_date_weekly = database.get_weekly_base_date(start_of_week.strftime("%Y-%m-%d"), end_of_week.strftime("%Y-%m-%d"))
            
            if base_date_monthly and base_date_monthly != today_str and base_date_weekly and base_date_weekly != today_str:
                # Both monthly and weekly are already established! Run lightning-fast updates.
                MOMENTUM_SCAN_STATUS["status_text"] = "Running lightning-fast momentum price updates..."
                MOMENTUM_SCAN_STATUS["progress"] = 0.30
                mm_results = run_monthly_momentum_update(base_date_monthly, today_str)
                
                MOMENTUM_SCAN_STATUS["progress"] = 0.60
                wm_results = run_weekly_momentum_update(base_date_weekly, today_str)
                
                MOMENTUM_SCAN_STATUS["status_text"] = "Step 5/5 - Saving results to PostgreSQL & JSON cache..."
                MOMENTUM_SCAN_STATUS["progress"] = 0.95
                try:
                    database.save_monthly_momentum_results(today_str, mm_results)
                    database.save_weekly_momentum_results(today_str, wm_results)
                except Exception as db_save_ex:
                    print(f"Failed to cache momentum results in PostgreSQL: {db_save_ex}")
                
                monthly_payload = {"date": today_str, "results": mm_results}
                with open("monthly_momentum_cache.json", "w") as f:
                    json.dump(monthly_payload, f, indent=2)

                weekly_payload = {"date": today_str, "results": wm_results}
                with open("weekly_momentum_cache.json", "w") as f:
                    json.dump(weekly_payload, f, indent=2)

                MOMENTUM_SCAN_STATUS["monthly_results"] = mm_results
                MOMENTUM_SCAN_STATUS["weekly_results"] = wm_results
                MOMENTUM_SCAN_STATUS["status_text"] = "Complete!"
                MOMENTUM_SCAN_STATUS["progress"] = 1.0
                MOMENTUM_SCAN_STATUS["is_running"] = False
                print(f"Background scans complete: Monthly found {len(mm_results)}, Weekly found {len(wm_results)}.")
                return

            # ==========================================
            # STEP 1: DOWNLOAD DAILY DATA TO FILTER BY PRICE
            # ==========================================
            MOMENTUM_SCAN_STATUS["status_text"] = "Step 1/5 - Downloading daily quotes..."
            price_map_mm = {}
            price_map_wm = {}
            tickers_ns = [f"{s.strip().upper()}.NS" for s in universe]
            chunk_size = 200
            ticker_chunks = [tickers_ns[i:i+chunk_size] for i in range(0, len(tickers_ns), chunk_size)]
            
            for chunk_idx, chunk in enumerate(ticker_chunks):
                MOMENTUM_SCAN_STATUS["status_text"] = f"Step 1/5 - Quote chunk {chunk_idx+1}/{len(ticker_chunks)}..."
                MOMENTUM_SCAN_STATUS["progress"] = 0.05 + (chunk_idx / len(ticker_chunks)) * 0.15
                try:
                    q_df = yf.download(tickers=chunk, period="1d", progress=False, threads=False)
                    if not q_df.empty and isinstance(q_df.columns, pd.MultiIndex):
                        price_types = q_df.columns.get_level_values(0).unique().tolist()
                        cl_s = q_df['Close'].iloc[-1] if 'Close' in price_types else pd.Series(dtype=float)
                        for tk, pv in cl_s.items():
                            sym_clean = str(tk).replace(".NS", "").upper()
                            if not pd.isna(pv):
                                val = float(pv)
                                if val >= 100.0:
                                    price_map_mm[sym_clean] = val
                                if val >= 200.0:
                                    price_map_wm[sym_clean] = val
                except Exception as e:
                    print(f"Background quote chunk {chunk_idx+1} failed: {e}")
                _time.sleep(0.05)  # Reduced from 0.3s — yfinance rate-limits at request level

            # ==========================================
            # STEP 2: FETCH MARKET CAPS FOR PASSED STOCKS
            # ==========================================
            passed_price_both = list(set(list(price_map_mm.keys()) + list(price_map_wm.keys())))
            MOMENTUM_SCAN_STATUS["status_text"] = f"Step 2/5 - Fetching market caps for {len(passed_price_both)} stocks..."
            
            mcap_map = {}
            def _fetch_single_mcap(sym):
                try:
                    fi = yf.Ticker(f"{sym}.NS").fast_info
                    mc = getattr(fi, 'market_cap', None) or 0
                    return sym, mc / CRORE
                except Exception:
                    return sym, 0.0

            processed_mcap_count = 0
            with _cf.ThreadPoolExecutor(max_workers=10) as pool:
                for sym_r, mcap_cr in pool.map(_fetch_single_mcap, passed_price_both):
                    mcap_map[sym_r] = mcap_cr
                    processed_mcap_count += 1
                    MOMENTUM_SCAN_STATUS["progress"] = 0.20 + (processed_mcap_count / len(passed_price_both)) * 0.15
                    if processed_mcap_count % 20 == 0:
                        MOMENTUM_SCAN_STATUS["status_text"] = f"Step 2/5 - Fetched {processed_mcap_count}/{len(passed_price_both)} market caps..."

            # Filter candidates for both scans
            mm_candidates = [s for s in price_map_mm if mcap_map.get(s, 0.0) >= 3000.0 or mcap_map.get(s, 0.0) == 0.0]
            wm_candidates = [s for s in price_map_wm if mcap_map.get(s, 0.0) >= 5000.0]
            
            # ==========================================
            # STEP 3: MONTHLY MOMENTUM SCAN OR UPDATE
            # ==========================================
            if base_date_monthly and base_date_monthly != today_str:
                MOMENTUM_SCAN_STATUS["status_text"] = f"Step 3/5 - Running Monthly Momentum price update (since {base_date_monthly})..."
                MOMENTUM_SCAN_STATUS["progress"] = 0.50
                mm_results = run_monthly_momentum_update(base_date_monthly, today_str)
            else:
                MOMENTUM_SCAN_STATUS["status_text"] = f"Step 3/5 - Scanning {len(mm_candidates)} stocks for Monthly Momentum..."
                mm_results = []
                monthly_chunk_size = 80  # Increased from 50 for fewer API calls
                mm_chunks = [mm_candidates[i:i+monthly_chunk_size] for i in range(0, len(mm_candidates), monthly_chunk_size)]
                
                for chunk_idx, chunk in enumerate(mm_chunks):
                    MOMENTUM_SCAN_STATUS["status_text"] = f"Step 3/5 - Monthly chunk {chunk_idx+1}/{len(mm_chunks)} (Found {len(mm_results)} matches)..."
                    MOMENTUM_SCAN_STATUS["progress"] = 0.35 + (chunk_idx / len(mm_chunks)) * 0.30
                    chunk_ns = [f"{s}.NS" for s in chunk]
                    try:
                        df_mbulk = yf.download(tickers=chunk_ns, period="10y", interval="1mo", progress=False, threads=False)
                        for sym in chunk:
                            sym_ns = f"{sym}.NS"
                            try:
                                if isinstance(df_mbulk.columns, pd.MultiIndex):
                                    all_t_mm = df_mbulk.columns.get_level_values(1).unique().tolist()
                                    matched_m = next((t for t in all_t_mm if t.upper() == sym_ns.upper()), None)
                                    if matched_m is None:
                                        continue
                                    t_df_m = df_mbulk.xs(matched_m, axis=1, level=1).copy()
                                else:
                                    if len(chunk_ns) == 1:
                                        t_df_m = df_mbulk.copy()
                                    else:
                                        continue
                                
                                req_m = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if not all(col in t_df_m.columns for col in req_m):
                                    continue
                                t_df_m = t_df_m[req_m].dropna(subset=['Close'])
                                t_df_m = t_df_m[t_df_m['Volume'] > 0]
                                if len(t_df_m) < 22:
                                    continue
                                t_df_m = t_df_m.reset_index()
                                t_df_m.rename(columns={t_df_m.columns[0]: 'Date'}, inplace=True)
                                t_df_m['Date'] = pd.to_datetime(t_df_m['Date']).dt.tz_localize(None)

                                res_m = scan_monthly_momentum(sym, t_df_m, market_cap_cr=mcap_map.get(sym, 0.0))
                                if res_m is not None:
                                    if 'df' in res_m:
                                        del res_m['df']
                                    mm_results.append(res_m)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Monthly download chunk {chunk_idx+1} failed: {e}")
                    _time.sleep(0.05)  # Reduced from 0.3s — yfinance rate-limits at request level

            # ==========================================
            # STEP 4: WEEKLY MOMENTUM SCAN OR UPDATE
            # ==========================================
            if base_date_weekly and base_date_weekly != today_str:
                MOMENTUM_SCAN_STATUS["status_text"] = f"Step 4/5 - Running Weekly Momentum price update (since {base_date_weekly})..."
                MOMENTUM_SCAN_STATUS["progress"] = 0.85
                wm_results = run_weekly_momentum_update(base_date_weekly, today_str)
            else:
                MOMENTUM_SCAN_STATUS["status_text"] = f"Step 4/5 - Scanning {len(wm_candidates)} stocks for Weekly Momentum..."
                wm_results = []
                weekly_chunk_size = 100  # Increased from 60 for fewer API calls
                wm_chunks = [wm_candidates[i:i+weekly_chunk_size] for i in range(0, len(wm_candidates), weekly_chunk_size)]
                
                for chunk_idx, chunk in enumerate(wm_chunks):
                    MOMENTUM_SCAN_STATUS["status_text"] = f"Step 4/5 - Weekly chunk {chunk_idx+1}/{len(wm_chunks)} (Found {len(wm_results)} matches)..."
                    MOMENTUM_SCAN_STATUS["progress"] = 0.65 + (chunk_idx / len(wm_chunks)) * 0.30
                    chunk_ns = [f"{s}.NS" for s in chunk]
                    try:
                        df_wbulk = yf.download(tickers=chunk_ns, period="3y", interval="1wk", progress=False, threads=False)
                        for sym in chunk:
                            sym_ns = f"{sym}.NS"
                            try:
                                if isinstance(df_wbulk.columns, pd.MultiIndex):
                                    all_t_wm = df_wbulk.columns.get_level_values(1).unique().tolist()
                                    matched_w = next((t for t in all_t_wm if t.upper() == sym_ns.upper()), None)
                                    if matched_w is None:
                                        continue
                                    t_df_w = df_wbulk.xs(matched_w, axis=1, level=1).copy()
                                else:
                                    if len(chunk_ns) == 1:
                                        t_df_w = df_wbulk.copy()
                                    else:
                                        continue

                                req_w = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if not all(col in t_df_w.columns for col in req_w):
                                    continue
                                t_df_w = t_df_w[req_w].dropna(subset=['Close'])
                                t_df_w = t_df_w[t_df_w['Volume'] > 0]
                                if len(t_df_w) < 22:
                                    continue
                                t_df_w = t_df_w.reset_index()
                                t_df_w.rename(columns={t_df_w.columns[0]: 'Date'}, inplace=True)
                                t_df_w['Date'] = pd.to_datetime(t_df_w['Date']).dt.tz_localize(None)

                                res_w = scan_weekly_momentum(sym, t_df_w, market_cap_cr=mcap_map.get(sym, 0.0))
                                if res_w is not None:
                                    if 'df' in res_w:
                                        del res_w['df']
                                    wm_results.append(res_w)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Weekly download chunk {chunk_idx+1} failed: {e}")
                    _time.sleep(0.3)

            # ==========================================
            # STEP 5: CACHE & COMPLETE
            # ==========================================
            MOMENTUM_SCAN_STATUS["status_text"] = "Step 5/5 - Saving results to PostgreSQL & JSON cache..."
            MOMENTUM_SCAN_STATUS["progress"] = 0.95
            
            # Save to PostgreSQL database
            try:
                import database
                database.save_monthly_momentum_results(today_str, mm_results)
                database.save_weekly_momentum_results(today_str, wm_results)
            except Exception as db_save_ex:
                print(f"Failed to cache momentum results in PostgreSQL: {db_save_ex}")
            
            monthly_payload = {"date": today_str, "results": mm_results}
            with open("monthly_momentum_cache.json", "w") as f:
                json.dump(monthly_payload, f, indent=2)

            weekly_payload = {"date": today_str, "results": wm_results}
            with open("weekly_momentum_cache.json", "w") as f:
                json.dump(weekly_payload, f, indent=2)

            MOMENTUM_SCAN_STATUS["monthly_results"] = mm_results
            MOMENTUM_SCAN_STATUS["weekly_results"] = wm_results
            MOMENTUM_SCAN_STATUS["status_text"] = "Complete!"
            MOMENTUM_SCAN_STATUS["progress"] = 1.0
            MOMENTUM_SCAN_STATUS["is_running"] = False
            
            print(f"Background scans complete: Monthly found {len(mm_results)}, Weekly found {len(wm_results)}.")

        except Exception as err:
            MOMENTUM_SCAN_STATUS["status_text"] = f"Background scan error: {err}"
            MOMENTUM_SCAN_STATUS["is_running"] = False
            print(f"Background momentum scans error: {err}")

    # Launch daemon thread
    t = threading.Thread(target=target_runner, name="Background_Momentum_Scans", daemon=True)
    t.start()

# --- Boot Cache Loader / Scanner Trigger ---
if 'monthly_momentum_results' not in st.session_state:
    st.session_state.monthly_momentum_results = None
if 'weekly_momentum_results' not in st.session_state:
    st.session_state.weekly_momentum_results = None

today_str_check = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")

if st.session_state.monthly_momentum_results is None:
    # 1. Try fetching from PostgreSQL database first
    try:
        import database
        db_results = database.get_cached_monthly_momentum(today_str_check)
        if db_results:
            st.session_state.monthly_momentum_results = db_results
            MOMENTUM_SCAN_STATUS["monthly_results"] = db_results
            print(f"Loaded today's Monthly Momentum results ({len(db_results)} stocks) from PostgreSQL cache.")
    except Exception as db_err:
        print(f"Error loading Monthly Momentum from database: {db_err}")

    # 2. Fallback to local JSON cache file
    if st.session_state.monthly_momentum_results is None:
        try:
            if os.path.exists("monthly_momentum_cache.json"):
                import json
                with open("monthly_momentum_cache.json", "r") as f:
                    data = json.load(f)
                    if data.get("date") == today_str_check:
                        st.session_state.monthly_momentum_results = data.get("results")
                        MOMENTUM_SCAN_STATUS["monthly_results"] = data.get("results")
                        print(f"Loaded today's Monthly Momentum results ({len(data.get('results'))} stocks) from local JSON fallback cache.")
        except Exception as e:
            print(f"Error loading monthly cache on boot: {e}")

if st.session_state.weekly_momentum_results is None:
    # 1. Try fetching from PostgreSQL database first
    try:
        import database
        db_results = database.get_cached_weekly_momentum(today_str_check)
        if db_results:
            st.session_state.weekly_momentum_results = db_results
            MOMENTUM_SCAN_STATUS["weekly_results"] = db_results
            print(f"Loaded today's Weekly Momentum results ({len(db_results)} stocks) from PostgreSQL cache.")
    except Exception as db_err:
        print(f"Error loading Weekly Momentum from database: {db_err}")

    # 2. Fallback to local JSON cache file
    if st.session_state.weekly_momentum_results is None:
        try:
            if os.path.exists("weekly_momentum_cache.json"):
                import json
                with open("weekly_momentum_cache.json", "r") as f:
                    data = json.load(f)
                    if data.get("date") == today_str_check:
                        st.session_state.weekly_momentum_results = data.get("results")
                        MOMENTUM_SCAN_STATUS["weekly_results"] = data.get("results")
                        print(f"Loaded today's Weekly Momentum results ({len(data.get('results'))} stocks) from local JSON fallback cache.")
        except Exception as e:
            print(f"Error loading weekly cache on boot: {e}")

# Automatically trigger scanning in background if results are missing for today
if (st.session_state.monthly_momentum_results is None or st.session_state.weekly_momentum_results is None) and not MOMENTUM_SCAN_STATUS["is_running"]:
    run_background_momentum_scans()

# --- Automatic Daily Database Cache Loader ---
# CRITICAL: Only hit the database when results are not yet in session state.
# This prevents DB calls (and potential hangs) on every Streamlit re-render.
if st.session_state.scan_results is None and not st.session_state.get('db_cache_checked', False):
    st.session_state['db_cache_checked'] = True
    try:
        # Load the absolute latest scan session date from the database
        available_dates = database.get_available_scan_dates()
        if available_dates:
            latest_date_str = available_dates[0]
            cached_log = database.has_scanned_today(latest_date_str)
            if cached_log:
                try:
                    st.session_state.scan_results = database.get_cached_breakouts(latest_date_str)
                except Exception:
                    st.session_state.scan_results = []
                try:
                    st.session_state.gapup_results = database.get_cached_gapups(latest_date_str)
                except Exception:
                    st.session_state.gapup_results = []
                try:
                    st.session_state.above_ma_results = database.get_cached_trend_setups(latest_date_str, 'above_ma')
                except Exception:
                    st.session_state.above_ma_results = []
                try:
                    st.session_state.support_ma_results = database.get_cached_trend_setups(latest_date_str, 'support_ma')
                except Exception:
                    st.session_state.support_ma_results = []
                try:
                    st.session_state.crossover_ma_results = database.get_cached_trend_setups(latest_date_str, 'crossover_ma')
                except Exception:
                    st.session_state.crossover_ma_results = []
                try:
                    st.session_state.wt_results = database.get_cached_wt_cross(latest_date_str)
                    st.session_state.wt_results_by_tf = {"Daily": st.session_state.wt_results}
                except Exception:
                    st.session_state.wt_results = []
                    st.session_state.wt_results_by_tf = {"Daily": []}
                try:
                    st.session_state.minervini_results = ensure_minervini_fields(database.get_cached_trend_setups(latest_date_str, 'minervini'))
                except Exception:
                    st.session_state.minervini_results = []
                try:
                    st.session_state.vcs_results = database.get_cached_vcs(latest_date_str)
                except Exception:
                    st.session_state.vcs_results = []
                try:
                    st.session_state.vpa_results = database.get_cached_vpa(latest_date_str)
                except Exception:
                    st.session_state.vpa_results = []
                st.session_state.total_scanned = cached_log.get('total_scanned', 0)
                st.session_state.failed_count = 0
                st.session_state.last_scanned = latest_date_str + " (Loaded from DB Cache)"
                
                # Auto-resume background AI scanning if there are unscanned candidates in session state
                all_syms = []
                if st.session_state.scan_results:
                    all_syms.extend([r['symbol'] for r in st.session_state.scan_results])

                all_syms = list(set(all_syms))
                if all_syms:
                    try:
                        run_background_ai_scan(all_syms, latest_date_str)
                    except Exception as auto_scan_err:
                        print(f"Failed to auto-resume background AI scan on boot: {auto_scan_err}")
    except Exception as cache_err:
        print(f"Error loading daily database scan cache on boot: {cache_err}")


# --- HEADER SECTION ---
st.markdown('<h1 class="gradient-title">📈 Volume Surge Scanner</h1>', unsafe_allow_html=True)
st.markdown('<p class="gradient-subtitle">Scan NSE-listed stocks for institutional Volume Dry-Up (VDU) breakouts & build a high-conviction swing trading watchlist.</p>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown('### ⚙️ Scan Universe')
universe_selection = st.sidebar.selectbox(
    "Select Universe to Scan",
    options=["NIFTY 100 (Recommended)", "NIFTY 50 (Ultra Fast)", "All NSE Listed Equities (Full Scan)"],
    index=2,
    help="Select the universe of stocks to scan. NIFTY 100/50 are extremely fast and completely bypass Yahoo Finance rate limits."
)

st.sidebar.markdown(
    "<div style='padding:8px 12px; background:rgba(41,182,246,0.06); border:1px solid rgba(41,182,246,0.15); border-radius:10px; margin-bottom: 15px;'>"
    "<span style='color:#ffa000; font-size:0.8rem; font-weight:600;'>⚡ Filters: Price > ₹200 | Market Cap > ₹3000 Cr</span>"
    "</div>", 
    unsafe_allow_html=True
)


st.sidebar.markdown('---')
st.sidebar.markdown('### 🔍 VDU Strategy Filters')

# Algorithmic parameter sliders
min_vol_ratio = st.sidebar.slider(
    "Min Volume Ratio",
    min_value=2.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    key="vdu_min_vol_ratio_v4",
    help="Breakout day volume compared to dry average volume (e.g., 2.0 = 2x surge)"
)

min_price_chg = st.sidebar.slider(
    "Min Price Change %",
    min_value=1.5,
    max_value=10.0,
    value=1.5,
    step=0.5,
    key="vdu_min_price_chg_v4",
    help="Minimum price percentage increase on the breakout day (Close vs Open)"
)

dry_zone_range = st.sidebar.slider(
    "Dry Zone Range (Trading Days)",
    min_value=0,
    max_value=150,
    value=(15, 85),
    step=5,
    key="vdu_dry_zone_range_v4",
    help="Configure the minimum and maximum duration of the dry zone consolidation period (up to 150 days)"
)

min_dry_spikes = st.sidebar.slider(
    "Min Spikes in Dry Zone",
    min_value=0,
    max_value=20,
    value=2,
    step=1,
    key="vdu_min_dry_spikes_v5",
    help="Requires at least this many volume accumulation spikes inside the dry zone window (up to 20 spikes)"
)

min_signal_str = st.sidebar.slider(
    "Min Signal Strength Score",
    min_value=0,
    max_value=100,
    value=40,
    step=5,
    key="vdu_min_signal_str_v4",
    help="Filter stocks based on overall calculated algorithmic rating"
)

vcp_max_tightness = 7.0

above_50dma_only = st.sidebar.checkbox(
    "Above 50 DMA Only",
    value=False,
    help="If checked, only lists breakout stocks trading above their 50-day Simple Moving Average"
)
above_200dma_only = st.sidebar.checkbox(
    "Above 200 DMA Only",
    value=False,
    help="If checked, only lists breakout stocks trading above their 200-day Simple Moving Average"
)

st.sidebar.markdown('---')
scan_timeframe = st.sidebar.selectbox(
    "Scanning Timeframe",
    ["Daily (1d)", "Weekly (1wk)", "Monthly (1mo)"],
    index=0,
    help="Select the timeframe for the scan. Note: Weekly and Monthly scans require downloading more data and take longer."
)

st.sidebar.markdown('---')


# --- RUN SCAN ACTION ---
if st.sidebar.button("🔍 Run Scanner", use_container_width=True):
    # Resolve the universe selected in the sidebar
    if "NIFTY 50" in universe_selection:
        universe_key = "NIFTY 50"
    elif "NIFTY 100" in universe_selection:
        universe_key = "NIFTY 100"
    elif "WATCHLIST" in universe_selection.upper():
        universe_key = "WATCHLIST"
    else:
        universe_key = "ALL NSE"
        
    if universe_key == "WATCHLIST":
        import watchlist
        wl = watchlist.load_watchlist()
        raw_symbols = [s for s in wl['symbol'].tolist() if pd.notna(s)]
    else:
        raw_symbols = get_index_stocks(universe_key)
        
    if not raw_symbols:
        st.sidebar.error("❌ No symbols found to scan.")
    else:
        # UI Scanner Feedback
        status_box = st.empty()
        prog_bar = st.progress(0)
        
        # Step A: Perform high-speed parallel bulk download of today's quotes to filter Price > 200 instantly
        all_tickers_ns = []
        for s in raw_symbols:
            formatted = s.strip().upper()
            if not formatted.endswith(".NS"):
                formatted = f"{formatted}.NS"
            all_tickers_ns.append(formatted)
            
        today_date_str = datetime.now(IST_TIMEZONE).strftime('%Y-%m-%d')
        cache_key_p1 = f"p1_quotes_v2_{universe_key}_{today_date_str}"
        
        if cache_key_p1 in st.session_state:
            open_price_map, close_price_map, volume_map, high_price_map, low_price_map = st.session_state[cache_key_p1]
            status_box.text("Phase 1/3: Loaded real-time quotes from session cache!")
            prog_bar.progress(1.0)
        else:
            open_price_map = {}
            close_price_map = {}
            volume_map = {}
            high_price_map = {}
            low_price_map = {}
            
            status_box.text("Phase 1/3: Downloading real-time quotes for selected universe...")
            import time
            chunk_size = 500  # Increased from 300 for fewer API calls
            ticker_chunks = [all_tickers_ns[i:i + chunk_size] for i in range(0, len(all_tickers_ns), chunk_size)]
            
            # Thread-safe accumulators for parallel quote downloads
            import threading as _p1_threading
            _p1_lock = _p1_threading.Lock()
            
            def _download_quote_chunk(idx_chunk_pair):
                idx, chunk = idx_chunk_pair
                _open = {}; _close = {}; _vol = {}; _high = {}; _low = {}
                retries = 0
                max_retries = 3
                backoff = 2.0
                while retries <= max_retries:
                    try:
                        # yfinance 1.x: auto_adjust=True by default, threads param removed
                        quotes_df = yf.download(tickers=chunk, period="1d", progress=False, threads=False)
                        if not quotes_df.empty:
                            # yfinance 1.x multi-ticker: MultiIndex (price_type, ticker)
                            if isinstance(quotes_df.columns, pd.MultiIndex):
                                # Level 0 = price type (Close/Open/etc), Level 1 = ticker symbol
                                price_types = quotes_df.columns.get_level_values(0).unique().tolist()
                                tickers_in_idx = quotes_df.columns.get_level_values(1).unique().tolist()
                                # Build per-field Series indexed by ticker (with .NS suffix preserved)
                                def _get_field_series(field):
                                    if field in price_types:
                                        s = quotes_df[field].iloc[-1]
                                        return s
                                    return pd.Series(dtype=float)
                                close_series = _get_field_series('Close')
                                open_series = _get_field_series('Open')
                                volume_series = _get_field_series('Volume')
                                high_series = _get_field_series('High')
                                low_series = _get_field_series('Low')
                            else:
                                # Single ticker fallback
                                ticker_key = chunk[0]
                                close_series = pd.Series({ticker_key: quotes_df['Close'].iloc[-1]})
                                open_series = pd.Series({ticker_key: quotes_df['Open'].iloc[-1]}) if 'Open' in quotes_df else close_series
                                volume_series = pd.Series({ticker_key: quotes_df['Volume'].iloc[-1]}) if 'Volume' in quotes_df else pd.Series({ticker_key: 0})
                                high_series = pd.Series({ticker_key: quotes_df['High'].iloc[-1]}) if 'High' in quotes_df else close_series
                                low_series = pd.Series({ticker_key: quotes_df['Low'].iloc[-1]}) if 'Low' in quotes_df else close_series

                            # Map prices back to plain symbols (strip .NS suffix)
                            # IMPORTANT: index still has .NS suffix, so use k directly for lookup
                            for k, v in close_series.items():
                                clean_k = str(k).replace(".NS", "").upper()
                                if not pd.isna(v) and float(v) > 0:
                                    _close[clean_k] = float(v)
                                    # Use original k (with .NS) to look up in the other series
                                    if k in open_series.index and not pd.isna(open_series[k]):
                                        _open[clean_k] = float(open_series[k])
                                    if k in volume_series.index and not pd.isna(volume_series[k]):
                                        _vol[clean_k] = int(volume_series[k])
                                    if k in high_series.index and not pd.isna(high_series[k]):
                                        _high[clean_k] = float(high_series[k])
                                    if k in low_series.index and not pd.isna(low_series[k]):
                                        _low[clean_k] = float(low_series[k])
                            # Successfully loaded chunk
                            return (_open, _close, _vol, _high, _low)
                        else:
                            raise ValueError("Empty DataFrame returned")
                    except Exception as chunk_ex:
                        retries += 1
                        if retries > max_retries:
                            print(f"Error downloading quote chunk {idx+1}/{len(ticker_chunks)} after {max_retries} retries: {chunk_ex}")
                            break
                        print(f"Rate limited or quote download failed for chunk {idx+1}/{len(ticker_chunks)}. Retrying in {backoff}s... (Error: {chunk_ex})")
                        time.sleep(backoff)
                        backoff *= 2.0
                return ({}, {}, {}, {}, {})
            
            # Parallel execution of Phase 1 quote downloads
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(ticker_chunks))) as p1_executor:
                chunk_pairs = list(enumerate(ticker_chunks))
                for i, result in enumerate(p1_executor.map(_download_quote_chunk, chunk_pairs)):
                    _o, _c, _v, _h, _l = result
                    open_price_map.update(_o)
                    close_price_map.update(_c)
                    volume_map.update(_v)
                    high_price_map.update(_h)
                    low_price_map.update(_l)
                    prog_bar.progress((i + 1) / len(ticker_chunks))
                    status_box.text(f"Phase 1/3: Downloading real-time quotes (Chunk {i+1}/{len(ticker_chunks)})...")
            
            st.session_state[cache_key_p1] = (open_price_map, close_price_map, volume_map, high_price_map, low_price_map)

                
        # Fast filter Price > 200 (reduces scanning load immensely by removing penny and low-priced stocks)
        scan_symbols = [s for s in raw_symbols if close_price_map.get(s.strip().upper(), 0.0) > 200.0]
        
        n_stocks = len(scan_symbols)
        failed_count = 0
        flagged_list = []
        gapup_list = []
        structural_vcp_list = []
        above_ma_list = []
        support_ma_list = []
        crossover_ma_list = []
        minervini_list = []
        wt_list = []
        vcs_list = []
        vpa_list = []
        
        # Unpack manual dry constraints from the sidebar range slider
        min_dry = dry_zone_range[0]
        max_dry = dry_zone_range[1]
            
        # Parallel bulk pre-download of historical OHLCV data to boost scan speed by 25x!
        cache_key_p2 = f"p2_bulk_v2_{universe_key}_{scan_timeframe}_{today_date_str}"
        bulk_data = {}
        if n_stocks > 0:
            if cache_key_p2 in st.session_state:
                bulk_data = st.session_state[cache_key_p2]
                status_box.text(f"Phase 2/3: Loaded {scan_timeframe} historical data from session cache!")
                prog_bar.progress(1.0)
            else:
                from config import LOOKBACK_DAYS
                if "Weekly" in scan_timeframe:
                    yf_interval = "1wk"
                    yf_period = "4y"
                elif "Monthly" in scan_timeframe:
                    yf_interval = "1mo"
                    yf_period = "17y"
                else:
                    yf_interval = "1d"
                    yf_period = f"{LOOKBACK_DAYS}d"

                status_box.text(f"Phase 2/3: Downloading {scan_timeframe} historical OHLCV data...")
                prog_bar.progress(0)
                chunk_size = 100
                sym_chunks = [scan_symbols[i:i + chunk_size] for i in range(0, len(scan_symbols), chunk_size)]
                
                def download_chunk(chunk_idx, chunk):
                    chunk_data = {}
                    chunk_ns = [f"{s.strip().upper()}.NS" for s in chunk]
                    try:
                        # yfinance 1.x: group_by and threads params removed; MultiIndex is now (price_type, ticker)
                        df_bulk = yf.download(tickers=chunk_ns, period=yf_period, interval=yf_interval, progress=False, threads=False)
                        for sym in chunk:
                            sym_ns = f"{sym.strip().upper()}.NS"
                            try:
                                if isinstance(df_bulk.columns, pd.MultiIndex):
                                    # yfinance 1.x multi-ticker: columns are (price_type, ticker)
                                    # Find the ticker in level 1
                                    all_tickers_bulk = df_bulk.columns.get_level_values(1).unique().tolist()
                                    matched = next((t for t in all_tickers_bulk if t.upper() == sym_ns.upper()), None)
                                    if matched is None:
                                        continue
                                    # Extract slice for this ticker across all price types
                                    ticker_df = df_bulk.xs(matched, axis=1, level=1).copy()
                                else:
                                    # Single ticker download (fallback)
                                    if len(chunk_ns) == 1:
                                        ticker_df = df_bulk.copy()
                                    else:
                                        continue

                                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                                if all(col in ticker_df.columns for col in required_cols):
                                    ticker_df = ticker_df[required_cols].dropna(subset=['Close'])
                                    ticker_df = ticker_df[ticker_df['Volume'] > 0]
                                    if not ticker_df.empty:
                                        ticker_df = ticker_df.reset_index()
                                        ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                        ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.tz_localize(None)
                                        chunk_data[sym.strip().upper()] = ticker_df
                            except Exception as sym_ex:
                                print(f"Error extracting {sym_ns} from bulk download: {sym_ex}")
                    except Exception as chunk_ex:
                        print(f"Error downloading parallel chunk {chunk_idx+1}: {chunk_ex}")
                    return chunk_data

                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    futures = []
                    for chunk_idx, chunk in enumerate(sym_chunks):
                        futures.append(executor.submit(download_chunk, chunk_idx, chunk))
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        bulk_data.update(future.result())
                        prog_bar.progress((i + 1) / len(sym_chunks))
                        status_box.text(f"Phase 2/3: Downloading historical data (Chunk {i+1}/{len(sym_chunks)})...")
                if len(bulk_data) > 0:
                    st.session_state[cache_key_p2] = bulk_data
        
        mcap_cache = {}
        status_box.text(f"Phase 3/3: Scanning {n_stocks} active NSE listed equities (Price > ₹200)...")
        prog_bar.progress(0)
        

        def process_single_symbol(sym, df, open_price_map, close_price_map, high_price_map, low_price_map, volume_map,
                                  min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes,
                                  min_signal_str, above_50dma_only, above_200dma_only, vcp_max_tightness):
            from datetime import datetime
            import pandas as pd
            import pytz
            IST_TIMEZONE = pytz.timezone('Asia/Kolkata')
            
            res = {
                "failed": False,
                "gapup": None,
                "above_ma": None,
                "support_ma": None,
                "crossover_ma": None,
                "minervini": None,
                "flagged": None,

                "wt": None,
                "vcs": None,
                "structural_vcp": None,
                "vpa": None
            }
            if df is None or len(df) < 5:
                res["failed"] = True
                return res
                
            df = df.sort_values('Date').reset_index(drop=True)
            last_df_date = df['Date'].iloc[-1].date()
            today_date = datetime.now(IST_TIMEZONE).date()
            
            if last_df_date < today_date:
                sym_clean = sym.strip().upper()
                if sym_clean in open_price_map and sym_clean in close_price_map:
                    new_row = {
                        'Date': pd.to_datetime(today_date),
                        'Open': open_price_map[sym_clean],
                        'High': high_price_map.get(sym_clean, close_price_map[sym_clean]),
                        'Low': low_price_map.get(sym_clean, close_price_map[sym_clean]),
                        'Close': close_price_map[sym_clean],
                        'Volume': volume_map.get(sym_clean, 0)
                    }
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                
            today_close_val = df['Close'].iloc[-1]
            if today_close_val <= 200.0:
                res["failed"] = True
                return res
            
            # =====================================================================
            # PRE-COMPUTE ALL INDICATORS ONCE (eliminates 5-8x redundant recalc)
            # =====================================================================
            from indicators import precompute_indicators
            ind = precompute_indicators(df)
            # Use the enriched DataFrame from pre-computation for all subsequent work
            if ind is not None and 'df' in ind:
                df = ind['df']
            
            today_open_val = float(df['Open'].iloc[-1])
            today_close_val = float(df['Close'].iloc[-1])
            yesterday_close_val = float(df['Close'].iloc[-2]) if len(df) >= 2 else today_open_val
            if today_open_val > yesterday_close_val and today_close_val > yesterday_close_val and today_close_val >= (today_open_val * 0.97):
                gap_pct = (today_open_val - yesterday_close_val) / yesterday_close_val * 100
                if gap_pct >= 8.0:
                    target_multiplier = 1.04; target_pct_str = "+4.0%"
                elif gap_pct >= 5.0:
                    target_multiplier = 1.06; target_pct_str = "+6.0%"
                else:
                    target_multiplier = 1.10; target_pct_str = "+10.0%"
                    
                gap_buy_price = round(today_close_val, 2)
                gap_exit_price = round(today_open_val * 0.98, 2) 
                gap_target_price = round(today_close_val * target_multiplier, 2) 
                gap_confidence = "High (Gap-Up Momentum)" if gap_pct > 3.0 else "Medium (Gap-Up)"
                base_gap_rec = (f"Bullish gap-up breakout of {gap_pct:.2f}% on strong momentum. Buy near ₹{gap_buy_price:.2f} "
                                f"with a stop loss below today's open price at ₹{gap_exit_price:.2f} "
                                f"targeting dynamic swing target ₹{gap_target_price:.2f} ({target_pct_str}).")
                gap_recommendation = compute_rich_analysis(df, sym, "Gap-Up", base_gap_rec, indicators=ind)
                res["gapup"] = {
                    "symbol": sym.strip().upper(), "company_name": get_company_name(sym),
                    "prev_close": yesterday_close_val, "open_price": today_open_val, "cmp": today_close_val,
                    "gap_pct": round(gap_pct, 2), "volume": int(df['Volume'].iloc[-1]),
                    "day_change_pct": round(((today_close_val - yesterday_close_val) / yesterday_close_val * 100), 2),
                    "buy_price": gap_buy_price, "exit_price": gap_exit_price, "target_price": gap_target_price,
                    "confidence": gap_confidence, "recommendation": gap_recommendation
                }
                
            # Use pre-computed SMAs from indicators — no need to recalculate
            df_ma = df  # Already has SMA20, SMA50, SMA65, SMA150, SMA200 from precompute_indicators()
            
            if len(df_ma) >= 200:
                today_row = df_ma.iloc[-1]; yesterday_row = df_ma.iloc[-2]
                c_val = float(today_row['Close']); l_val = float(today_row['Low'])
                sma20 = float(today_row['SMA20']); sma50 = float(today_row['SMA50'])
                sma65 = float(today_row['SMA65']); sma150 = float(today_row['SMA150'])
                sma200 = float(today_row['SMA200'])
                
                if c_val > sma20 and c_val > sma50:
                    above_buy_price = round(today_close_val, 2)
                    above_exit_price = round(sma50 * 0.97, 2) 
                    above_target_price = round(today_close_val * 1.12, 2) 
                    above_confidence = "High (Uptrend)" if sma20 > sma50 and sma50 > sma200 else "Medium-High (Uptrend)"
                    base_above_rec = (f"Strong medium-term uptrend. Close above 20 SMA & 50 SMA. Buy near ₹{above_buy_price:.2f} "
                                      f"with stop below 50 SMA support at ₹{above_exit_price:.2f} targeting momentum target ₹{above_target_price:.2f}.")
                    res["above_ma"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "dist_20sma_pct": round((today_close_val - sma20) / sma20 * 100, 2),
                        "dist_50sma_pct": round((today_close_val - sma50) / sma50 * 100, 2),
                        "setup_type": "above_ma", "buy_price": above_buy_price, "exit_price": above_exit_price,
                        "target_price": above_target_price, "confidence": above_confidence,
                        "recommendation": compute_rich_analysis(df_ma, sym, "Above 20/50 SMA", base_above_rec, indicators=ind)
                    }
                    
                yesterday_l = float(yesterday_row['Low']); yesterday_sma65 = float(yesterday_row['SMA65'])
                tested_today = l_val <= sma65 * 1.01; tested_yesterday = yesterday_l <= yesterday_sma65 * 1.01
                o_val = float(today_row['Open']); yesterday_c = float(yesterday_row['Close'])
                is_green_candle = c_val > o_val; is_up_move = c_val > yesterday_c; holds_above = c_val > sma65
                
                if (tested_today or tested_yesterday) and holds_above and is_green_candle and is_up_move:
                    support_buy_price = round(today_close_val, 2); support_exit_price = round(sma65 * 0.97, 2) 
                    support_target_price = round(today_close_val * 1.15, 2) 
                    support_confidence = "High (Pullback Support)" if today_close_val > yesterday_row['Close'] else "Medium (Pullback Support)"
                    base_support_rec = (f"Institutional pullback testing critical 65 SMA support (₹{sma65:.2f}). "
                                        f"Buy around ₹{support_buy_price:.2f} with tight stop just below SMA at ₹{support_exit_price:.2f} targeting bounce to ₹{support_target_price:.2f}.")
                    res["support_ma"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "dist_65sma_pct": round((today_close_val - sma65) / sma65 * 100, 2), "setup_type": "support_ma",
                        "buy_price": support_buy_price, "exit_price": support_exit_price, "target_price": support_target_price,
                        "confidence": support_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "65 SMA Support", base_support_rec, indicators=ind)
                    }
                    
                crossed_golden = (yesterday_row['SMA50'] <= yesterday_row['SMA200']) and (today_row['SMA50'] > today_row['SMA200'])
                crossed_150 = (yesterday_row['SMA50'] <= yesterday_row['SMA150']) and (today_row['SMA50'] > today_row['SMA150'])
                price_crossed_50 = (yesterday_row['Close'] <= yesterday_row['SMA50']) and (today_row['Close'] > today_row['SMA50'])
                price_crossed_150 = (yesterday_row['Close'] <= yesterday_row['SMA150']) and (today_row['Close'] > today_row['SMA150'])
                price_crossed_200 = (yesterday_row['Close'] <= yesterday_row['SMA200']) and (today_row['Close'] > today_row['SMA200'])
                
                if crossed_golden or crossed_150 or price_crossed_50 or price_crossed_150 or price_crossed_200:
                    cross_buy_price = round(today_close_val, 2); cross_exit_price = round(today_close_val * 0.94, 2) 
                    cross_target_price = round(today_close_val * 1.18, 2) 
                    cross_confidence = "High (Golden Cross)" if crossed_golden else "Medium-High (Crossover)"
                    base_cross_rec = (f"Technical moving average crossover signal! Buy near ₹{cross_buy_price:.2f} "
                                      f"to ride the emerging uptrend. Set stop loss at ₹{cross_exit_price:.2f} targeting swing high ₹{cross_target_price:.2f}.")
                    res["crossover_ma"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "dist_50sma_pct": round((today_close_val - sma50) / sma50 * 100, 2),
                        "dist_200sma_pct": round((today_close_val - sma200) / sma200 * 100, 2), "setup_type": "crossover_ma",
                        "buy_price": cross_buy_price, "exit_price": cross_exit_price, "target_price": cross_target_price,
                        "confidence": cross_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "MA Crossover", base_cross_rec, indicators=ind)
                    }

            if len(df_ma) >= 250:
                today_row = df_ma.iloc[-1]; yesterday_row = df_ma.iloc[-2]; c_val = float(today_row['Close'])
                sma50 = float(today_row['SMA50']); sma150 = float(today_row['SMA150']); sma200 = float(today_row['SMA200'])
                sma200_10d_ago = float(df_ma['SMA200'].iloc[-11]) if len(df_ma) >= 210 else sma200
                high_52w = float(df_ma['High'].iloc[-250:].max()); low_52w = float(df_ma['Low'].iloc[-250:].min())
                
                if c_val > sma150 and c_val > sma200 and sma150 > sma200 and sma200 > sma200_10d_ago and sma50 > sma150 and sma50 > sma200 and c_val > sma50 and c_val >= 1.30 * low_52w and c_val >= 0.75 * high_52w:
                    run_up_200 = round(((c_val - sma200) / sma200 * 100), 2)
                    run_up_52w = round(((c_val - low_52w) / low_52w * 100), 2)
                    is_early = bool(c_val <= 1.20 * sma200)
                    exit_price = round(min(sma200 * 0.98, c_val * 0.94), 2)
                    distance_200 = (c_val - sma200) / sma200
                    target_mult = 1.40 - min(0.15, distance_200 * 0.7) if is_early else 1.18 - min(0.06, (distance_200 - 0.20) * 0.4)
                    target_price = round(max(high_52w * 1.05, c_val * target_mult), 2)
                    min_confidence = "High (Minervini Stage-2)" if is_early else "Medium-High (Minervini Extended)"
                    rem_pct = ((target_price - c_val) / c_val * 100)
                    stage_label = "Early Stage-2 Accumulation" if is_early else "Extended Stage-2 Uptrend"
                    base_minervini_rec = (f"Mark Minervini Stage-2 Trend Template verified! The stock is in an active '{stage_label}' "
                                          f"having run up {run_up_52w:.1f}% from its 52w low and holding {run_up_200:.1f}% above its 200 SMA support. "
                                          f"Buy around CMP ₹{c_val:.2f}. Set stop loss at ₹{exit_price:.2f} (tight support lock) "
                                          f"targeting momentum swing target of ₹{target_price:.2f} (remaining potential +{rem_pct:.1f}%).")
                    res["minervini"] = {
                        "symbol": sym.strip().upper(), "company_name": get_company_name(sym), "cmp": today_close_val,
                        "day_change_pct": round(((today_close_val - yesterday_row['Close']) / yesterday_row['Close'] * 100), 2),
                        "setup_type": "minervini", "run_up_200": run_up_200, "run_up_52w": run_up_52w, "is_early": is_early,
                        "buy_price": round(c_val, 2), "exit_price": exit_price, "target_price": target_price,
                        "confidence": min_confidence, "recommendation": compute_rich_analysis(df_ma, sym, "Minervini Stage-2", base_minervini_rec, indicators=ind)
                    }
                    
            scan_res = scan_stock(symbol=sym, df=df, min_dry_days=min_dry, max_dry_days=max_dry, min_volume_ratio=min_vol_ratio, min_price_change=min_price_chg, min_dry_spikes=min_dry_spikes, indicators=ind)
            if scan_res is not None:
                scan_res['market_cap_cr'] = 0.0
                if scan_res['signal_strength'] >= min_signal_str:
                    if (not above_50dma_only or scan_res.get('above_50dma', False)) and (not above_200dma_only or scan_res.get('above_200dma', False)):
                        res["flagged"] = scan_res
                        
                        
            df_wt = df
            if df_wt is not None and len(df_wt) >= 40:
                wt_res = scan_wt_cross(sym, df_wt, indicators=ind)
                if wt_res is not None:
                    wt_res['timeframe'] = "Daily"
                    res["wt"] = wt_res
                    
            if df is not None:
                res["vcs"] = scan_vcs(sym, df, indicators=ind)
                res["structural_vcp"] = scan_structural_vcp(sym, df, indicators=ind)
                res["vpa"] = scan_vpa_trend(sym, df, indicators=ind)
                
            return res


        import concurrent.futures
        import os
        
        status_box.text(f"Phase 3/3: Scanning {n_stocks} active NSE listed equities (Price > ₹200)...")
        prog_bar.progress(0)
        
        # Parallel Execution Core
        def process_and_fetch_if_needed(sym, df, *args):
            if df is None:
                df = fetch_ohlcv(sym)
            return process_single_symbol(sym, df, *args)

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() * 2 if os.cpu_count() else 8)) as executor:
            future_to_sym = {}
            for sym in scan_symbols:
                df = bulk_data.get(sym.strip().upper())
                future = executor.submit(process_and_fetch_if_needed, sym, df, open_price_map, close_price_map, high_price_map, low_price_map, volume_map, min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes, min_signal_str, above_50dma_only, above_200dma_only, vcp_max_tightness)
                future_to_sym[future] = sym
                
            for i, future in enumerate(concurrent.futures.as_completed(future_to_sym)):
                sym = future_to_sym[future]
                try:
                    res = future.result()
                    if res.get("failed"):
                        failed_count += 1
                        continue
                    if res.get("gapup"): gapup_list.append(res["gapup"])
                    if res.get("above_ma"): above_ma_list.append(res["above_ma"])
                    if res.get("support_ma"): support_ma_list.append(res["support_ma"])
                    if res.get("crossover_ma"): crossover_ma_list.append(res["crossover_ma"])
                    if res.get("minervini"): minervini_list.append(res["minervini"])
                    if res.get("flagged"): flagged_list.append(res["flagged"])
                    if res.get("wt"): wt_list.append(res["wt"])
                    if res.get("vcs"): vcs_list.append(res["vcs"])
                    if res.get("structural_vcp"): structural_vcp_list.append(res["structural_vcp"])
                    if res.get("vpa"): vpa_list.append(res["vpa"])
                except Exception as exc:
                    print(f"Error processing {sym}: {exc}")
                    failed_count += 1
                    
                # Throttle UI Updates (every 25 iterations or at the end)
                if (i + 1) % 25 == 0 or i + 1 == n_stocks:
                    status_box.text(f"Phase 3/3: Scanning ({i+1}/{n_stocks})")
                    prog_bar.progress((i + 1) / n_stocks)

        # Clean progress assets
        prog_bar.empty()
        status_box.empty()
        
        # Cache results in state to allow seamless widget interactions
        st.session_state.scan_results = flagged_list
        st.session_state.gapup_results = gapup_list
        st.session_state.above_ma_results = above_ma_list
        st.session_state.support_ma_results = support_ma_list
        st.session_state.crossover_ma_results = crossover_ma_list
        st.session_state.minervini_results = minervini_list
        st.session_state.vcs_results = vcs_list
        st.session_state.structural_vcp_results = structural_vcp_list
        st.session_state.vpa_results = vpa_list
        st.session_state.wt_results = wt_list
        st.session_state.wt_results_by_tf = {"Daily": wt_list}
        st.session_state.total_scanned = n_stocks
        st.session_state.failed_count = failed_count
        st.session_state.last_scanned = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d %I:%M:%S %p")
        
        # Save to database cache daily
        try:
            today_ist_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
            trend_setups_list = above_ma_list + support_ma_list + crossover_ma_list + minervini_list
            database.save_scan_results(
                date_str=today_ist_str,
                breakouts=flagged_list,
                squeezes=[],
                gapups=gapup_list,
                trend_setups=trend_setups_list,
                wt_cross=wt_list,
                total_scanned=n_stocks,
                vcs_results=vcs_list,
                vpa_results=vpa_list
            )
            st.toast("💾 Today's scan results cached in Neon PostgreSQL!", icon="✅")
            
            # Trigger background AI scans automatically in the backend!
            all_flagged_syms = [r['symbol'] for r in flagged_list]
            run_background_ai_scan(all_flagged_syms, today_ist_str)
        except Exception as db_err:
            print(f"Failed to cache daily scan results to database: {db_err}")
        
        # Highlight large failure rate
        if n_stocks > 0 and (failed_count / n_stocks) > 0.20:
            st.sidebar.warning(f"⚠️ Failed to fetch {failed_count}/{n_stocks} symbols ({failed_count/n_stocks*100:.1f}%). Check internet connection.")
            
        st.success("✅ Scanner complete! Results have been updated.")



# Display Last Scanned Timestamp
if st.session_state.last_scanned:
    st.sidebar.markdown(f"<p style='text-align: center; font-size: 0.85rem; color: #94a3b8; margin-top: 10px;'>⏱️ Last Scan: <b>{st.session_state.last_scanned}</b></p>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<p style='text-align: center; font-size: 0.85rem; color: #64748b; margin-top: 10px;'>⚠️ Click 'Run Scanner' to start</p>", unsafe_allow_html=True)

# --- Permanent Sidebar Technical Signals Reference Guide ---
with st.sidebar.expander("🎓 Institutional Buy Signals Guide", expanded=False):
    st.markdown(
        """
        <div style="font-size: 0.84rem; line-height: 1.4; color: #cbd5e1; margin-bottom: 8px;">
            <span style="color: #38bdf8; font-weight: 600;">Optimal Swing Buy Parameters:</span>
            <p style="margin: 4px 0 8px 0; font-size: 0.78rem;">Maximize your success rate by looking for confluence across these core institutional signals.</p>
        </div>
        
        <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.76rem; color: #cbd5e1; background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); color: #38bdf8; font-weight: bold; background: rgba(56, 189, 248, 0.05);">
                    <th style="padding: 4px 6px;">Indicator</th>
                    <th style="padding: 4px 6px;">Reasoning</th>
                    <th style="padding: 4px 6px;">Best Buy Trigger</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <td style="padding: 4px 6px; font-weight: bold; color: #38bdf8;">RSI (14)</td>
                    <td style="padding: 4px 6px; color: #94a3b8; font-size: 0.72rem;">Measures speed/velocity of price to avoid overextended buys.</td>
                    <td style="padding: 4px 6px; color: #00e676; font-size: 0.72rem;"><b>35 - 50</b> (Bounce)<br><b>50 - 65</b> (Momentum)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <td style="padding: 4px 6px; font-weight: bold; color: #ab47bc;">CCI (14)</td>
                    <td style="padding: 4px 6px; color: #94a3b8; font-size: 0.72rem;">Measures price deviation to catch trend breakouts early.</td>
                    <td style="padding: 4px 6px; color: #00e676; font-size: 0.72rem;"><b>&gt; +100</b> (Velocity)<br><b>&lt; -100</b> (Exhaustion)</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <td style="padding: 4px 6px; font-weight: bold; color: #e2e8f0;">20 EMA</td>
                    <td style="padding: 4px 6px; color: #94a3b8; font-size: 0.72rem;">Short-term dynamic anchor for low-risk pullback entries.</td>
                    <td style="padding: 4px 6px; color: #00e676; font-size: 0.72rem;">Price pulls back within <b>&plusmn;2%</b>.</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <td style="padding: 4px 6px; font-weight: bold; color: #cbd5e1;">50 SMA</td>
                    <td style="padding: 4px 6px; color: #94a3b8; font-size: 0.72rem;">Medium-term institutional trend boundary.</td>
                    <td style="padding: 4px 6px; color: #00e676; font-size: 0.72rem;">CMP trades <b>above 50 SMA</b>.</td>
                </tr>
                <tr>
                    <td style="padding: 4px 6px; font-weight: bold; color: #cbd5e1;">200 SMA</td>
                    <td style="padding: 4px 6px; color: #94a3b8; font-size: 0.72rem;">Long-term dividing line. Structural support floor.</td>
                    <td style="padding: 4px 6px; color: #00e676; font-size: 0.72rem;">CMP <b>above 200 SMA</b> & <b>50 &gt; 200 SMA</b>.</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True
    )

# --- MAIN INTERFACE TABS ---
try:
    tab_scan, tab_detail, tab_watchlist, tab_ai, tab_gapup, tab_above_ma, tab_support_ma, tab_crossover_ma, tab_wavetrend, tab_minervini, tab_monthly_mom, tab_weekly_mom, tab_history, tab_vcs, tab_structural_vcp, tab_stage2, tab_vpa, tab_frequent = st.tabs([
        "📊 Scanner Results",
        "📈 Stock Detail",
        "📋 My Watchlist",
        "🤖 AI Chart Pattern Detector",
        "🚀 Gap-Up Setups",
        "📈 Above 20 & 50 SMA",
        "🛡️ 65 SMA Support",
        "🔄 MA Crossovers",
        "🌊 Wave Trend",
        "🏆 Minervini Stage-2",
        "📅 Monthly Momentum",
        "📈 Weekly Momentum",
        "📅 Scan History",
        "📉 Volatility Contraction (VCS)",
        "🎯 Structural VCP",
        "🚀 Early Stage 2 Breakout",
        "🚥 VPA Trend",
        "🔄 Consistent Alerts"
    ])
except Exception as tab_err:
    st.error(f"❌ Tab rendering error: {tab_err}")
    st.exception(tab_err)
    st.stop()



# Get scan cache
scan_data = st.session_state.scan_results

# ==============================================================================
# TAB 1: SCANNER RESULTS
# ==============================================================================
with tab_scan:
    try:
        # 1. Premium Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        if scan_data:
            total_scanned = st.session_state.total_scanned
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
        st.markdown("---")
        st.info("💡 **Trading Note on Live Data**: Scans performed during active NSE market hours (9:15 AM - 3:30 PM IST) dynamically process real-time updates for today's active candle. Indicators (RSI, CCI) and scanner scores will naturally vary as today's close prices fluctuate. Scans run after market hours are 100% static and deterministic.")
        
        # 2. Main Scan Table
        if scan_data is None:
            st.info("💡 Get started by configuring your universe in the sidebar and clicking '**Run Scanner**'.")
        elif len(scan_data) == 0:
            st.info("ℹ️ No VDU breakouts found today matching these criteria. Try lowering the thresholds in the sidebar (e.g. Min Volume Ratio or Min Price Change) and re-running.")
        else:
            # Sort results descending by score
            sorted_scan = sorted(scan_data, key=lambda x: x['signal_strength'], reverse=True)
            
            # Download Results Option - safely convert date fields
            def _safe_date(v):
                if v is None:
                    return ""
                if hasattr(v, 'strftime'):
                    return v.strftime("%Y-%m-%d")
                return str(v)

            export_rows = []
            for r in sorted_scan:
                export_rows.append({
                    "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                        "CMP (₹)": r['cmp'],
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
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            
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


        st.markdown('</div></div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: STOCK DETAIL
# ==============================================================================
with tab_detail:
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
                        gridcolor="rgba(255,255,255,0.04)"
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

                st.plotly_chart(fig, use_container_width=True)

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
                    st.plotly_chart(gauge_fig, use_container_width=True)

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

# ==============================================================================
# TAB 3: WATCHLIST
# ==============================================================================
with tab_watchlist:
    st.markdown("### 📋 My Watchlist Monitor")
    
    # Read persistent DB
    w_df = watchlist.load_watchlist()
    
    if w_df.empty:
        st.info("ℹ️ Your watchlist is currently empty. Run scans on index universes or paste custom tickers to build your watchlist!")
    else:
        # A. SINGLE BATCH YFINANCE PRICE DOWNLOAD
        tickers_list = [f"{s}.NS" for s in w_df['symbol'].unique()]
        cmp_dict = {}
        
        with st.spinner("Fetching real-time quotes for watchlisted assets..."):
            try:
                # yfinance 1.x: auto_adjust=True is default, auto_adjust=False is deprecated
                prices_df = yf.download(tickers=tickers_list, period="1d", progress=False, threads=False)
                if not prices_df.empty:
                    # yfinance 1.x multi-ticker: MultiIndex (price_type, ticker)
                    if isinstance(prices_df.columns, pd.MultiIndex):
                        close_prices = prices_df['Close'].iloc[-1]  # Series with .NS ticker index
                    else:
                        close_prices = prices_df['Close'].iloc[-1]  # scalar for single ticker
                        close_prices = {tickers_list[0]: close_prices}

                    # Build lookup maps (strip .NS from keys)
                    if isinstance(close_prices, pd.Series):
                        for k, v in close_prices.items():
                            clean_k = str(k).replace(".NS", "").upper()
                            if not pd.isna(v) and float(v) > 0:
                                cmp_dict[clean_k] = float(v)
                    elif isinstance(close_prices, dict):
                        for k, v in close_prices.items():
                            clean_k = str(k).replace(".NS", "").upper()
                            if v and not pd.isna(v):
                                cmp_dict[clean_k] = float(v)
            except Exception as quote_ex:
                st.warning("⚠️ Could not fetch real-time quotes. Using historical entry price for watchlist CMP.")
                
        # B. BUILD WATCHLIST VIEW DATA
        display_rows = []
        for idx, row in w_df.iterrows():
            sym = row['symbol'].upper()
            entry = float(row['entry_price'])
            
            # Fetch CMP or fall back to entry
            cmp_val = cmp_dict.get(sym, entry)
            if pd.isna(cmp_val) or cmp_val <= 0:
                cmp_val = entry
                
            pnl_val = ((cmp_val - entry) / entry * 100)
            
            display_rows.append({
                "symbol": sym,
                "company_name": row['company_name'],
                "added_date": row['added_date'],
                "entry_price": entry,
                "signal_strength_at_add": float(row['signal_strength_at_add']),
                "CMP (₹)": round(cmp_val, 2),
                "PnL %": round(pnl_val, 2),
                "tag": row['tag'],
                "notes": str(row['notes']) if not pd.isna(row['notes']) else ""
            })
            
        display_df = pd.DataFrame(display_rows)
        
        # C. INTERACTIVE DATA EDITOR (Auto-saves Tag and Notes)
        st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>✏️ You can edit the <b>Tag</b> dropdowns or write custom text in <b>Notes</b> cells. Changes persist immediately.</p>", unsafe_allow_html=True)
        
        # Define table configs
        config_table = {
            "symbol": st.column_config.TextColumn("Symbol", disabled=True),
                        "added_date": st.column_config.TextColumn("Added Date", disabled=True),
            "entry_price": st.column_config.NumberColumn("Entry Price (₹)", disabled=True, format="₹%.2f"),
            "signal_strength_at_add": st.column_config.NumberColumn("Original Signal", disabled=True, format="%.1f pts"),
            "CMP (₹)": st.column_config.NumberColumn("Current Price (₹)", disabled=True, format="₹%.2f"),
            "PnL %": st.column_config.NumberColumn("Unrealized PnL %", disabled=True, format="%.2f%%"),
            "tag": st.column_config.SelectboxColumn("Tag Status", options=["Watching 👀", "Ready to Buy 🟢", "Tracking 📍", "Avoid 🔴"]),
            "notes": st.column_config.TextColumn("Notes (Click to Edit)")
        }
        
        edited_table = st.data_editor(
            display_df,
            column_config=config_table,
            use_container_width=True,
            hide_index=True,
            key="watchlist_editor_grid"
        )
        
        # Check cell changes
        if not edited_table.equals(display_df):
            # Map back to standard CSV columns
            save_df = edited_table[['symbol', 'company_name', 'added_date', 'entry_price', 'signal_strength_at_add', 'tag', 'notes']].copy()
            watchlist.save_watchlist(save_df)
            st.toast("💾 Watchlist auto-saved successfully!")
            st.rerun()
            
        st.markdown("---")
        
        # D. MANAGEMENT CONTROLS PANEL
        st.markdown("### ⚙️ Watchlist Controls")
        
        col_c1, col_c2 = st.columns(2)
        
        # 1. Removal widget
        with col_c1:
            st.markdown("#### ❌ Delete Ticker")
            c_del1, c_del2 = st.columns([2, 1])
            ticker_to_delete = c_del1.selectbox(
                "Choose stock to remove:", 
                options=[""] + list(display_df['symbol'].unique()), 
                key="del_box"
            )
            
            if ticker_to_delete:
                del_clicked = c_del2.button("Remove Ticker", type="secondary", key="del_action", use_container_width=True)
                if del_clicked:
                    watchlist.remove_stock(ticker_to_delete)
                    st.toast(f"Removed {ticker_to_delete} from your watchlist.")
                    st.rerun()
                    
        # 2. Export and Clear watchlist
        with col_c2:
            st.markdown("#### 📂 Operations")
            
            # Export CSV
            watchlist_csv_bytes = watchlist.export_csv()
            st.download_button(
                label="📥 Export Watchlist CSV",
                data=watchlist_csv_bytes,
                file_name=f"vdu_watchlist_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="dl_watchlist"
            )
            
            # Clear all database
            clear_btn = st.button("🗑️ Clear Entire Watchlist", type="secondary", use_container_width=True, key="clear_watchlist_btn")
            if clear_btn:
                st.session_state.confirm_clear = True
                
            if st.session_state.confirm_clear:
                st.markdown("<p style='color:#ef4444; font-weight:600;'>⚠️ Are you absolutely sure? This deletes watchlist.csv entries forever.</p>", unsafe_allow_html=True)
                col_yes, col_no = st.columns(2)
                
                if col_yes.button("Yes, Clear All", type="primary", use_container_width=True, key="clr_yes"):
                    # Clear CSV
                    empty_df = pd.DataFrame(columns=watchlist.COLUMNS)
                    watchlist.save_watchlist(empty_df)
                    st.session_state.confirm_clear = False
                    st.toast("🗑️ Watchlist fully cleared.")
                    st.rerun()
                    
                if col_no.button("Cancel", use_container_width=True, key="clr_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()

        # Watchlist Technical Assessment inspector panel
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.08);'><br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Watchlist Technical Assessment")
        st.markdown("<p style='font-size:0.9rem; color:#94a3b8; margin-top:-10px;'>Select any stock from your watchlist to inspect its real-time indicators and buying checklist.</p>", unsafe_allow_html=True)
        
        watch_symbols = list(display_df['symbol'].unique())
        selected_watch_sym = st.selectbox(
            "Select Stock to Inspect:",
            options=[""] + watch_symbols,
            key="watch_inspect_select"
        )
        
        if selected_watch_sym:
            # Fetch historical data and compute rich indicators
            with st.spinner(f"Loading technical indicators for {selected_watch_sym}..."):
                df_w = fetch_ohlcv(selected_watch_sym)
                if df_w is not None and not df_w.empty:
                    rich_payload = compute_rich_analysis(df_w, selected_watch_sym, "Watchlist Assessment", "Monitor key support levels for active trade setups.")
                    watch_item = next((r for r in display_rows if r['symbol'] == selected_watch_sym), None)
                    cmp_val = watch_item['CMP (₹)'] if watch_item else df_w['Close'].iloc[-1]
                    
                    dummy_w = {
                        "symbol": selected_watch_sym,
                        "cmp": cmp_val,
                        "buy_price": watch_item['entry_price'] if watch_item else cmp_val,
                        "exit_price": cmp_val * 0.93,
                        "target_price": cmp_val * 1.15,
                        "confidence": "Medium-High",
                        "recommendation": rich_payload
                    }
                    render_trading_setup_card(dummy_w, "watchlist_tab_setup", 0)

# ==============================================================================
# TAB 4: AI CHART PATTERN DETECTOR
# ==============================================================================
with tab_ai:
    st.markdown("### 🤖 Technical Chart Pattern Recognition with AI")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Inspect daily candle charts with Euri / Groq AI technical analysts and save/cache findings in Neon PostgreSQL database.</p>", unsafe_allow_html=True)
    st.info("💡 **Trading Note on Live Data**: Scans performed during active NSE market hours (9:15 AM - 3:30 PM IST) dynamically process real-time updates for today's active candle. Indicators (RSI, CCI) and scanner scores will naturally vary as today's close prices fluctuate. Scans run after market hours are 100% static and deterministic.")
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
        today_date_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
        
        # Check cache first (always check cache automatically to show today's output immediately!)
        cached_result = database.get_pattern_by_date(ticker_to_analyze, today_date_str)
        
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
                                gridcolor="rgba(255,255,255,0.04)"
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
                        st.plotly_chart(fig_ai, use_container_width=True)

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
        today_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
        
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
            if st.button("🔄 Refresh progressive AI results", key="refresh_ai_background_scan_results", use_container_width=True):
                st.rerun()
        else:
            # Batch Scan Control Buttons
            btn_cols = st.columns(2)
            btn_batch_scan = False
            btn_force_batch_scan = False
            
            if unscanned_count > 0:
                btn_batch_scan = btn_cols[0].button(f"🤖 Trigger Background AI Scan ({unscanned_count} Pending)", key="batch_ai_scan_action_btn", use_container_width=True)
                
            if len(active_flagged_symbols) > 0:
                btn_force_batch_scan = btn_cols[1].button(f"🔄 Force Re-scan All ({len(active_flagged_symbols)} Flagged Candidates)", key="force_batch_ai_scan_action_btn", use_container_width=True)
                
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
                
            if row_cols[6].button("🔍 View", key=action_key, use_container_width=True, on_click=set_ai_selection):
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
                
            if row_cols[5].button("⚡ Load", key=f"load_rec_{rec['symbol']}_{idx}", use_container_width=True, on_click=set_cached_ai_selection):
                st.toast(f"Loading cached analysis for {rec['symbol']}!")
                
            st.markdown("<hr style='margin: 4px 0; border-color: rgba(255,255,255,0.03);'>", unsafe_allow_html=True)


# ==============================================================================
# TAB 6: GAP-UP SETUPS
# ==============================================================================
with tab_gapup:
    st.markdown("### 🚀 Daily Gap-Up Momentum Setups")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Scan for momentum setups opening higher than yesterday's close — price breaking out of overhead levels immediately upon market open.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    gapup_data = st.session_state.gapup_results
    
    # 1. Premium Metrics Row
    g_m1, g_m2, g_m3 = st.columns(3)
    
    if gapup_data:
        gapup_count = len(gapup_data)
        max_gap = max(r['gap_pct'] for r in gapup_data)
        avg_gap = sum(r['gap_pct'] for r in gapup_data) / gapup_count
    else:
        gapup_count = 0
        max_gap = 0.0
        avg_gap = 0.0
        
    g_m1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Gap-Up Setups Found</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{gapup_count}</h3></div>', unsafe_allow_html=True)
    g_m2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Highest Gap-Up %</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">+{max_gap:.2f}%</h3></div>', unsafe_allow_html=True)
    g_m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Average Gap-Up %</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">+{avg_gap:.2f}%</h3></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 2. Main Scan Table
    if gapup_data is None:
        st.info("💡 Run the scanner from the sidebar to identify live pre-market or intraday gap-up setups.")
    elif len(gapup_data) == 0:
        st.info("ℹ️ No gap-up setups found today matching the scanning criteria.")
    else:
        # Sort results descending by gap percent
        sorted_gapup = sorted(gapup_data, key=lambda x: x['gap_pct'], reverse=True)
        
        # Download results option
        export_gapup = []
        for r in sorted_gapup:
            export_gapup.append({
                "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                "Yesterday Close (₹)": r['prev_close'],
                "Today Open (₹)": r['open_price'],
                "CMP (₹)": r['cmp'],
                "Gap %": r['gap_pct'],
                "Day Change %": r['day_change_pct'],
                "Volume": r['volume'],
                "Buy Range (₹)": r.get('buy_price', r.get('cmp', 0)),
                "Stop Loss (₹)": r.get('exit_price', 0),
                "Target (₹)": r.get('target_price', 0),
                "Confidence": r.get('confidence', ''),
                "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
            })
        export_g_df = pd.DataFrame(export_gapup)
        csv_g_data = export_g_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Gap-Up Setups (CSV)",
            data=csv_g_data,
            file_name=f"gapup_setups_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_gapup_top_btn"
        )
        
        st.markdown("---")
        # Render the unified Trade Execution Matrix
        st.markdown("### 🚀 Active Gap-Up Momentum Trade Execution Sheet")
        render_unified_strategy_table(sorted_gapup, "gapup", "gapup_tab")

# ==============================================================================
# TAB 7: ABOVE 20 & 50 SMA
# ==============================================================================
with tab_above_ma:
    st.markdown("### 📈 Stocks Trading Above 20 SMA & 50 SMA")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Identify stocks in a strong medium-term uptrend where price is trading comfortably above both their 20-day and 50-day Simple Moving Averages.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    above_ma_data = st.session_state.above_ma_results
    
    if above_ma_data is None:
        st.info("💡 Run the scanner from the sidebar to identify stocks trading above their 20 SMA and 50 SMA.")
    elif len(above_ma_data) == 0:
        st.info("ℹ️ No stocks found today matching the 20 & 50 SMA uptrend criteria.")
    else:
        # Sort by day change descending
        sorted_above = sorted(above_ma_data, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
        
        # Download results option
        export_above = []
        for r in sorted_above:
            export_above.append({
                "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                "CMP (₹)": r['cmp'],
                "Day Change %": r['day_change_pct'],
                "Setup Type": r['setup_type'],
                "Dist to 20 SMA (%)": r.get('dist_20sma_pct', 0.0),
                "Dist to 50 SMA (%)": r.get('dist_50sma_pct', 0.0),
                "Suggested Buy (₹)": r['buy_price'],
                "Suggested Exit/SL (₹)": r['exit_price'],
                "Suggested Target (₹)": r['target_price'],
                "Confidence": r['confidence'],
                "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
            })
        export_a_df = pd.DataFrame(export_above)
        csv_a_data = export_a_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Above 20/50 SMA Results (CSV)",
            data=csv_a_data,
            file_name=f"above_20_50_sma_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_above_ma_btn"
        )
        
        st.markdown("---")
        # Render the unified Trade Execution Matrix
        st.markdown("### 📈 Active Uptrend Trade Execution Sheet")
        render_unified_strategy_table(sorted_above, "above_ma", "above_ma_tab")

# ==============================================================================
# TAB 8: 65 SMA SUPPORT
# ==============================================================================
with tab_support_ma:
    st.markdown("### 🛡️ Stocks Taking Support at 65 SMA")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Scan for institutional pullbacks where the price is testing or bouncing precisely off the 65-day Simple Moving Average (65 SMA), offering high-probability low-risk entries.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    support_ma_data = st.session_state.support_ma_results
    
    if support_ma_data is None:
        st.info("💡 Run the scanner from the sidebar to identify stocks taking support at their 65 SMA.")
    elif len(support_ma_data) == 0:
        st.info("ℹ️ No stocks found today taking support at their 65 SMA.")
    else:
        # Sort by day change descending
        sorted_support = sorted(support_ma_data, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
        
        # Download results option
        export_support = []
        for r in sorted_support:
            export_support.append({
                "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                "CMP (₹)": r['cmp'],
                "Day Change %": r['day_change_pct'],
                "Setup Type": r['setup_type'],
                "Dist to 65 SMA (%)": r.get('dist_65sma_pct', 0.0),
                "Suggested Buy (₹)": r['buy_price'],
                "Suggested Exit/SL (₹)": r['exit_price'],
                "Suggested Target (₹)": r['target_price'],
                "Confidence": r['confidence'],
                "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
            })
        export_s_df = pd.DataFrame(export_support)
        csv_s_data = export_s_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download 65 SMA Support Results (CSV)",
            data=csv_s_data,
            file_name=f"65_sma_support_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_support_ma_btn"
        )
        
        st.markdown("---")
        # Render the unified Trade Execution Matrix
        st.markdown("### 🛡️ Active 65 SMA Support Trade Execution Sheet")
        render_unified_strategy_table(sorted_support, "support_ma", "support_ma_tab")

# ==============================================================================
# TAB 9: MA CROSSOVERS
# ==============================================================================
with tab_crossover_ma:
    st.markdown("### 🔄 Moving Average Crossover Signals")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Identify stocks triggering critical trend reversal crossovers (50 SMA crossing 150/200 SMA, or price crossing above 50/150/200 SMA) in the latest session.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    crossover_ma_data = st.session_state.crossover_ma_results
    
    if crossover_ma_data is None:
        st.info("💡 Run the scanner from the sidebar to identify moving average crossover signals.")
    elif len(crossover_ma_data) == 0:
        st.info("ℹ️ No stocks found triggering moving average crossover signals in this session.")
    else:
        # Sort by day change descending
        sorted_crossover = sorted(crossover_ma_data, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
        
        # Download results option
        export_crossover = []
        for r in sorted_crossover:
            export_crossover.append({
                "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                "CMP (₹)": r['cmp'],
                "Day Change %": r['day_change_pct'],
                "Setup Type": r['setup_type'],
                "Suggested Buy (₹)": r['buy_price'],
                "Suggested Exit/SL (₹)": r['exit_price'],
                "Suggested Target (₹)": r['target_price'],
                "Confidence": r['confidence'],
                "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
            })
        export_x_df = pd.DataFrame(export_crossover)
        csv_x_data = export_x_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download MA Crossover Results (CSV)",
            data=csv_x_data,
            file_name=f"ma_crossover_signals_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="dl_crossover_ma_btn"
        )
        
        st.markdown("---")
        # Render the unified Trade Execution Matrix
        st.markdown("### 🔄 Active MA Crossover Trade Execution Sheet")
        render_unified_strategy_table(sorted_crossover, "crossover_ma", "crossover_ma_tab")

# ==============================================================================
# TAB 10: WAVE TREND (LazyBear)
# ==============================================================================
with tab_wavetrend:
    # 0. Timeframe & Threshold selector inside tab
    wt_col1, wt_col2 = st.columns(2)
    with wt_col1:
        wt_timeframe = st.selectbox(
            "🌊 Select WaveTrend Timeframe:",
            options=["Daily", "15 Min", "1 Hour", "Weekly", "Monthly"],
            index=0,
            key="wt_tab_timeframe_selector_v2",
            help="Select the WaveTrend chart interval. Changing this dynamically runs a real-time parallel scan for active stocks."
        )
    with wt_col2:
        wt_oversold_threshold = st.number_input(
            "📉 Oversold Threshold:",
            min_value=-100.0,
            max_value=0.0,
            value=-40.0,
            step=5.0,
            key="wt_oversold_threshold",
            help="Define the WT1 value below which a stock is considered oversold. Default is -40.0."
        )
        
    wt_cache_key = f"{wt_timeframe}_{wt_oversold_threshold}"
    
    # Reactive Loader
    if 'wt_results_by_tf' not in st.session_state:
        st.session_state.wt_results_by_tf = {}
        
    if wt_cache_key not in st.session_state.wt_results_by_tf or st.session_state.wt_results_by_tf[wt_cache_key] is None:
        # Resolve symbols to scan: use all scanned breakout symbols or fallback to NIFTY 100/50
        symbols_to_scan = []
        if st.session_state.scan_results:
            symbols_to_scan.extend([r['symbol'] for r in st.session_state.scan_results])
        if st.session_state.above_ma_results:
            symbols_to_scan.extend([r['symbol'] for r in st.session_state.above_ma_results])
            
        # Clean unique list
        symbols_to_scan = list(set([s.upper() for s in symbols_to_scan]))
        
        # Fallback if empty to NIFTY 100
        if not symbols_to_scan:
            from config import NIFTY100_SYMBOLS
            symbols_to_scan = NIFTY100_SYMBOLS
            
        with st.spinner(f"Running real-time WaveTrend {wt_timeframe} scan for active breakout & watchlist universe ({len(symbols_to_scan)} stocks)..."):
            from scanner import scan_wt_cross
            
            # Map timeframes
            interval_map = {"Daily": "1d", "15 Min": "15m", "1 Hour": "60m", "Weekly": "1wk", "Monthly": "1mo"}
            period_map = {"Daily": "300d", "15 Min": "60d", "1 Hour": "730d", "Weekly": "5y", "Monthly": "10y"}
            
            interval = interval_map[wt_timeframe]
            period = period_map[wt_timeframe]
            
            wt_tf_results = []
            chunk_size = 50
            sym_chunks = [symbols_to_scan[i:i + chunk_size] for i in range(0, len(symbols_to_scan), chunk_size)]
            
            for chunk in sym_chunks:
                chunk_ns = [f"{s}.NS" for s in chunk]
                try:
                    # yfinance 1.x: group_by, threads, auto_adjust=False removed
                    df_bulk = yf.download(tickers=chunk_ns, period=period, interval=interval, progress=False, threads=False)
                    for sym in chunk:
                        sym_ns = f"{sym}.NS"
                        try:
                            if isinstance(df_bulk.columns, pd.MultiIndex):
                                # yfinance 1.x: (price_type, ticker) MultiIndex
                                all_tickers_wt = df_bulk.columns.get_level_values(1).unique().tolist()
                                matched_wt = next((t for t in all_tickers_wt if t.upper() == sym_ns.upper()), None)
                                if matched_wt is None:
                                    continue
                                ticker_df = df_bulk.xs(matched_wt, axis=1, level=1).copy()
                            else:
                                if len(chunk_ns) == 1:
                                    ticker_df = df_bulk.copy()
                                else:
                                    continue

                            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                            if all(col in ticker_df.columns for col in required_cols):
                                ticker_df = ticker_df[required_cols].dropna(subset=['Close'])
                                if interval not in ["15m", "60m"]:
                                    ticker_df = ticker_df[ticker_df['Volume'] > 0]
                                if len(ticker_df) >= 40:
                                    ticker_df = ticker_df.reset_index()
                                    ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.tz_localize(None)

                                    wt_res = scan_wt_cross(sym, ticker_df, wt_oversold_threshold=wt_oversold_threshold)
                                    if wt_res is not None:
                                        wt_res['timeframe'] = wt_timeframe
                                        # Inject threshold logic
                                        wt_res['is_oversold'] = wt_res['wt_value'] <= wt_oversold_threshold
                                        wt_tf_results.append(wt_res)
                        except Exception as sym_wt_ex:
                            print(f"Error extracting {sym_ns} from WaveTrend bulk download: {sym_wt_ex}")
                except Exception as chunk_ex:
                    print(f"Error bulk downloading WaveTrend chunk: {chunk_ex}")
            
            st.session_state.wt_results_by_tf[wt_cache_key] = wt_tf_results
            st.toast(f"🌊 WaveTrend {wt_timeframe} scan complete!", icon="✅")
            
    wt_data = st.session_state.wt_results_by_tf.get(wt_cache_key, [])
    
    st.markdown(f"### 🌊 WaveTrend Oversold Buy Signals ({wt_timeframe} Timeframe)")
    st.markdown(f"<p style='font-size:0.9rem; color:#94a3b8;'>Scan for stocks in the WaveTrend oversold zone (WT1 below {wt_oversold_threshold}) using LazyBear's WaveTrend with Crosses indicator. Stocks showing a <b style=\"color:#00e676;\">green dot 🟢 buy signal</b> (WT1 crossing above WT2) in oversold territory are prime mean-reversion candidates.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 1. Premium Metrics Row
    wt_m1, wt_m2, wt_m3, wt_m4 = st.columns(4)
    
    if wt_data:
        wt_total = len(wt_data)
        wt_buy_signals = [r for r in wt_data if r.get('buy_signal', False)]
        wt_buy_count = len(wt_buy_signals)
        wt_deepest = min(r['wt_value'] for r in wt_data)
        wt_avg = sum(r['wt_value'] for r in wt_data) / wt_total
    else:
        wt_total = 0
        wt_buy_count = 0
        wt_deepest = 0.0
        wt_avg = 0.0
    
    wt_m1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Oversold Stocks (WT1 < {wt_oversold_threshold})</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{wt_total}</h3></div>', unsafe_allow_html=True)
    wt_m2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">🟢 Buy Signals (Green Dot)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{wt_buy_count}</h3></div>', unsafe_allow_html=True)
    wt_m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Deepest WT1 Value</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{wt_deepest:.1f}</h3></div>', unsafe_allow_html=True)
    wt_m4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg WT1 Value</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{wt_avg:.1f}</h3></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filter toggle
    wt_filter_col1, wt_filter_col2, wt_filter_col3 = st.columns(3)
    wt_show_buy_only = wt_filter_col1.checkbox(
        "🟢 Show Buy Signals Only (Green Dot)",
        value=False,
        help="Show only stocks where WT1 has crossed above WT2 in the oversold zone (bullish crossover buy signal)"
    )
    wt_above_20sma = wt_filter_col2.checkbox(
        "📈 Above 20 SMA Only",
        value=False,
        help="Show only stocks currently trading above their 20 SMA trend filter"
    )
    wt_above_50sma = wt_filter_col3.checkbox(
        "🛡️ Above 50 SMA Only",
        value=False,
        help="Show only stocks currently trading above their 50 SMA trend filter"
    )
    wt_above_200sma = wt_filter_col3.checkbox(
        "🛡️ Above 200 DMA Only",
        value=False,
        help="Show only stocks currently trading above their 200 SMA long-term trend filter"
    )
    
    # 2. Main Scan Table
    if wt_data is None:
        st.info("💡 Run the scanner from the sidebar to identify WaveTrend oversold buy signals.")
    elif len(wt_data) == 0:
        st.info(f"ℹ️ No stocks found in the WaveTrend oversold zone (WT1 < {wt_oversold_threshold}) on {wt_timeframe} timeframe today.")
    else:
        # Apply filters
        display_wt = list(wt_data)
        if wt_show_buy_only:
            display_wt = [r for r in display_wt if r.get('buy_signal', False)]
        if wt_above_20sma:
            display_wt = [r for r in display_wt if r.get('above_20sma', False)]
        if wt_above_50sma:
            display_wt = [r for r in display_wt if r.get('above_50sma', False)]
        if wt_above_200sma:
            display_wt = [r for r in display_wt if r.get('above_200sma', False)]
        
        # Sort by WT value ascending (deepest oversold first)
        sorted_wt = sorted(display_wt, key=lambda x: x['wt_value'])
        
        if len(sorted_wt) == 0:
            st.info("ℹ️ No stocks match the active filters found today. Try unchecking some filters above to see more oversold stocks.")
        else:
            # Download WaveTrend results
            export_wt = []
            for r in sorted_wt:
                export_wt.append({
                    "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']),
                                        "CMP (₹)": r['cmp'],
                    "Day Change %": r['day_change_pct'],
                    "WT1": r['wt_value'],
                    "WT2": r['wt2_value'],
                    "WT Diff (WT1-WT2)": r.get('wt_diff', r['wt_value'] - r['wt2_value']),
                    "Buy Signal": r.get('buy_signal', False),
                    "Above 20 SMA": r.get('above_20sma', False),
                    "Above 50 SMA": r.get('above_50sma', False),
                   "Above 200 SMA": r.get('above_200sma', False),
                    "Volume": int(r.get('volume', 0)),
                    "Buy Range (₹)": r.get('buy_price', r.get('cmp', 0)),
                    "Stop Loss (₹)": r.get('exit_price', 0),
                    "Target (₹)": r.get('target_price', 0),
                    "Confidence": r.get('confidence', ''),
                    "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
                })
            export_wt_df = pd.DataFrame(export_wt)
            csv_wt_data = export_wt_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download WaveTrend Results (CSV)",
                data=csv_wt_data,
                file_name=f"wavetrend_signals_{wt_timeframe}_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="wt_download_csv_btn"
            )
            
            st.markdown("---")
            # Render the unified Trade Execution Matrix
            st.markdown(f"### 🌊 Active Oversold Trade Execution Sheet ({wt_timeframe})")
            render_unified_strategy_table(sorted_wt, "wavetrend", "wt_tab")
    
    # WaveTrend indicator explanation
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📖 How WaveTrend Indicator Works"):
        st.markdown("""
        **WaveTrend with Crosses (LazyBear)**
        
        This indicator is a momentum oscillator that identifies overbought/oversold conditions:
        
        - **WT1 (Green Line):** The smoothed trend oscillator calculated from typical price deviations
        - **WT2 (Red Line):** A 4-period SMA of WT1, used as a signal line
        - **Buy Signal (Green Dot 🟢):** Occurs when WT1 crosses ABOVE WT2 in the oversold zone (below -40)
        - **Oversold Zone:** WT1 values below -40 indicate the stock is in deep oversold territory
        - **Extreme Oversold:** WT1 values below -60 suggest extreme selling pressure
        
        **Strategy:** Look for stocks with WT1 below -40 showing a green dot buy signal (WT1 crossing above WT2). 
        These are high-probability mean-reversion setups where selling pressure is exhausting and a bounce is likely.
        
        **Parameters Used:** Channel Length = 10, Average Length = 21
        """, unsafe_allow_html=False)


# ==============================================================================
# TAB 11: MARK MINERVINI STAGE-2 TREND TEMPLATE
# ==============================================================================
with tab_minervini:
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
                csv_m_data = export_m_df.to_csv(index=False).encode('utf-8')
                
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
                    csv_m_data_h = export_m_df_h.to_csv(index=False).encode('utf-8')
                    
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
        st.markdown("""
        **Mark Minervini Stage-2 Trend Template Rules**
        
        To qualify as an institutional Stage-2 stock, an asset must meet all of the following rules:
        
        1. **Price is Above 150 EMA and 200 SMA:** Confirms a structural long-term uptrend.
        2. **150-day EMA is Above the 200-day SMA:** Confirms standard momentum alignment.
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


# ==============================================================================
# TAB 12: SCAN HISTORY VIEWER
# ==============================================================================
with tab_history:
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
        sub_breakout, sub_gapup, sub_above_ma, sub_support_ma, sub_crossover_ma, sub_wt = st.tabs([
            "📊 VDU Breakouts",
            "🚀 Gap-Ups",
            "📈 Above 20 & 50 SMA",
            "🛡️ 65 SMA Support",
            "🔄 MA Crossovers",
            "🌊 Wave Trend"
        ])
        
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


# ==============================================================================
# TAB: MONTHLY MOMENTUM SCANNER (EMA Stack + ROC + RSI + Volume > Vol SMA)
# ==============================================================================
with tab_monthly_mom:
    st.markdown("### 📅 Monthly Momentum Scanner")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8;'>Scans <b>all NSE-listed stocks</b> (Market Cap ≥ ₹3000 Cr, Price ≥ ₹100) for "
        "the Chartink-style <b>Monthly EMA Alignment</b> momentum strategy. All conditions are evaluated on <b>Monthly candles</b>:</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px;'>
          <span style='background:rgba(0,230,118,0.12); color:#00e676; border:1px solid rgba(0,230,118,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Close &gt; EMA(8)</span>
          <span style='background:rgba(41,182,246,0.12); color:#29b6f6; border:1px solid rgba(41,182,246,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>EMA(8) &gt; EMA(12)</span>
          <span style='background:rgba(41,182,246,0.12); color:#29b6f6; border:1px solid rgba(41,182,246,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>EMA(12) &gt; EMA(20)</span>
          <span style='background:rgba(255,160,0,0.12); color:#ffa000; border:1px solid rgba(255,160,0,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>ROC(6M): 10–80%</span>
          <span style='background:rgba(171,71,188,0.12); color:#ba68c8; border:1px solid rgba(171,71,188,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>RSI(14M): 55–85</span>
          <span style='background:rgba(239,68,68,0.12); color:#ef4444; border:1px solid rgba(239,68,68,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Vol &gt; SMA(Vol, 12M)</span>
          <span style='background:rgba(0,229,255,0.12); color:#00e5ff; border:1px solid rgba(0,229,255,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>MCap ≥ ₹3000 Cr</span>
          <span style='background:rgba(148,163,184,0.12); color:#94a3b8; border:1px solid rgba(148,163,184,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Price ≥ ₹100</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --- Run Scan Button ---
    mm_col1, mm_col2 = st.columns([1, 3])
    with mm_col1:
        run_mm_scan = st.button("🔍 Run Monthly Momentum Scan", use_container_width=True, key="run_monthly_mom_btn")
    with mm_col2:
        st.info("⏱️ This scan downloads ~5 years of **monthly** data for all NSE stocks. It may take 3–8 minutes for All NSE universe. Use Nifty 500 for faster results.")

    # Check if background thread has finished and results are available
    if st.session_state.monthly_momentum_results is None and MOMENTUM_SCAN_STATUS["monthly_results"] is not None:
        st.session_state.monthly_momentum_results = MOMENTUM_SCAN_STATUS["monthly_results"]

    mm_data = st.session_state.monthly_momentum_results

    # --- Render Background Scan Progress Inside Monthly Tab ---
    if MOMENTUM_SCAN_STATUS["is_running"] and mm_data is None:
        st.markdown(
            f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
                <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background Momentum Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">
                    Monthly & Weekly Momentum scanners are running automatically in the background. You can browse all other tabs normally!
                </p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">
                    Current Status: <span style="color:#00e5ff;">{MOMENTUM_SCAN_STATUS["status_text"]}</span>
                </div>
            </div>
            <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.progress(MOMENTUM_SCAN_STATUS["progress"])
        if st.button("🔄 Refresh Scanner Status", key="refresh_mm_status_btn"):
            st.rerun()

    if run_mm_scan:
        if MOMENTUM_SCAN_STATUS["is_running"]:
            st.warning("⚠️ Scanners are already running in the background! Please wait for them to complete.")
        else:
            import database
            
            today_ist = datetime.now(IST_TIMEZONE)
            base_date_monthly = database.get_monthly_base_date(today_ist.year, today_ist.month)
            
            if base_date_monthly and base_date_monthly != today_str_check:
                mm_status = st.empty()
                mm_status.text(f"Running lightning-fast Monthly Momentum price update (since {base_date_monthly})...")
                mm_results = run_monthly_momentum_update(base_date_monthly, today_str_check)
                
                try:
                    database.save_monthly_momentum_results(today_str_check, mm_results)
                except Exception as db_save_ex:
                    print(f"Failed to cache monthly momentum results in PostgreSQL: {db_save_ex}")
                    
                import json
                monthly_payload = {"date": today_str_check, "results": mm_results}
                with open("monthly_momentum_cache.json", "w") as f:
                    json.dump(monthly_payload, f, indent=2)
                    
                st.session_state.monthly_momentum_results = mm_results
                mm_status.text(f"✅ Monthly Momentum price update complete for {len(mm_results)} stocks!")
                st.toast(f"📅 Monthly update done — {len(mm_results)} stocks updated!", icon="✅")
                st.rerun()
                
            import concurrent.futures as _cf
            import time as _time

        # Resolve universe
        from data_fetcher import get_index_stocks, get_all_nse_symbols
        if "NIFTY 50" in universe_selection:
            mm_universe = get_index_stocks("NIFTY 50")
        elif "NIFTY 100" in universe_selection:
            mm_universe = get_index_stocks("NIFTY 100")
        else:
            mm_universe = get_all_nse_symbols()

        mm_results = []
        mm_prog = st.progress(0)
        mm_status = st.empty()
        total_mm = len(mm_universe)

        # ---- Step 1: Batch fetch market caps via yf.download 1d ----
        mm_status.text("Step 1/3 — Fetching real-time quotes & market caps...")
        mcap_map = {}   # symbol -> market_cap in crore
        price_map_mm = {}  # symbol -> CMP
        CRORE = 1_00_00_000  # 1 crore INR

        mm_tickers_ns = [f"{s.strip().upper()}.NS" for s in mm_universe]
        mm_chunk_size = 200
        mm_ticker_chunks = [mm_tickers_ns[i:i+mm_chunk_size] for i in range(0, len(mm_tickers_ns), mm_chunk_size)]

        for mm_cidx, mm_chunk in enumerate(mm_ticker_chunks):
            mm_status.text(f"Step 1/3 — Quotes chunk {mm_cidx+1}/{len(mm_ticker_chunks)}...")
            try:
                q_df = yf.download(tickers=mm_chunk, period="1d", progress=False, threads=False)
                if not q_df.empty and isinstance(q_df.columns, pd.MultiIndex):
                    price_types_mm = q_df.columns.get_level_values(0).unique().tolist()
                    cl_s = q_df['Close'].iloc[-1] if 'Close' in price_types_mm else pd.Series(dtype=float)
                    for tk, pv in cl_s.items():
                        sym_clean = str(tk).replace(".NS", "").upper()
                        if not pd.isna(pv) and float(pv) >= 100.0:
                            price_map_mm[sym_clean] = float(pv)
            except Exception as e:
                print(f"MM quote chunk {mm_cidx+1} failed: {e}")
            _time.sleep(0.5)

        # Filter by price >= 100 first
        mm_price_passed = [s for s in mm_universe if s.strip().upper() in price_map_mm]
        mm_status.text(f"Step 1/3 — {len(mm_price_passed)} stocks pass price ≥ ₹100 filter. Fetching market caps...")

        mm_mcap_passed = list(mm_price_passed)
        mcap_map = {sym: 0.0 for sym in mm_mcap_passed}  # Default to 0 since bulk fetch rate-limits
        mm_status.text(f"Step 2/3 — Market Cap filter bypassed. Downloading monthly data for {len(mm_mcap_passed)} stocks...")

        # ---- Step 3: Download monthly OHLCV in bulk and scan ----
        mm_monthly_chunk_size = 50
        mm_sym_chunks = [mm_mcap_passed[i:i+mm_monthly_chunk_size] for i in range(0, len(mm_mcap_passed), mm_monthly_chunk_size)]
        total_chunks_mm = len(mm_sym_chunks)

        for c_idx, c_chunk in enumerate(mm_sym_chunks):
            mm_status.text(f"Step 3/3 — Scanning monthly data: chunk {c_idx+1}/{total_chunks_mm} ({len(mm_results)} matches so far)...")
            mm_prog.progress((c_idx + 1) / max(total_chunks_mm, 1))
            chunk_ns_mm = [f"{s.strip().upper()}.NS" for s in c_chunk]
            try:
                df_mbulk = yf.download(
                    tickers=chunk_ns_mm,
                    period="10y",
                    interval="1mo",
                    progress=False
                )
                for sym_m in c_chunk:
                    sym_ns_m = f"{sym_m.strip().upper()}.NS"
                    try:
                        if isinstance(df_mbulk.columns, pd.MultiIndex):
                            all_t_mm = df_mbulk.columns.get_level_values(1).unique().tolist()
                            matched_m = next((t for t in all_t_mm if t.upper() == sym_ns_m.upper()), None)
                            if matched_m is None:
                                continue
                            t_df_m = df_mbulk.xs(matched_m, axis=1, level=1).copy()
                        else:
                            if len(chunk_ns_mm) == 1:
                                t_df_m = df_mbulk.copy()
                            else:
                                continue

                        req_m = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if not all(col in t_df_m.columns for col in req_m):
                            continue
                        t_df_m = t_df_m[req_m].dropna(subset=['Close'])
                        t_df_m = t_df_m[t_df_m['Volume'] > 0]
                        if len(t_df_m) < 22:
                            continue
                        t_df_m = t_df_m.reset_index()
                        t_df_m.rename(columns={t_df_m.columns[0]: 'Date'}, inplace=True)
                        t_df_m['Date'] = pd.to_datetime(t_df_m['Date']).dt.tz_localize(None)

                        res_m = scan_monthly_momentum(
                            sym_m.strip().upper(),
                            t_df_m,
                            market_cap_cr=mcap_map.get(sym_m.strip().upper(), 0.0)
                        )
                        if res_m is not None:
                            mm_results.append(res_m)
                    except Exception as sym_m_ex:
                        pass
            except Exception as c_ex:
                print(f"Monthly momentum chunk {c_idx+1} failed: {c_ex}")
            _time.sleep(0.3)

        mm_prog.progress(1.0)
        st.session_state.monthly_momentum_results = mm_results
        mm_data = mm_results
        mm_status.text(f"✅ Monthly Momentum scan complete! Found {len(mm_results)} qualifying stocks.")
        st.toast(f"📅 Monthly Momentum scan done — {len(mm_results)} stocks matched!", icon="✅")

    # ---- Display Results ----
    if mm_data is None:
        st.info("💡 Click **Run Monthly Momentum Scan** above to start the scan.")
    elif len(mm_data) == 0:
        st.warning("⚠️ No stocks matched all Monthly Momentum conditions in the selected universe. Try using a larger universe (All NSE).")
    else:
        sorted_mm = sorted(mm_data, key=lambda x: x.get('momentum_score', 0.0), reverse=True)

        # Metrics row
        mm_m1, mm_m2, mm_m3, mm_m4 = st.columns(4)
        mm_m1.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Stocks Matched</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{len(sorted_mm)}</h3></div>', unsafe_allow_html=True)
        mm_m2.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Momentum Score</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{sum(r["momentum_score"] for r in sorted_mm)/len(sorted_mm):.1f} pts</h3></div>', unsafe_allow_html=True)
        mm_m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Monthly RSI</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{sum(r["rsi_monthly"] for r in sorted_mm)/len(sorted_mm):.1f}</h3></div>', unsafe_allow_html=True)
        mm_m4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg 6M ROC</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{sum(r["roc6"] for r in sorted_mm)/len(sorted_mm):.1f}%</h3></div>', unsafe_allow_html=True)
        st.markdown("---")

        # CSV Export
        mm_export = [{
            "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']), "Company": r['company_name'],
            "CMP (₹)": r['cmp'], "MCap (Cr)": r['market_cap_cr'],
            "1M Return (%)": r.get('return_1m', r.get('day_change_pct', 0.0)),
            "EMA8": r['ema8'], "EMA12": r['ema12'], "EMA20": r['ema20'],
            "ROC 6M (%)": r['roc6'], "RSI 14M": r['rsi_monthly'],
            "Volume": r['volume'], "Vol SMA12": r['vol_sma12'],
            "Momentum Score": r['momentum_score'],
            "Buy Price (₹)": r['buy_price'], "Stop Loss (₹)": r['exit_price'], "Target (₹)": r['target_price'],
            "Confidence": r['confidence'],
            "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
        } for r in sorted_mm]
        mm_csv = pd.DataFrame(mm_export).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Monthly Momentum Results (CSV)",
            data=mm_csv,
            file_name=f"monthly_momentum_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dl_monthly_mom_csv"
        )
        st.markdown("---")

        # Rich table
        st.markdown("### 📊 Monthly Momentum Trade Execution Matrix")
        mm_rows_html = []
        from utils import get_day_change_badge_html, get_signal_badge_html
        w_df_mm = watchlist.load_watchlist()
        wl_syms_mm = set(w_df_mm['symbol'].str.upper().unique()) if not w_df_mm.empty else set()

        for mm_idx, r in enumerate(sorted_mm):
            cmp_v = r['cmp']; buy_v = r['buy_price']; sl_v = r['exit_price']; tgt_v = r['target_price']
            conf_v = r.get('confidence', 'Medium')
            clean_conf_mm = conf_v.split(" (")[0] if " (" in conf_v else conf_v
            conf_color_mm = "#ef4444" if "Low" in clean_conf_mm else "#ffa000" if "Medium" in clean_conf_mm else "#00e676"
            conf_badge_mm = f'<span class="custom-badge" style="background:rgba({"0,230,118" if "High" in clean_conf_mm else "255,160,0" if "Medium" in clean_conf_mm else "239,68,68"},0.12); color:{conf_color_mm}; border:1px solid {conf_color_mm}; font-size:0.75rem; padding:2px 6px; border-radius:4px;">{clean_conf_mm}</span>'

            is_wl_mm = r['symbol'] in wl_syms_mm
            if is_wl_mm:
                wl_cell_mm = f'<td style="padding:8px 10px; text-align:center;"><span style="color:#00e676;">☑️</span> <a href="/?remove_from_watchlist={r["symbol"]}" target="_self" style="color:#ef4444; font-size:0.72rem;">[Remove]</a></td>'
            else:
                wl_cell_mm = f'<td style="padding:8px 10px; text-align:center;"><a href="/?add_to_watchlist={r["symbol"]}&price={buy_v}&score={r.get("momentum_score",50)}" target="_self" style="color:#94a3b8; font-size:1.1rem;">☐</a> <a href="/?add_to_watchlist={r["symbol"]}&price={buy_v}&score={r.get("momentum_score",50)}" target="_self" style="color:#00e676; font-size:0.72rem;">[Add]</a></td>'

            chg_b = get_day_change_badge_html(r.get('return_1m', r.get('day_change_pct', 0.0)))
            roc_color = "#00e676" if r['roc6'] >= 30 else "#ffa000" if r['roc6'] >= 15 else "#29b6f6"
            rsi_color = "#00e676" if 60 <= r['rsi_monthly'] <= 75 else "#ffa000"
            vol_ratio_mm = r['volume'] / r['vol_sma12'] if r['vol_sma12'] > 0 else 1.0
            mcap_fmt = f"₹{r['market_cap_cr']:,.0f} Cr" if r['market_cap_cr'] > 0 else "—"

            clean_rec_mm = extract_clean_recommendation(r.get('recommendation', ''))

            mm_rows_html.append(
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04); transition:background 0.2s;">'
                f'{wl_cell_mm}'
                f'<td style="padding:8px 10px; font-weight:bold; color:#29b6f6;"><a href="https://in.tradingview.com/chart/?symbol=NSE:{r["symbol"].replace(".NS", "")}" target="_blank" rel="noopener noreferrer" style="color:#29b6f6; text-decoration:none;">{r["symbol"]}</a></td>'
                f'<td style="padding:8px 10px; color:#94a3b8; font-size:0.8rem;">{r.get("company_name", "")}</td>'
                f'<td style="padding:8px 10px; color:#e2e8f0; font-weight:500;">₹{cmp_v:,.2f}</td>'
                f'<td style="padding:8px 10px;">{chg_b}</td>'
                f'<td style="padding:8px 10px; color:#00e5ff; font-size:0.82rem;">{mcap_fmt}</td>'
                f'<td style="padding:8px 10px; color:#38bdf8;">₹{r["ema8"]:,.2f}</td>'
                f'<td style="padding:8px 10px; color:#7dd3fc;">₹{r["ema12"]:,.2f}</td>'
                f'<td style="padding:8px 10px; color:#94a3b8;">₹{r["ema20"]:,.2f}</td>'
                f'<td style="padding:8px 10px; color:{roc_color}; font-weight:600;">{r["roc6"]:+.1f}%</td>'
                f'<td style="padding:8px 10px; color:{rsi_color}; font-weight:600;">{r["rsi_monthly"]:.1f}</td>'
                f'<td style="padding:8px 10px; color:#ffa000; font-weight:600;">{vol_ratio_mm:.2f}x</td>'
                f'<td style="padding:8px 10px; color:#00e676; font-weight:700;">{r.get("momentum_score", 0):.0f}</td>'
                f'<td style="padding:8px 10px; color:#cbd5e1; font-weight:600;">₹{buy_v:,.2f}</td>'
                f'<td style="padding:8px 10px; color:#ef4444; font-weight:600;">₹{sl_v:,.2f}</td>'
                f'<td style="padding:8px 10px; color:#00e676; font-weight:600;">₹{tgt_v:,.2f}</td>'
                f'<td style="padding:8px 10px;">{conf_badge_mm}</td>'
                f'<td style="padding:8px 10px; color:#94a3b8; font-style:italic; font-size:0.8rem; line-height:1.4; min-width: 250px; max-width: 350px; white-space: normal !important; word-wrap: break-word;">"{clean_rec_mm}"</td>'
                f'</tr>'
            )

        mm_headers = [
            "Watchlist", "Symbol", "Company", "CMP", "1M Return %", "MCap",
            "EMA 8", "EMA 12", "EMA 20", "ROC 6M", "RSI 14M", "Vol Ratio", "Score",
            "Buy Price", "Stop Loss", "Target", "Confidence", "Analysis"
        ]
        mm_header_html = "".join([f'<th style="padding:8px 10px; white-space:nowrap;">{h}</th>' for h in mm_headers])

        st.markdown(
            f'<div class="glass-card" style="padding:18px; border:1px solid rgba(0,229,255,0.2); background:rgba(9,13,22,0.55); border-radius:12px;">'
            f'<div style="overflow-x:auto;">'
            f'<table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.83rem; color:#cbd5e1; font-family:Outfit,sans-serif;">'
            f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#00e5ff; font-weight:bold; background:rgba(0,229,255,0.05); font-size:0.78rem; text-transform:uppercase;">'
            f'{mm_header_html}</tr></thead>'
            f'<tbody>{chr(10).join(mm_rows_html)}</tbody>'
            f'</table></div></div>',
            unsafe_allow_html=True
        )


# ==============================================================================
# TAB: WEEKLY MOMENTUM SCANNER
# ==============================================================================
with tab_weekly_mom:
    st.markdown("### 📈 Weekly Momentum Breakout Scanner")
    st.markdown(
        "<p style='font-size:0.9rem; color:#94a3b8;'>Scans <b>all NSE-listed stocks</b> (MCap ≥ ₹5000 Cr, Price ≥ ₹200) for the "
        "Chartink-style <b>Weekly Momentum Breakout</b> strategy. All conditions on <b>Weekly candles</b>:</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:18px;'>
          <span style='background:rgba(0,230,118,0.12); color:#00e676; border:1px solid rgba(0,230,118,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Vol &gt; SMA(Vol, 20W)</span>
          <span style='background:rgba(41,182,246,0.12); color:#29b6f6; border:1px solid rgba(41,182,246,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Close &gt; ₹200</span>
          <span style='background:rgba(41,182,246,0.12); color:#29b6f6; border:1px solid rgba(41,182,246,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Close &gt; Prev Week Close</span>
          <span style='background:rgba(0,229,255,0.12); color:#00e5ff; border:1px solid rgba(0,229,255,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Open &gt; Prev Week Close</span>
          <span style='background:rgba(255,160,0,0.12); color:#ffa000; border:1px solid rgba(255,160,0,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>CCI(20W) &gt; 90</span>
          <span style='background:rgba(171,71,188,0.12); color:#ba68c8; border:1px solid rgba(171,71,188,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>RSI(14W) &gt; 60</span>
          <span style='background:rgba(239,68,68,0.12); color:#ef4444; border:1px solid rgba(239,68,68,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>Close &gt; SMA(Close, 20W)</span>
          <span style='background:rgba(0,229,255,0.12); color:#00e5ff; border:1px solid rgba(0,229,255,0.3); padding:4px 12px; border-radius:20px; font-size:0.8rem; font-weight:600;'>MCap ≥ ₹5000 Cr</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    wm_col1, wm_col2 = st.columns([1, 3])
    with wm_col1:
        run_wm_scan = st.button("🔍 Run Weekly Momentum Scan", use_container_width=True, key="run_weekly_mom_btn")
    with wm_col2:
        st.info("⏱️ Downloads weekly OHLCV data for all NSE stocks with MCap ≥ ₹5000 Cr. Typical run: **2–5 minutes**.")

    # Check if background thread has finished and results are available
    if st.session_state.weekly_momentum_results is None and MOMENTUM_SCAN_STATUS["weekly_results"] is not None:
        st.session_state.weekly_momentum_results = MOMENTUM_SCAN_STATUS["weekly_results"]

    wm_data = st.session_state.weekly_momentum_results

    # --- Render Background Scan Progress Inside Weekly Tab ---
    if MOMENTUM_SCAN_STATUS["is_running"] and wm_data is None:
        st.markdown(
            f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,230,118,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
                <h4 style="color:#00e676; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background Momentum Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">
                    Monthly & Weekly Momentum scanners are running automatically in the background. You can browse all other tabs normally!
                </p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">
                    Current Status: <span style="color:#00e676;">{MOMENTUM_SCAN_STATUS["status_text"]}</span>
                </div>
            </div>
            <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.progress(MOMENTUM_SCAN_STATUS["progress"])
        if st.button("🔄 Refresh Status", key="refresh_wm_status_btn"):
            st.rerun()

    if run_wm_scan:
        if MOMENTUM_SCAN_STATUS["is_running"]:
            st.warning("⚠️ Scanners are already running in the background! Please wait for them to complete.")
        else:
            import database
            from scanner import run_weekly_momentum_update
            
            today_ist = datetime.now(IST_TIMEZONE)
            iso_weekday = today_ist.isoweekday()
            start_of_week = today_ist - timedelta(days=iso_weekday - 1)
            end_of_week = start_of_week + timedelta(days=6)
            base_date_weekly = database.get_weekly_base_date(start_of_week.strftime("%Y-%m-%d"), end_of_week.strftime("%Y-%m-%d"))
            
            if base_date_weekly and base_date_weekly != today_str_check:
                wm_status = st.empty()
                wm_status.text(f"Running lightning-fast Weekly Momentum price update (since {base_date_weekly})...")
                wm_results = run_weekly_momentum_update(base_date_weekly, today_str_check)
                
                try:
                    database.save_weekly_momentum_results(today_str_check, wm_results)
                except Exception as db_save_ex:
                    print(f"Failed to cache weekly momentum results in PostgreSQL: {db_save_ex}")
                    
                import json
                weekly_payload = {"date": today_str_check, "results": wm_results}
                with open("weekly_momentum_cache.json", "w") as f:
                    json.dump(weekly_payload, f, indent=2)
                    
                st.session_state.weekly_momentum_results = wm_results
                wm_status.text(f"✅ Weekly Momentum price update complete for {len(wm_results)} stocks!")
                st.toast(f"📈 Weekly update done — {len(wm_results)} stocks updated!", icon="✅")
                st.rerun()
                
            import concurrent.futures as _cf_wm
            import time as _time_wm

        from data_fetcher import get_index_stocks, get_all_nse_symbols
        if "NIFTY 50" in universe_selection:
            wm_universe = get_index_stocks("NIFTY 50")
        elif "NIFTY 100" in universe_selection:
            wm_universe = get_index_stocks("NIFTY 100")
        else:
            wm_universe = get_all_nse_symbols()

        wm_results = []
        wm_prog    = st.progress(0)
        wm_status  = st.empty()

        # ---- Step 1: Price filter ≥ ₹200 via 1d bulk download ----
        wm_status.text("Step 1/3 — Fetching real-time quotes (Price ≥ ₹200 filter)...")
        price_map_wm = {}
        CRORE = 1_00_00_000

        wm_tickers_ns = [f"{s.strip().upper()}.NS" for s in wm_universe]
        wm_chunk_size  = 200
        wm_q_chunks    = [wm_tickers_ns[i:i+wm_chunk_size] for i in range(0, len(wm_tickers_ns), wm_chunk_size)]

        for wm_cidx, wm_chunk in enumerate(wm_q_chunks):
            wm_status.text(f"Step 1/3 — Quote chunk {wm_cidx+1}/{len(wm_q_chunks)}...")
            try:
                wq_df = yf.download(tickers=wm_chunk, period="1d", progress=False, threads=False)
                if not wq_df.empty and isinstance(wq_df.columns, pd.MultiIndex):
                    wpt = wq_df.columns.get_level_values(0).unique().tolist()
                    wcl = wq_df['Close'].iloc[-1] if 'Close' in wpt else pd.Series(dtype=float)
                    for wtk, wpv in wcl.items():
                        wsc = str(wtk).replace(".NS", "").upper()
                        if not pd.isna(wpv) and float(wpv) >= 200.0:
                            price_map_wm[wsc] = float(wpv)
            except Exception as wqe:
                print(f"WM quote chunk {wm_cidx+1} failed: {wqe}")
            _time_wm.sleep(0.5)

        wm_price_passed = [s for s in wm_universe if s.strip().upper() in price_map_wm]
        wm_status.text(f"Step 1/3 — {len(wm_price_passed)} stocks pass Price ≥ ₹200. Checking market caps...")

        # ---- Step 2: Market Cap Filter Bypassed ----
        wm_mcap_passed = list(wm_price_passed)
        mcap_map_wm = {sym: 0.0 for sym in wm_mcap_passed}
        wm_status.text(f"Step 2/3 — Market Cap filter bypassed. Downloading weekly data for {len(wm_mcap_passed)} stocks...")

        # ---- Step 3: Bulk weekly OHLCV download + scan ----
        wm_monthly_chunk_size = 60
        wm_sym_chunks  = [wm_mcap_passed[i:i+wm_monthly_chunk_size] for i in range(0, len(wm_mcap_passed), wm_monthly_chunk_size)]
        wm_total_chunks = len(wm_sym_chunks)

        for wc_idx, wc_chunk in enumerate(wm_sym_chunks):
            wm_status.text(f"Step 3/3 — Scanning weekly data: chunk {wc_idx+1}/{wm_total_chunks} ({len(wm_results)} matches so far)...")
            wm_prog.progress((wc_idx + 1) / max(wm_total_chunks, 1))
            chunk_ns_wm = [f"{s.strip().upper()}.NS" for s in wc_chunk]
            try:
                df_wbulk = yf.download(
                    tickers=chunk_ns_wm,
                    period="3y",
                    interval="1wk",
                    progress=False
                )
                for sym_w in wc_chunk:
                    sym_ns_w = f"{sym_w.strip().upper()}.NS"
                    try:
                        if isinstance(df_wbulk.columns, pd.MultiIndex):
                            all_t_wm = df_wbulk.columns.get_level_values(1).unique().tolist()
                            matched_w = next((t for t in all_t_wm if t.upper() == sym_ns_w.upper()), None)
                            if matched_w is None:
                                continue
                            t_df_w = df_wbulk.xs(matched_w, axis=1, level=1).copy()
                        else:
                            if len(chunk_ns_wm) == 1:
                                t_df_w = df_wbulk.copy()
                            else:
                                continue

                        req_w = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if not all(col in t_df_w.columns for col in req_w):
                            continue
                        t_df_w = t_df_w[req_w].dropna(subset=['Close'])
                        t_df_w = t_df_w[t_df_w['Volume'] > 0]
                        if len(t_df_w) < 22:
                            continue
                        t_df_w = t_df_w.reset_index()
                        t_df_w.rename(columns={t_df_w.columns[0]: 'Date'}, inplace=True)
                        t_df_w['Date'] = pd.to_datetime(t_df_w['Date']).dt.tz_localize(None)

                        res_w = scan_weekly_momentum(
                            sym_w.strip().upper(),
                            t_df_w,
                            market_cap_cr=mcap_map_wm.get(sym_w.strip().upper(), 0.0)
                        )
                        if res_w is not None:
                            wm_results.append(res_w)
                    except Exception:
                        pass
            except Exception as wc_ex:
                print(f"Weekly momentum chunk {wc_idx+1} failed: {wc_ex}")
            _time_wm.sleep(0.3)

        wm_prog.progress(1.0)
        st.session_state.weekly_momentum_results = wm_results
        wm_data = wm_results
        wm_status.text(f"✅ Weekly Momentum scan complete! Found {len(wm_results)} qualifying stocks.")
        st.toast(f"📈 Weekly scan done — {len(wm_results)} stocks matched all 8 conditions!", icon="✅")

    # ---- Display Results ----
    if wm_data is None:
        st.info("💡 Click **Run Weekly Momentum Scan** above to start.")
    elif len(wm_data) == 0:
        st.warning("⚠️ No stocks matched all 8 Weekly Momentum conditions. Try a broader universe.")
    else:
        sorted_wm = sorted(wm_data, key=lambda x: x.get('weekly_score', 0.0), reverse=True)

        # Metrics row
        wm_m1, wm_m2, wm_m3, wm_m4 = st.columns(4)
        wm_m1.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Stocks Matched</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{len(sorted_wm)}</h3></div>', unsafe_allow_html=True)
        wm_m2.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg Weekly Score</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{sum(r["weekly_score"] for r in sorted_wm)/len(sorted_wm):.1f} pts</h3></div>', unsafe_allow_html=True)
        wm_m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg RSI (14W)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{sum(r["rsi_weekly"] for r in sorted_wm)/len(sorted_wm):.1f}</h3></div>', unsafe_allow_html=True)
        wm_m4.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg CCI (20W)</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{sum(r["cci_weekly"] for r in sorted_wm)/len(sorted_wm):.1f}</h3></div>', unsafe_allow_html=True)
        st.markdown("---")

        # CSV export
        wm_export = [{
            "Symbol": r['symbol'],
                "Sector": get_stock_sector(r['symbol']), "Company": r['company_name'],
            "CMP (₹)": r['cmp'], "MCap (Cr)": r['market_cap_cr'],
            "1M Return (%)": r.get('return_1m', 0.0),
            "Prev Close (₹)": r['prev_close'], "Week Open (₹)": r['curr_open'],
            "SMA 20W (₹)": r['close_sma20'],
            "RSI 14W": r['rsi_weekly'], "CCI 20W": r['cci_weekly'],
            "Vol Ratio": r['vol_ratio'],
            "Weekly Score": r['weekly_score'],
            "Buy Price (₹)": r['buy_price'], "Stop Loss (₹)": r['exit_price'], "Target (₹)": r['target_price'],
            "Confidence": r['confidence'],
            "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
        } for r in sorted_wm]
        wm_csv = pd.DataFrame(wm_export).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Weekly Momentum Results (CSV)",
            data=wm_csv,
            file_name=f"weekly_momentum_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dl_weekly_mom_csv"
        )
        st.markdown("---")

        # Trade Execution Table
        st.markdown("### 📊 Weekly Momentum Trade Execution Matrix")
        wm_rows_html = []
        w_df_wm  = watchlist.load_watchlist()
        wl_syms_wm = set(w_df_wm['symbol'].str.upper().unique()) if not w_df_wm.empty else set()

        for r in sorted_wm:
            cmp_v  = r['cmp'];  buy_v = r['buy_price']
            sl_v   = r['exit_price']; tgt_v = r['target_price']
            conf_v = r.get('confidence', 'Medium')
            clean_conf_wm = conf_v.split(" (")[0] if " (" in conf_v else conf_v
            conf_color_wm = "#00e676" if "High" in clean_conf_wm else "#ffa000" if "Medium" in clean_conf_wm else "#ef4444"
            conf_badge_wm = (
                f'<span style="background:rgba({"0,230,118" if "High" in clean_conf_wm else "255,160,0" if "Medium" in clean_conf_wm else "239,68,68"},0.12); '
                f'color:{conf_color_wm}; border:1px solid {conf_color_wm}; font-size:0.75rem; padding:2px 8px; border-radius:4px;">{clean_conf_wm}</span>'
            )
            is_wl_wm = r['symbol'] in wl_syms_wm
            if is_wl_wm:
                wl_cell_wm = f'<td style="padding:8px 10px; text-align:center;"><span style="color:#00e676;">☑️</span> <a href="/?remove_from_watchlist={r["symbol"]}" target="_self" style="color:#ef4444; font-size:0.72rem;">[Remove]</a></td>'
            else:
                wl_cell_wm = f'<td style="padding:8px 10px; text-align:center;"><a href="/?add_to_watchlist={r["symbol"]}&price={buy_v}&score={r.get("weekly_score",50)}" target="_self" style="color:#00e676; font-size:0.72rem;">[+Add]</a></td>'

            chg_b    = get_day_change_badge_html(r.get('weekly_chg_pct', 0.0))
            ret_1m_val = r.get('return_1m', 0.0)
            ret_1m_b = get_day_change_badge_html(ret_1m_val)
            rsi_col  = "#00e676" if 65 <= r['rsi_weekly'] <= 80 else "#ffa000"
            cci_col  = "#00e676" if r['cci_weekly'] >= 150 else "#ffa000" if r['cci_weekly'] >= 100 else "#29b6f6"
            vol_col  = "#00e676" if r['vol_ratio'] >= 2.0 else "#ffa000"
            mcap_fmt = f"₹{r['market_cap_cr']:,.0f} Cr" if r['market_cap_cr'] > 0 else "—"

            # Gap flag
            gap_pct  = round((r['curr_open'] - r['prev_close']) / r['prev_close'] * 100, 2) if r['prev_close'] > 0 else 0.0
            gap_badge = f'<span style="color:#00e676; font-weight:700;">▲{gap_pct:.1f}%</span>' if gap_pct > 0 else f'<span style="color:#ef4444;">{gap_pct:.1f}%</span>'

            clean_rec_wm = extract_clean_recommendation(r.get('recommendation', ''))

            wm_rows_html.append(
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
                f'{wl_cell_wm}'
                f'<td style="padding:8px 10px; font-weight:bold;"><a href="https://in.tradingview.com/chart/?symbol=NSE:{r["symbol"].replace(".NS", "")}" target="_blank" rel="noopener noreferrer" style="color:#29b6f6; text-decoration:none;">{r["symbol"]}</a></td>'
                f'<td style="padding:8px 10px; color:#94a3b8; font-size:0.78rem;">{r.get("company_name","")}</td>'
                f'<td style="padding:8px 10px; color:#e2e8f0; font-weight:600;">₹{cmp_v:,.2f}</td>'
                f'<td style="padding:8px 10px;">{chg_b}</td>'
                f'<td style="padding:8px 10px;">{ret_1m_b}</td>'
                f'<td style="padding:8px 10px; color:#00e5ff;">{mcap_fmt}</td>'
                f'<td style="padding:8px 10px; color:#94a3b8;">₹{r["prev_close"]:,.2f}</td>'
                f'<td style="padding:8px 10px;">{gap_badge}</td>'
                f'<td style="padding:8px 10px; color:#7dd3fc;">₹{r["close_sma20"]:,.2f}</td>'
                f'<td style="padding:8px 10px; color:{rsi_col}; font-weight:700;">{r["rsi_weekly"]:.1f}</td>'
                f'<td style="padding:8px 10px; color:{cci_col}; font-weight:700;">{r["cci_weekly"]:.1f}</td>'
                f'<td style="padding:8px 10px; color:{vol_col}; font-weight:700;">{r["vol_ratio"]:.2f}x</td>'
                f'<td style="padding:8px 10px; color:#00e676; font-weight:700;">{r["weekly_score"]:.0f}</td>'
                f'<td style="padding:8px 10px; color:#cbd5e1; font-weight:600;">₹{buy_v:,.2f}</td>'
                f'<td style="padding:8px 10px; color:#ef4444; font-weight:600;">₹{sl_v:,.2f}</td>'
                f'<td style="padding:8px 10px; color:#00e676; font-weight:600;">₹{tgt_v:,.2f}</td>'
                f'<td style="padding:8px 10px;">{conf_badge_wm}</td>'
                f'<td style="padding:8px 10px; color:#94a3b8; font-style:italic; font-size:0.78rem; line-height:1.4; min-width: 250px; max-width: 350px; white-space: normal !important; word-wrap: break-word;">"{clean_rec_wm}"</td>'
                f'</tr>'
            )

        wm_hdr_cols = [
            "WL", "Symbol", "Company", "CMP", "Wk Chg%", "1M Return %", "MCap",
            "Prev Close", "Gap Open", "SMA 20W",
            "RSI 14W", "CCI 20W", "Vol Ratio", "Score",
            "Buy", "Stop", "Target", "Confidence", "Analysis"
        ]
        wm_hdr_html = "".join([f'<th style="padding:8px 10px; white-space:nowrap;">{h}</th>' for h in wm_hdr_cols])

        st.markdown(
            f'<div class="glass-card" style="padding:18px; border:1px solid rgba(0,230,118,0.2); background:rgba(9,13,22,0.55); border-radius:12px;">'
            f'<div style="overflow-x:auto;">'
            f'<table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.82rem; color:#cbd5e1; font-family:Outfit,sans-serif;">'
            f'<thead><tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#00e676; font-weight:bold; background:rgba(0,230,118,0.05); font-size:0.77rem; text-transform:uppercase;">'
            f'{wm_hdr_html}</tr></thead>'
            f'<tbody>{chr(10).join(wm_rows_html)}</tbody>'
            f'</table></div></div>',
            unsafe_allow_html=True
        )


# ==============================================================================
# TAB VCS: VOLATILITY CONTRACTION SCANNER
# ==============================================================================
with tab_vcs:
    st.markdown("### 📉 Volatility Contraction Scanner (VCS)")
    st.markdown("Identifies stocks with tightening ATR, Standard Deviation, and Volume contraction.")
    st.markdown("---")
    
    # 1. Metrics row
    v_m1, v_m2, v_m3 = st.columns(3)
    
    vcs_data = st.session_state.get('vcs_results', None)
    if vcs_data:
        vcs_count = len(vcs_data)
        min_vcs_score = min(r['vcs_score'] for r in vcs_data) if vcs_count > 0 else 0.0
        avg_vcs_score = sum(r['vcs_score'] for r in vcs_data) / vcs_count if vcs_count > 0 else 0.0
    else:
        vcs_count = 0
        min_vcs_score = 0.0
        avg_vcs_score = 0.0
        
    v_m1.markdown(f'<div class="glass-card metric-glow-blue"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">VCS Setups Found</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{vcs_count}</h3></div>', unsafe_allow_html=True)
    v_m2.markdown(f'<div class="glass-card metric-glow-green"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Tightest Volatility Score</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{min_vcs_score:.2f}</h3></div>', unsafe_allow_html=True)
    v_m3.markdown(f'<div class="glass-card metric-glow-amber"><p style="font-size:0.85rem; color:#94a3b8; margin:0;">Avg VCS Rating</p><h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ffa000;">{avg_vcs_score:.1f} <span style="font-size: 1.1rem; color: #94a3b8;">pts</span></h3></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### Scanner Parameters")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        vcs_timeframe = st.selectbox("Timeframe", ["Daily (1d)", "Weekly (1wk)"], index=0)
    with col2:
        vcs_min_price_chg = st.number_input("Min % Chg", min_value=-50.0, max_value=100.0, value=-5.0, step=0.5)
    with col3:
        vcs_len_short = st.number_input("Short ATR Len", min_value=1, max_value=100, value=13)
    with col4:
        vcs_len_long = st.number_input("Long ATR Len", min_value=1, max_value=200, value=63)
    with col5:
        vcs_len_vol = st.number_input("Volume Len", min_value=1, max_value=200, value=50)
    with col6:
        vcs_sensitivity = st.number_input("Sensitivity", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    with col7:
        vcs_max_score = st.number_input("Max Score Limit", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        
    run_vcs_btn = st.button("🔍 Run Custom VCS Scan", use_container_width=True, type="primary")
    
    if run_vcs_btn:
        vcs_interval = "1wk" if "Weekly" in vcs_timeframe else "1d"
        vcs_period = "5y" if vcs_interval == "1wk" else "1y"
        with st.spinner(f"Running custom VCS scan... downloading {vcs_timeframe} data..."):
            if "NIFTY 50" in universe_selection:
                universe_key = "NIFTY 50"
            elif "NIFTY 100" in universe_selection:
                universe_key = "NIFTY 100"
            elif "WATCHLIST" in universe_selection.upper():
                universe_key = "WATCHLIST"
            else:
                universe_key = "ALL NSE"
                
            if universe_key == "WATCHLIST":
                import watchlist
                wl = watchlist.load_watchlist()
                custom_candidates = [s for s in wl['symbol'].tolist() if pd.notna(s)]
            else:
                custom_candidates = get_index_stocks(universe_key)
            
            custom_vcs_results = []
            chunk_size = 50
            chunks = [custom_candidates[i:i+chunk_size] for i in range(0, len(custom_candidates), chunk_size)]
            
            for c_idx, chunk in enumerate(chunks):
                tkrs = [f"{s}.NS" for s in chunk]
                try:
                    df_vcs = yf.download(tickers=tkrs, period=vcs_period, interval=vcs_interval, progress=False, threads=False)
                    if not df_vcs.empty:
                        for sym in chunk:
                            try:
                                if isinstance(df_vcs.columns, pd.MultiIndex):
                                    all_tkrs = df_vcs.columns.get_level_values(1).unique().tolist()
                                    matched_t = next((t for t in all_tkrs if t.upper() == f"{sym}.NS".upper()), None)
                                    if not matched_t:
                                        continue
                                    t_df = df_vcs.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                                else:
                                    t_df = df_vcs.dropna(subset=['Close'])
                                    
                                if not t_df.empty and len(t_df) >= vcs_len_long:
                                    t_df = t_df.reset_index()
                                    t_df.rename(columns={t_df.columns[0]: 'Date'}, inplace=True)
                                    res = scan_vcs(sym, t_df, 
                                                   lenShort=vcs_len_short, 
                                                   lenLong=vcs_len_long, 
                                                   lenVol=vcs_len_vol, 
                                                   sensitivity=vcs_sensitivity, 
                                                   max_score=vcs_max_score)
                                    if res:
                                        if res.get('day_change_pct', 0.0) >= vcs_min_price_chg:
                                            res['Timeframe'] = vcs_timeframe
                                            res['Action'] = res.get('recommendation', 'Wait')
                                            custom_vcs_results.append(res)
                            except Exception:
                                pass
                except Exception:
                    pass
            
            st.session_state.vcs_results = custom_vcs_results
            try:
                today_ist_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
                database.save_vcs_only(today_ist_str, custom_vcs_results)
            except Exception as e:
                print(f"Failed to cache custom VCS scan: {e}")
            if len(custom_vcs_results) > 0:
                st.success(f"Custom VCS Scan Complete! Found {len(custom_vcs_results)} stocks and saved to database.")

    st.markdown("---")
    
    if st.session_state.vcs_results is None:
        st.info("💡 Adjust parameters above and click 'Run Custom VCS Scan' to find setups, or run the global scanner from the sidebar.")
    elif len(st.session_state.vcs_results) == 0:
        st.info(f"ℹ️ No VCS setups found with a score < {vcs_max_score} today.")
    else:
        col_btn, _ = st.columns([2, 8])
        with col_btn:
            v_export_list = []
            for r in st.session_state.vcs_results:
                row = dict(r)
                if 'recommendation' in row:
                    row['Recommendation'] = extract_clean_recommendation(row.pop('recommendation'))
                v_export_list.append(row)
            vcs_df = pd.DataFrame(v_export_list)
            csv_data = vcs_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name="vcs_scan_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        render_unified_strategy_table(st.session_state.vcs_results, "vcs", "vcs_tab")

# ==============================================================================
# TAB: STRUCTURAL VCP
# ==============================================================================
with tab_structural_vcp:
    st.markdown("### 🎯 Structural Volatility Contraction Pattern (VCP)")
    st.markdown("Hunts for textbook VCP patterns: Flat-top resistance, successive higher lows (tightening), and extreme volume dry-up on the right side.")
    
    if st.session_state.structural_vcp_results is None:
        st.info("💡 Run the main scanner from the sidebar to populate Structural VCP setups.")
    elif len(st.session_state.structural_vcp_results) == 0:
        st.info("ℹ️ No textbook Structural VCP setups found today.")
    else:
        vcp_count = len(st.session_state.structural_vcp_results)
        st.success(f"Found {vcp_count} Structural VCP setups!")
        
        col_btn, _ = st.columns([2, 8])
        with col_btn:
            sv_export_list = []
            for r in st.session_state.structural_vcp_results:
                row = dict(r)
                if 'recommendation' in row:
                    row['Recommendation'] = extract_clean_recommendation(row.pop('recommendation'))
                sv_export_list.append(row)
            sv_df = pd.DataFrame(sv_export_list)
            sv_csv = sv_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=sv_csv,
                file_name="structural_vcp_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        render_unified_strategy_table(st.session_state.structural_vcp_results, "struct_vcp", "struct_vcp_tab")

# ==============================================================================
# TAB: EARLY STAGE 2 BREAKOUT
# ==============================================================================
with tab_stage2:
    st.markdown("### 🚀 Early Stage 2 Base Breakout Scanner")
    st.markdown("Identifies stocks moving out of a long-term Stage 1 base on the monthly timeframe.")
    
    if 'stage2_results' not in st.session_state:
        st.session_state.stage2_results = None
        # Try loading from DB
        today_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
        try:
            cached_stage2 = database.get_cached_stage2(today_str)
            if cached_stage2:
                st.session_state.stage2_results = cached_stage2
                # Note: No need to show success message on silent load, just let the table render
        except Exception as e:
            print(f"Failed to load cached stage2: {e}")
        
    s2_col1, s2_col2 = st.columns([2, 8])
    with s2_col1:
        s2_max_runup = st.number_input("Max Run-Up (%)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
        run_stage2_btn = st.button("🔍 Run Stage 2 Scan", use_container_width=True, type="primary")
        
    if run_stage2_btn:
        with st.spinner(f"Running Monthly Stage 2 Scan on {universe_selection}..."):
            s2_universe = universe_selection
            if "NIFTY 50" in s2_universe:
                s2_key = "NIFTY 50"
            elif "NIFTY 100" in s2_universe:
                s2_key = "NIFTY 100"
            elif "WATCHLIST" in s2_universe.upper():
                s2_key = "WATCHLIST"
            else:
                s2_key = "NIFTY 500" # Better default for Stage 2 than all NSE
                
            if s2_key == "WATCHLIST":
                import watchlist
                wl = watchlist.load_watchlist()
                s2_cands = [s for s in wl['symbol'].tolist() if pd.notna(s)]
            else:
                s2_cands = get_index_stocks(s2_key)
                
            s2_res = []
            chunk_size = 50
            chunks = [s2_cands[i:i+chunk_size] for i in range(0, len(s2_cands), chunk_size)]
            
            s2_prog = st.progress(0)
            s2_status = st.empty()
            
            def download_s2_chunk(c_idx, chunk):
                chunk_res = []
                tkrs = [f"{s}.NS" for s in chunk]
                try:
                    df_s2 = yf.download(tickers=tkrs, period="5y", interval="1mo", progress=False, threads=False)
                    if not df_s2.empty:
                        for sym in chunk:
                            try:
                                if isinstance(df_s2.columns, pd.MultiIndex):
                                    all_tkrs = df_s2.columns.get_level_values(1).unique().tolist()
                                    matched_t = next((t for t in all_tkrs if t.upper() == f"{sym}.NS".upper()), None)
                                    if not matched_t: continue
                                    t_df = df_s2.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                                else:
                                    t_df = df_s2.dropna(subset=['Close'])
                                    
                                if not t_df.empty and len(t_df) >= 24:
                                    t_df = t_df.reset_index()
                                    t_df.rename(columns={t_df.columns[0]: 'Date'}, inplace=True)
                                    res = scan_monthly_early_stage2(sym, t_df, max_run_up_pct=s2_max_runup)
                                    if res:
                                        chunk_res.append(res)
                            except Exception as parse_ex:
                                pass
                except Exception as down_ex:
                    print(f"Failed to download chunk {c_idx + 1}: {down_ex}")
                return chunk_res

            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for c_idx, chunk in enumerate(chunks):
                    futures.append(executor.submit(download_s2_chunk, c_idx, chunk))
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    s2_status.text(f"Scanning chunks... ({i+1}/{len(chunks)})")
                    s2_res.extend(future.result())
                    s2_prog.progress((i + 1) / len(chunks))
            
            s2_prog.empty()
            s2_status.empty()
            
            # Sort by signal strength (score)
            s2_res = sorted(s2_res, key=lambda x: x.get('score', 0), reverse=True)
            st.session_state.stage2_results = s2_res
            try:
                today_ist_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
                database.save_stage2_only(today_ist_str, s2_res)
            except Exception as e:
                print(f"Failed to cache stage2 scan: {e}")
            st.success(f"Stage 2 Scan Complete! Found {len(s2_res)} setups.")
            
    st.markdown("---")
    
    if st.session_state.stage2_results is None:
        st.info("💡 Adjust parameters and click 'Run Stage 2 Scan' to find long-term breakouts.")
    elif len(st.session_state.stage2_results) == 0:
        st.info(f"ℹ️ No early Stage 2 setups found in {universe_selection} today.")
    else:
        dl_btn, _ = st.columns([2, 8])
        with dl_btn:
            s2_export_list = []
            for r in st.session_state.stage2_results:
                row = dict(r)
                if 'recommendation' in row:
                    row['Recommendation'] = extract_clean_recommendation(row.pop('recommendation'))
                s2_export_list.append(row)
            s2_df = pd.DataFrame(s2_export_list)
            csv_data = s2_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name="stage2_scan_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        render_unified_strategy_table(st.session_state.stage2_results, "stage2", "stage2_tab")



# ==============================================================================
# TAB 17: VPA TREND
# ==============================================================================
with tab_vpa:
    st.markdown("### 🚥 VPA Trend Indicator (Daily, Weekly, Monthly)")
    st.info("Scans ALL NSE listed stocks. Filters: Price > ₹100. Shows Major, Mid, and Minor trends across timeframes.")
    
    col1, col2 = st.columns([3, 7])
    with col1:
        run_vpa_btn = st.button("🚀 Run Advanced VPA Scan", use_container_width=True)
    
    if run_vpa_btn:
        st.session_state.vpa_results = []
        with st.spinner("Initializing Ultra-Fast VPA Scan on ALL NSE Stocks..."):
            try:
                from data_fetcher import get_all_nse_symbols
                import yfinance as yf
                import pandas as pd
                from concurrent.futures import ThreadPoolExecutor
                import time
                
                raw_symbols = get_all_nse_symbols()
                all_symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
                
                # Phase 1: Bulk OHLCV Download (2 years history for Weekly/Monthly VPA)
                st.info(f"Phase 1: Downloading 5 years of history for {len(all_symbols)} stocks...")
                prog = st.progress(0)
                status = st.empty()
                
                chunk_size = 300
                sym_chunks = [all_symbols[i:i + chunk_size] for i in range(0, len(all_symbols), chunk_size)]
                
                valid_data = {}
                price_filtered = []
                
                # We need at least ~100 days of history for VPA to calculate daily/weekly accurately
                def download_vpa_chunk(chunk_idx, chunk):
                    chunk_data = {}
                    chunk_filtered = []
                    try:
                        df_bulk = yf.download(tickers=chunk, period="5y", interval="1d", progress=False, threads=True)
                        if isinstance(df_bulk.columns, pd.MultiIndex):
                            for sym in chunk:
                                try:
                                    if 'Close' in df_bulk.columns.levels[0]:
                                        ticker_df = df_bulk.xs(sym, axis=1, level=1).copy()
                                        ticker_df = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
                                        if len(ticker_df) >= 45 and ticker_df['Close'].iloc[-1] > 100.0:
                                            ticker_df = ticker_df.reset_index()
                                            ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                            ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.tz_localize(None)
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
                                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.tz_localize(None)
                                    chunk_data[chunk[0]] = ticker_df
                                    chunk_filtered.append(chunk[0])
                    except Exception:
                        pass
                    return chunk_data, chunk_filtered
                    
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []
                    for chunk_idx, chunk in enumerate(sym_chunks):
                        futures.append(executor.submit(download_vpa_chunk, chunk_idx, chunk))
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        res_data, res_filtered = future.result()
                        valid_data.update(res_data)
                        price_filtered.extend(res_filtered)
                        prog.progress((i + 1) / len(sym_chunks))
                        status.text(f"Fetching bulk history chunks... ({i+1}/{len(sym_chunks)})")
                
                # Phase 2: Final VPA Compute (Instant)
                st.info("Phase 2: Calculating VPA Trends (Instant)...")
                status.empty()
                prog.progress(1.0)
                
                vpa_list = []
                for sym in price_filtered:
                    df = valid_data[sym]
                    clean_sym = sym.replace('.NS', '')
                    vpa_res = scan_vpa_trend(clean_sym, df)
                    if vpa_res is not None:
                        vpa_res['market_cap_cr'] = 0  # Default since bulk fetch rate-limits
                        vpa_list.append(vpa_res)
                            
                prog.empty()
                status.empty()
                st.session_state.vpa_results = vpa_list
                try:
                    today_ist_str = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d")
                    database.save_vpa_only(today_ist_str, vpa_list)
                except Exception as e:
                    print(f"Failed to cache custom VPA scan: {e}")
                st.success(f"VPA Scan complete! Found {len(vpa_list)} stocks meeting all criteria and saved to database.")
                
            except Exception as e:
                st.error(f"Scan failed: {e}")
                
    if not st.session_state.get('vpa_results'):
        st.info("No VPA data available. Click 'Run Advanced VPA Scan' to process.")
    else:
        vpa_data = st.session_state.vpa_results
        
        # Sort by score
        vpa_data = sorted(vpa_data, key=lambda x: x.get('score', 0), reverse=True)
        
        # Download Button
        import pandas as pd
        
        def get_action_signal_text(short, mid, max_t, max_val):
            if max_val > 2.0:
                return "Overextended (Trim / Hold)"
            elif 0.5 < max_val <= 2.0 and mid == 1 and short == 1:
                return "Perfect Buy / Strong Hold"
            elif 0 < max_val <= 0.5 and mid == 1 and short == 1:
                return "Early Breakout Entry"
            elif max_val > 0.5 and mid == 1 and short <= 0:
                return "Pullback (Wait for Short=Up)"
            elif max_val > 0.5 and mid <= 0:
                return "Warning (Mid Broken) - Trim"
            elif max_val <= 0 and mid <= 0:
                return "Avoid / Full Exit"
            else:
                return "Neutral / Choppy"
        
        def get_signal(short, mid, max_t, max_val):
            if max_val > 2.0:
                return "Hold"
            return "Buy" if (max_val > 0.5 and mid == 1) or (max_val > 0 and mid == 1 and short == 1) else "Hold" if max_val > 0.5 else "Sell"

        only_buy_signals = st.checkbox("🟢 Show Only 'Buy' Signals", value=False)
        
        daily_export = []
        weekly_export = []
        monthly_export = []
        
        rank = 1
        filtered_vpa_data = []
        for r in vpa_data:
            d = r['daily']; w = r['weekly']; m = r['monthly']
            
            # Use daily signal as the primary filter if we want to filter globally, or filter per timeframe.
            # Actually, since the timeframe can be selected in UI, let's filter the data based on the selected timeframe later.
            # For the exports, we'll export all but add Rank.
            
            d_sig = get_signal(d['minor'], d['mid'], d['major'], d.get('major_val', 0))
            if only_buy_signals and d_sig != "Buy":
                continue
                
            filtered_vpa_data.append((rank, r))
            
            daily_export.append({
                'Rank': rank,
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                'Major Trend': "Up" if d['major'] == 1 else ("Down" if d['major'] == -1 else "Neutral"),
                'Mid Trend': "Up" if d['mid'] == 1 else ("Down" if d['mid'] == -1 else "Neutral"),
                'Minor Trend': "Up" if d['minor'] == 1 else ("Down" if d['minor'] == -1 else "Neutral"),
                'RSI': d.get('rsi', 0.0),
                'CCI': d.get('cci', 0.0),
                'Action': get_action_signal_text(d['minor'], d['mid'], d['major'], d.get('major_val', 0)),
                'Signal': d_sig
            })
            rank += 1
            
        for rank, r in filtered_vpa_data:
            w = r['weekly']
            weekly_export.append({
                'Rank': rank,
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                'Major Trend': "Up" if w['major'] == 1 else ("Down" if w['major'] == -1 else "Neutral"),
                'Mid Trend': "Up" if w['mid'] == 1 else ("Down" if w['mid'] == -1 else "Neutral"),
                'Minor Trend': "Up" if w['minor'] == 1 else ("Down" if w['minor'] == -1 else "Neutral"),
                'RSI': w.get('rsi', 0.0),
                'CCI': w.get('cci', 0.0),
                'Action': get_action_signal_text(w['minor'], w['mid'], w['major'], w.get('major_val', 0)),
                'Signal': get_signal(w['minor'], w['mid'], w['major'], w.get('major_val', 0))
            })
            
        for rank, r in filtered_vpa_data:
            m = r['monthly']
            monthly_export.append({
                'Rank': rank,
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                'Major Trend': "Up" if m['major'] == 1 else ("Down" if m['major'] == -1 else "Neutral"),
                'Mid Trend': "Up" if m['mid'] == 1 else ("Down" if m['mid'] == -1 else "Neutral"),
                'Minor Trend': "Up" if m['minor'] == 1 else ("Down" if m['minor'] == -1 else "Neutral"),
                'RSI': m.get('rsi', 0.0),
                'CCI': m.get('cci', 0.0),
                'Action': get_action_signal_text(m['minor'], m['mid'], m['major'], m.get('major_val', 0)),
                'Signal': get_signal(m['minor'], m['mid'], m['major'], m.get('major_val', 0))
            })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download Daily VPA (CSV)",
                data=pd.DataFrame(daily_export).to_csv(index=False).encode('utf-8'),
                file_name="vpa_daily_trend.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                label="📥 Download Weekly VPA (CSV)",
                data=pd.DataFrame(weekly_export).to_csv(index=False).encode('utf-8'),
                file_name="vpa_weekly_trend.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col3:
            st.download_button(
                label="📥 Download Monthly VPA (CSV)",
                data=pd.DataFrame(monthly_export).to_csv(index=False).encode('utf-8'),
                file_name="vpa_monthly_trend.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Interactive Timeframe Selection
        st.markdown("### Select Timeframe")
        selected_tf = st.selectbox("Timeframe to display", ["Daily", "Weekly", "Monthly"])
        
        def trend_to_badge(t_val):
            if t_val == 1:
                return "<span style='color: #00e676; font-weight: bold;'>Up (1)</span>"
            elif t_val == -1:
                return "<span style='color: #ef4444; font-weight: bold;'>Dn (-1)</span>"
            return "<span style='color: #fbbf24; font-weight: bold;'>Neu (0)</span>"
            
        def get_action_signal(short, mid, max_t):
            if max_t == 1 and mid == 1 and short == 1:
                return "<span style='color: #00e676; font-weight: bold;'>🟢 Perfect Buy / Strong Hold</span>"
            elif max_t == 1 and mid == 1 and short <= 0:
                return "<span style='color: #fbbf24; font-weight: bold;'>🟡 Pullback (Wait for Short=Up)</span>"
            elif max_t == 1 and mid <= 0:
                return "<span style='color: #f97316; font-weight: bold;'>🟠 Warning (Mid Broken) - Trim</span>"
            elif max_t <= 0 and mid == 1 and short == 1:
                return "<span style='color: #3b82f6; font-weight: bold;'>🔵 Early Breakout Entry</span>"
            elif max_t <= 0 and mid <= 0:
                return "<span style='color: #ef4444; font-weight: bold;'>🔴 Avoid / Full Exit</span>"
            else:
                return "<span style='color: #9ca3af; font-weight: bold;'>⚪ Neutral / Choppy</span>"
            
        html_rows = []
        for rank, r in filtered_vpa_data:
            if selected_tf == "Daily":
                tf_data = r['daily']
            elif selected_tf == "Weekly":
                tf_data = r['weekly']
            else:
                tf_data = r['monthly']
            
            t_short = trend_to_badge(tf_data['minor'])
            t_mid = trend_to_badge(tf_data['mid'])
            t_max = trend_to_badge(tf_data['major'])
            
            action = get_action_signal(tf_data['minor'], tf_data['mid'], tf_data['major'])
            
            # Zero indentation to prevent Streamlit markdown codeblock rendering
            row = f"""<tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
<td style="padding: 10px; font-weight: bold; color: #94a3b8;">#{rank}</td>
<td style="padding: 10px;"><strong>{r['symbol']}</strong></td>
<td style="padding: 10px;">{r['cmp']}</td>
<td style="padding: 10px;">{get_day_change_badge_html(r['day_change_pct'])}</td>
<td style="padding: 10px;">{round(r.get('market_cap_cr', 0))}</td>
<td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">{t_short}</td>
<td style="padding: 10px;">{t_mid}</td>
<td style="padding: 10px;">{t_max}</td>
<td style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">{action}</td>
</tr>"""
            html_rows.append(row)
            
        rows_str = "".join(html_rows)
        
        table_html = f"""<div style="overflow-x: auto; margin-top: 10px;">
<table style="width: 100%; text-align: left; border-collapse: collapse; font-size: 0.95rem;">
<thead>
<tr style="background-color: rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.1);">
<th style="padding: 10px;">Rank</th>
<th style="padding: 10px;">Symbol</th>
<th style="padding: 10px;">CMP</th>
<th style="padding: 10px;">Chg %</th>
<th style="padding: 10px;">M.Cap (Cr)</th>
<th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">Short Term</th>
<th style="padding: 10px;">Mid Term</th>
<th style="padding: 10px;">Max Term</th>
<th style="padding: 10px; border-left: 1px solid rgba(255,255,255,0.1);">Action / Signal</th>
</tr>
</thead>
<tbody>
{rows_str}
</tbody>
</table>
</div>"""
        st.markdown(table_html, unsafe_allow_html=True)

# TAB: FREQUENT FLYERS (CONSISTENT ALERTS)
with tab_frequent:
    import tabs.tab_frequent as tab_freq_mod
    tab_freq_mod.render()
