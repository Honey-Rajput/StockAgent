# ui_components.py
# Shared Streamlit UI helpers for strategy result tables and trade setup cards.

import json

import streamlit as st

import watchlist
from data_fetcher import get_stock_sector
from utils import get_day_change_badge_html, get_signal_badge_html


def matches_sma_timeframe_filter(record: dict, timeframe: str) -> bool:
    """
    Filter above_ma scan rows by Daily / Weekly / All timeframe.
    Records loaded from DB before passes_* columns existed have no flags — include them
    so cached results still appear (same as pre-filter behavior).
    """
    passes_daily = record.get("passes_daily")
    passes_weekly = record.get("passes_weekly")
    if passes_daily is None and passes_weekly is None:
        return True
    if timeframe == "Daily":
        return bool(passes_daily)
    if timeframe == "Weekly":
        return bool(passes_weekly)
    if timeframe == "All (Daily + Weekly Convergence)":
        return bool(passes_daily) and bool(passes_weekly)
    return True


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
        "Remaining Target %": lambda x: float((((x.get('target_price') or 0.0) - (x.get('cmp') or 1.0)) / (x.get('cmp') or 1.0) * 100) if (x.get('cmp') or 0.0) > 0 else 0.0)
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
        
        cells.append(f'<td style="padding: 10px 12px; color: #e2e8f0; font-weight: 500;">₹{r.get("cmp", 0.0):,.2f}</td>')
        
        if strategy_type == "vdu_breakout":
            setup_val = r.get('setup_type') or 'VDU Breakout'
            if 'Pre-Breakout' in setup_val:
                setup_badge = f'<span class="custom-badge" style="background: rgba(255,160,0,0.15); color: #ffa000; border: 1px solid #ffa000; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px;">⏳ Pre-Breakout</span>'
            else:
                setup_badge = f'<span class="custom-badge" style="background: rgba(0,230,118,0.15); color: #00e676; border: 1px solid #00e676; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px;">🚀 Breakout</span>'
            cells.append(f'<td style="padding: 10px 12px; font-weight: 600;">{setup_badge}</td>')
            
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
                d20 = float(r.get('dist_20sma_pct') or 0.0)
                d50 = float(r.get('dist_50sma_pct') or 0.0)
                d200 = float(r.get('dist_200sma_pct') or 0.0)
                cells.append(f'<td style="padding: 10px 12px; font-size:0.85rem;"><span style="color:#00e676;">20: +{d20:.1f}%</span><br><span style="color:#29b6f6;">50: +{d50:.1f}%</span><br><span style="color:#ffa000;">200: +{d200:.1f}%</span></td>')
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
            
        elif strategy_type == "bb_squeeze":
            setup_badge = f'<span class="custom-badge" style="background: rgba(41,182,246,0.15); color: #29b6f6; border: 1px solid #29b6f6; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px;">{r.get("setup", "")}</span>'
            cells.append(f'<td style="padding: 10px 12px; font-weight: 600;">{setup_badge}</td>')
            chg_badge = get_day_change_badge_html(r.get('day_change_pct', 0.0))
            cells.append(f'<td style="padding: 10px 12px;">{chg_badge}</td>')
            d9 = r.get("dist_9ema")
            d9 = 0.0 if d9 is None else float(d9)
            d21 = r.get("dist_21ema")
            d21 = 0.0 if d21 is None else float(d21)
            d9_col = "#00e676" if d9 >= 0 else "#ef4444"
            d21_col = "#00e676" if d21 >= 0 else "#ef4444"
            cells.append(f'<td style="padding: 10px 12px; color: {d9_col}; font-weight: 600;">{d9:+.2f}%</td>')
            cells.append(f'<td style="padding: 10px 12px; color: {d21_col}; font-weight: 600;">{d21:+.2f}%</td>')
            cross_badge = '<span style="color:#00e676;">Yes ✅</span>' if r.get('crossover') else '<span style="color:#94a3b8;">No</span>'
            cells.append(f'<td style="padding: 10px 12px;">{cross_badge}</td>')
            
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
        headers.extend(["Setup", "Day Chg %", "Volume", "Dry Avg Vol", "Vol Ratio", "Dry Days", "Spikes", "Score"])
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
    elif strategy_type == "bb_squeeze":
        headers.extend(["Setup", "Day Chg %", "Dist to 9 EMA", "Dist to 21 EMA", "Crossover"])
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
