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
    st.markdown("### 🏆 Minervini Trend Template — Stage Analyzer")
    st.markdown("Analyzes Minervini's 8 Trend Template criteria to classify stocks into Stage 1, Stage 2 (Uptrend), Stage 3 (Topping), or Stage 4 (Decline).")
    
    col_sa1, col_sa2 = st.columns([1, 2])
    run_sa_btn = col_sa1.button("🔍 Run Stage Analysis Scan", type="primary", use_container_width=True)
    
    if run_sa_btn:
        st.session_state.stage_analysis_results = None
        
        with st.spinner(f"Running Stage Analysis Scan on {st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)')}..."):
            import yfinance as yf
            import concurrent.futures
           
            from scanner import scan_stage_analysis
            from data_fetcher import get_index_stocks, get_all_nse_symbols
            
            # Fetch NIFTY 50 return
            try:
                nifty_df = yf.download("^NSEI", period="1y", interval="1d", progress=False, timeout=15)
                if len(nifty_df) >= 127:
                    bC = float(nifty_df['Close'].iloc[-1].item() if hasattr(nifty_df['Close'].iloc[-1], 'item') else nifty_df['Close'].iloc[-1])
                    bCold = float(nifty_df['Close'].iloc[-127].item() if hasattr(nifty_df['Close'].iloc[-127], 'item') else nifty_df['Close'].iloc[-127])
                    bRet = (bC - bCold) / bCold
                else:
                    bRet = 0.0
            except Exception as e:
                print(f"Error fetching benchmark: {e}")
                bRet = 0.0
                
            sa_universe = "ALL NSE"
            if "NIFTY 500" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): sa_universe = "NIFTY 500"
            elif "NIFTY 100" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): sa_universe = "NIFTY 100"
            elif "NIFTY 50" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): sa_universe = "NIFTY 50"
            elif "WATCHLIST" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)').upper(): sa_universe = "WATCHLIST"
            
            raw_symbols = get_index_stocks(sa_universe) if sa_universe != "ALL NSE" else get_all_nse_symbols()
            all_syms = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results_list = []
            chunk_size = 50
            chunks = [all_syms[i:i + chunk_size] for i in range(0, len(all_syms), chunk_size)]
            
            def process_sa_chunk(c_idx, chunk):
                chunk_results = []
                try:
                    chunk_ns = [s if s.endswith('.NS') else f"{s}.NS" for s in chunk]
                    data = yf.download(chunk_ns, period="2y", interval="1d", progress=False, threads=False, timeout=15)
                    for sym in chunk:
                        try:
                            sym_ns = sym if sym.endswith('.NS') else f"{sym}.NS"
                            if isinstance(data.columns, pd.MultiIndex):
                                all_tkrs = data.columns.get_level_values(1).unique().tolist()
                                matched_t = next((t for t in all_tkrs if t.upper() == sym_ns.upper()), None)
                                if not matched_t:
                                    continue
                                df = data.xs(matched_t, axis=1, level=1).copy()
                            else:
                                if len(chunk) == 1:
                                    df = data.copy()
                                else:
                                    continue
                            df = df.dropna(subset=['Close'])
                            if len(df) >= 50:
                                res = scan_stage_analysis(sym, df, bRet)
                                if res: chunk_results.append(res)
                        except Exception as e: pass
                except Exception as e: pass
                return chunk_results
                
            import time
            for c_idx, chunk in enumerate(chunks):
                status_text.text(f"Scanning chunk {c_idx+1}/{len(chunks)}...")
                chunk_res = process_sa_chunk(c_idx, chunk)
                results_list.extend(chunk_res)
                progress_bar.progress((c_idx + 1) / len(chunks))
                time.sleep(0.5) # Throttle to prevent rate limit
                
            today_str = get_market_date(for_display=False)
            database.save_stage_analysis_only(today_str, results_list)
            st.session_state.stage_analysis_results = results_list
            status_text.text("✅ Stage Analysis Scan Complete!")
            st.rerun()

    # Display logic
    if 'stage_analysis_results' not in st.session_state:
        st.session_state.stage_analysis_results = None
        
    sa_today_str = get_market_date(for_display=True)
    if st.session_state.stage_analysis_results is None:
        st.session_state.stage_analysis_results = database.get_cached_stage_analysis(sa_today_str)
        
    if st.session_state.stage_analysis_results is not None:
        sa_list = st.session_state.stage_analysis_results
        
        # Apply Universe Filter
        if "ALL NSE" not in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)').upper() and len(sa_list) > 0:
            from data_fetcher import get_index_stocks
            resolved_univ = "ALL NSE"
            if "NIFTY 500" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): resolved_univ = "NIFTY 500"
            elif "NIFTY 100" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): resolved_univ = "NIFTY 100"
            elif "NIFTY 50" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)'): resolved_univ = "NIFTY 50"
            elif "WATCHLIST" in st.session_state.get('universe_selection', 'Top 1000 NSE Stocks (By Market Cap)').upper(): resolved_univ = "WATCHLIST"
            if resolved_univ != "ALL NSE":
                raw_symbols = get_index_stocks(resolved_univ)
                valid_set = set([str(s).replace('.NS', '').strip().upper() for s in raw_symbols if str(s).strip()])
                sa_list = [r for r in sa_list if r['symbol'] in valid_set]
                
        if len(sa_list) > 0:
            st.markdown(f"### 📊 Stage Analysis Setups ({len(sa_list)} stocks)")
            
            # Excel Download button
            import io
            df_export = pd.DataFrame(sa_list)
            if 'sRet' in df_export.columns:
                df_export.rename(columns={'sRet': 'sret'}, inplace=True)
            df_export['symbol'] = df_export['symbol'].astype(str).str.replace('.NS', '', regex=False)
            df_export = df_export[['symbol', 'company_name', 'cmp', 'stage', 'template_str', 'score', 'sret', 'lo52', 'hi52']]
            df_export.columns = ['Symbol', 'Company', 'CMP', 'Stage', 'Template', 'Score', '6M Return', '52W Low', '52W High']
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df_export.to_excel(writer, index=False, sheet_name='Stage Analysis')
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer.getvalue(),
                file_name=f"Stage_Analysis_{sa_today_str}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_sa_excel"
            )
            
            # HTML Table rendering
            html_parts = []
            html_parts.append('<table class="styled-table" style="width:100%; border-collapse:collapse; background:#0D1B2A; color:white; border: 1px solid #1e293b;">')
            html_parts.append('<thead>')
            html_parts.append('<tr style="background:#1e293b; color:#C9A84C; font-size:14px; text-align:left;">')
            html_parts.append('<th style="padding:10px;">STOCK</th>')
            html_parts.append('<th style="padding:10px;">STAGE</th>')
            html_parts.append('<th style="padding:10px;">TEMPLATE</th>')
            html_parts.append('</tr>')
            html_parts.append('</thead>')
            html_parts.append('<tbody>')
            
            for r in sa_list:
                sym = r['symbol']
                sym_display = sym.replace('.NS', '')
                stg = r['stage']
                tmpl = r['template_str']
                sc = r['score']
                
                # Colors based on Pine Script mapping
                if stg == 2:
                    stg_lbl = "STAGE 2 ▲"
                    stg_col = "#00FF00" # lime
                elif stg == 4:
                    stg_lbl = "STAGE 4 ▼"
                    stg_col = "#FF0000" # red
                elif stg == 3:
                    stg_lbl = "STAGE 3 ◆"
                    stg_col = "#FFA500" # orange
                else:
                    stg_lbl = "STAGE 1 ▬"
                    stg_col = "#C0C0C0" # silver
                    
                tmpl_col = "#00FF00" if sc >= 7 else ("#FFFF00" if sc >= 5 else "#808080")
                
                html_parts.append('<tr style="border-bottom:1px solid #1e293b;">')
                html_parts.append(f'<td style="padding:10px; font-weight:bold;"><a href="https://in.tradingview.com/chart/?symbol=NSE:{sym_display}" target="_blank" style="color:#ffffff; text-decoration:none;">{sym_display}</a></td>')
                html_parts.append(f'<td style="padding:10px; color:{stg_col}; font-weight:bold; font-size:13px;">{stg_lbl}</td>')
                html_parts.append(f'<td style="padding:10px; color:{tmpl_col}; font-weight:bold; font-size:13px;">{tmpl}</td>')
                html_parts.append('</tr>')
                
            html_parts.append("</tbody></table>")
            final_html = "".join(html_parts)
            st.markdown(final_html, unsafe_allow_html=True)
            
        else:
            st.info("✅ Scan completed — no setups found for the selected universe.")
    else:
        st.warning("⚠️ Scan has not been run yet. Click **'Run Stage Analysis Scan'** above to start.")
