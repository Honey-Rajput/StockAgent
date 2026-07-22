
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
from scan_orchestrator import process_single_symbol
from ui_components import (
    extract_clean_recommendation,
    matches_sma_timeframe_filter,
    render_quick_trade_board,
    render_trading_setup_card,
    render_unified_strategy_table,
)
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

    def scan_and_save(sym, df_hist):
        try:
            # Check if already scanned today to avoid redundant API queries
            if not force:
                existing = database.get_pattern_by_date(sym, date_str)
                if existing and existing.get('pattern_name') not in ["Error", "Pending"]:
                    return sym, True
                
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
            
        from local_cache_manager import bulk_get_cached_ohlcv
        bulk_cached = bulk_get_cached_ohlcv([s.strip().upper() for s in to_scan], "1d")
        
        def run_one(sym):
            df_hist = bulk_cached.get(sym.strip().upper())
            if df_hist is not None and not df_hist.empty:
                scan_and_save(sym, df_hist)
                
        max_workers = min(5, len(to_scan)) # Reduced from 20 to 5 for Streamlit Cloud memory limits
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(run_one, to_scan)
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
    page_title="VDU Breakout Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure database tables exist (fixes scan_logs SQLite missing table error on cloud)
import database
database.init_db()

# =============================================================================
# Custom CSS for UI Elegance & Readability
# =============================================================================
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



# --- Initialize Session State ---
import database
latest_dates = database.get_available_scan_dates()
latest_date_str = latest_dates[0] if latest_dates else None

# Clean up data older than 30 days automatically
try:
    database.cleanup_old_data(days=30)
except Exception as e:
    print(f"Cleanup error (non-fatal): {e}")

# First time app load check
is_startup = 'scan_results' not in st.session_state

if 'ema_support_results' not in st.session_state:
    st.session_state.ema_support_results = database.get_cached_ema_support(latest_date_str) if is_startup and latest_date_str else None
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = database.get_cached_breakouts(latest_date_str) if is_startup and latest_date_str else None
if 'total_scanned' not in st.session_state:
    log_entry = database.has_scanned_today(latest_date_str) if is_startup and latest_date_str else None
    st.session_state.total_scanned = log_entry.get('total_scanned', 0) if log_entry else 0
if 'failed_count' not in st.session_state:
    st.session_state.failed_count = 0
if 'last_scanned' not in st.session_state:
    st.session_state.last_scanned = latest_date_str if is_startup else None
if 'confirm_clear' not in st.session_state:
    st.session_state.confirm_clear = False
if 'ai_selected_stock' not in st.session_state:
    st.session_state.ai_selected_stock = ""
if 'ai_custom_sym_input' not in st.session_state:
    st.session_state.ai_custom_sym_input = ""
if 'vpa_results' not in st.session_state:
    st.session_state.vpa_results = database.get_cached_vpa(latest_date_str) if is_startup and latest_date_str else []
if 'vp_results' not in st.session_state:
    st.session_state.vp_results = database.get_cached_volume_profile(latest_date_str) if is_startup and latest_date_str else []
if 'gapup_results' not in st.session_state:
    st.session_state.gapup_results = database.get_cached_gapups(latest_date_str) if is_startup and latest_date_str else None
if 'above_ma_results' not in st.session_state:
    st.session_state.above_ma_results = database.get_cached_trend_setups(latest_date_str, 'above_ma') if is_startup and latest_date_str else None
if 'support_ma_results' not in st.session_state:
    st.session_state.support_ma_results = database.get_cached_trend_setups(latest_date_str, 'support_ma') if is_startup and latest_date_str else None
if 'crossover_ma_results' not in st.session_state:
    st.session_state.crossover_ma_results = database.get_cached_trend_setups(latest_date_str, 'crossover_ma') if is_startup and latest_date_str else None
if 'wt_results' not in st.session_state:
    st.session_state.wt_results = database.get_cached_wt_cross(latest_date_str) if is_startup and latest_date_str else None
if 'wt_results_by_tf' not in st.session_state:
    st.session_state.wt_results_by_tf = {}
if 'minervini_results' not in st.session_state:
    st.session_state.minervini_results = database.get_cached_trend_setups(latest_date_str, 'minervini') if is_startup and latest_date_str else None
if 'vcs_results' not in st.session_state:
    st.session_state.vcs_results = database.get_cached_vcs(latest_date_str) if is_startup and latest_date_str else None
if 'monthly_momentum_results' not in st.session_state:
    st.session_state.monthly_momentum_results = database.get_cached_monthly_momentum(latest_date_str) if is_startup and latest_date_str else None
if 'weekly_momentum_results' not in st.session_state:
    st.session_state.weekly_momentum_results = database.get_cached_weekly_momentum(latest_date_str) if is_startup and latest_date_str else None
if 'vpa_squeeze_results' not in st.session_state:
    st.session_state.vpa_squeeze_results = database.get_cached_vpa_squeeze(latest_date_str) if is_startup and latest_date_str else None
if 'near_30sma_results' not in st.session_state:
    st.session_state.near_30sma_results = database.get_cached_near_30sma(latest_date_str) if is_startup and latest_date_str else None
if 'near_30sma_weekly_results' not in st.session_state:
    st.session_state.near_30sma_weekly_results = database.get_cached_near_30sma_weekly(latest_date_str) if is_startup and latest_date_str else []
if 'near_30sma_monthly_results' not in st.session_state:
    st.session_state.near_30sma_monthly_results = database.get_cached_near_30sma_monthly(latest_date_str) if is_startup and latest_date_str else []
if 'dan_zanger_results' not in st.session_state:
    st.session_state.dan_zanger_results = database.get_cached_zanger(latest_date_str) if is_startup and latest_date_str else None
if 'vcp_minervini_results' not in st.session_state:
    st.session_state.vcp_minervini_results = database.get_cached_vcp_minervini(latest_date_str) if is_startup and latest_date_str else []
if 'stage2_results' not in st.session_state:
    st.session_state.stage2_results = database.get_cached_stage2(latest_date_str) if is_startup and latest_date_str else None
if 'support_rsi_results' not in st.session_state:
    st.session_state.support_rsi_results = database.get_cached_support_rsi(latest_date_str) if is_startup and latest_date_str else None
if 'stage_analysis_results' not in st.session_state:
    st.session_state.stage_analysis_results = database.get_cached_stage_analysis(latest_date_str) if is_startup and latest_date_str else None
# Initialize global status dictionary if not present (shared across all threads/sessions)
if "MOMENTUM_SCAN_STATUS" not in globals():
    # Removed redundant global statement
    MOMENTUM_SCAN_STATUS = {
        "is_running": False,
        "status_text": "Not started",
        "progress": 0.0,
        "monthly_results": None,
        "weekly_results": None
    }

# Initialize global status dictionary for ALL individual tab background scans
if "st.session_state.ALL_TAB_SCAN_STATUS" not in st.session_state:
    st.session_state.ALL_TAB_SCAN_STATUS = {
        "is_running": False,
        "current_scanner": "",
        "status_text": "Not started",
        "progress": 0.0,
        "wt_results": None,
        "vcs_results": None,
        "stage2_results": None,
        "vpa_results": None,
        "vp_results": None,
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
            today_str = get_market_date()
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
                    q_df = yf.download(tickers=chunk, period="1d", progress=False, threads=False, timeout=15)
                    if q_df is None or q_df.empty:
                        print("⚠️ Yahoo Finance Rate Limit hit. Aborting background scan chunk.")
                        break
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
            with _cf.ThreadPoolExecutor(max_workers=6) as pool:
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
                        df_mbulk = yf.download(tickers=chunk_ns, period="10y", interval="1mo", progress=False, threads=False, timeout=15)
                        if df_mbulk is None or df_mbulk.empty:
                            print("⚠️ Yahoo Finance Rate Limit hit. Aborting background monthly scan chunk.")
                            break
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
                                t_df_m['Date'] = pd.to_datetime(t_df_m['Date'], utc=True).dt.tz_localize(None)

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
                        df_wbulk = yf.download(tickers=chunk_ns, period="3y", interval="1wk", progress=False, threads=False, timeout=15)
                        if df_wbulk is None or df_wbulk.empty:
                            print("⚠️ Yahoo Finance Rate Limit hit. Aborting background weekly scan chunk.")
                            break
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
                                t_df_w['Date'] = pd.to_datetime(t_df_w['Date'], utc=True).dt.tz_localize(None)

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


def run_background_ema_support_scan(force=False):
    if st.session_state.ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
        return
        
    is_already_running = any(t.name == "Background_BB_Squeeze" for t in __import__('threading').enumerate())
    if is_already_running:
        return
        
    st.session_state.ALL_TAB_SCAN_STATUS["ema_support_running"] = True
    st.session_state.ema_support_running = True
    
    def thread_runner():
        import yfinance as yf
        import pandas as pd
        import concurrent.futures
        
        try:
            today_str = get_market_date()
            if not force:
                cached_bb = database.get_cached_ema_support(today_str)
                if cached_bb and len(cached_bb) > 0:
                    st.session_state.ALL_TAB_SCAN_STATUS["ema_support_results"] = cached_bb
                    st.session_state.ema_support_results = cached_bb
                    st.session_state.ALL_TAB_SCAN_STATUS["ema_support_running"] = False
                    st.session_state.ema_support_running = False
                    return
                
            from scanner import scan_ema_support
            raw_symbols = get_index_stocks("ALL NSE")
            symbols_to_scan = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
            from local_cache_manager import bulk_get_cached_ohlcv
            
            bb_results = []
            
            # Fetch all symbols from master cache at once
            bulk_cached = bulk_get_cached_ohlcv(symbols_to_scan, "1d")
            
            for sym_ns in symbols_to_scan:
                try:
                    sym = sym_ns.replace('.NS', '')
                    d_df = bulk_cached.get(sym)
                    
                    if d_df is None or d_df.empty:
                        continue
                        
                    res = scan_ema_support(sym, d_df)
                    if res:
                        bb_results.append(res)
                except Exception:
                    pass
                    
            st.session_state.ALL_TAB_SCAN_STATUS["ema_support_results"] = bb_results
            st.session_state.ema_support_results = bb_results
            try:
                database.save_ema_support_only(today_str, bb_results)
            except:
                pass
            
        except Exception as e:
            print(f"BB Squeeze background thread crashed: {e}")
        finally:
            st.session_state.ALL_TAB_SCAN_STATUS["ema_support_running"] = False
            st.session_state.ema_support_running = False
            
    import threading
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    t = threading.Thread(target=thread_runner, name="Background_BB_Squeeze", daemon=True)
    add_script_run_ctx(t)
    t.start()


def run_background_all_tab_scans():
    """
    Runs WaveTrend, VCS, Stage-2, VPA and Volume Profile scanners sequentially
    in a background daemon thread when Enable Auto-Background Scans is checked.
    Skips any scanner whose results are already cached in the database for today.
    """
    if st.session_state.ALL_TAB_SCAN_STATUS["is_running"]:
        return

    # Guard to prevent duplicate concurrent background scanning threads
    import threading
    is_already_running = any(t.name == "Background_All_Tab_Scans" for t in threading.enumerate())
    if is_already_running:
        print("Background all-tab scan thread is already active. Skipping duplicate thread launch.")
        return

    st.session_state.ALL_TAB_SCAN_STATUS["is_running"] = True
    st.session_state.ALL_TAB_SCAN_STATUS["status_text"] = "Initializing all-tab background scans..."
    st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 0.0

    def thread_runner():
        import yfinance as yf
        import pandas as pd
        import concurrent.futures
        import json
        import database
        from datetime import datetime

        try:
            today_str = get_market_date()
            print(f"[BG All-Tab] Starting background scans for date: {today_str}")

            from scanner import scan_wt_cross, scan_vcs, scan_monthly_early_stage2, scan_vpa_trend, scan_volume_profile, scan_vpa_ma_squeeze
            from data_fetcher import get_all_nse_symbols, get_index_stocks

            # Phase 1: Check what needs to be run
            run_wt = not bool(database.get_cached_wt_cross(today_str))
            run_vcs = not bool(database.get_cached_vcs(today_str))
            run_vpa = not bool(database.get_cached_vpa(today_str))
            run_vp = not bool(database.get_cached_volume_profile(today_str))
            run_s2 = not bool(database.get_cached_stage2(today_str))
            run_vpa_sq = not bool(database.get_cached_vpa_squeeze(today_str))
            run_near_30sma = not bool(database.get_cached_near_30sma(today_str))

            if not (run_wt or run_vcs or run_vpa or run_vp or run_s2 or run_vpa_sq or run_near_30sma):
                st.session_state.ALL_TAB_SCAN_STATUS["status_text"] = "All background tab scans already cached!"
                st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 1.0
                st.session_state.ALL_TAB_SCAN_STATUS["current_scanner"] = "Complete"
                st.session_state.ALL_TAB_SCAN_STATUS["is_running"] = False
                print("[BG All-Tab] All background tab scans already cached. Skipping.")
                return

            from data_fetcher import get_top1000_nse_symbols
            raw_symbols = get_top1000_nse_symbols()
            all_symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
            # Phase 2: Shared Daily Download (1y or 2y)
            shared_daily_data = {}
            if run_wt or run_vcs or run_vpa or run_vp or run_vpa_sq or run_near_30sma:
                st.session_state.ALL_TAB_SCAN_STATUS["current_scanner"] = "Downloading Shared Data"
                st.session_state.ALL_TAB_SCAN_STATUS["status_text"] = "Downloading shared daily data for NSE symbols..."
                st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 0.05
                print("[BG All-Tab] Downloading shared daily data...")
                
                # Use incremental fetching from data_fetcher for efficiency
                from data_fetcher import fetch_ohlcv
                import concurrent.futures
                
                processed_count = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(fetch_ohlcv, sym.replace('.NS', '')): sym.replace('.NS', '') for sym in all_symbols}
                    for future in concurrent.futures.as_completed(futures):
                        sym = futures[future]
                        processed_count += 1
                        if processed_count % 50 == 0:
                            st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 0.05 + (processed_count / len(all_symbols)) * 0.25
                        try:
                            df = future.result()
                            if df is not None and not df.empty and len(df) >= 40:
                                # Filter out penny stocks to save processing time
                                if float(df['Close'].iloc[-1]) >= 100.0:
                                    shared_daily_data[sym] = df
                        except Exception as e:
                            pass
                        
                print(f"[BG All-Tab] Shared data downloaded for {len(shared_daily_data)} stocks.")

            # Worker definitions
            def run_wt_worker(sym, df):
                res = scan_wt_cross(sym, df, wt_oversold_threshold=-40.0)
                if res:
                    res['timeframe'] = "Daily"
                    res['is_oversold'] = res['wt_value'] <= -40.0
                return ("wt", res)

            def run_vcs_worker(sym, df):
                if len(df) >= 63:
                    res = scan_vcs(sym, df)
                    if res:
                        res['Timeframe'] = "Daily (1d)"
                        res['Action'] = res.get('recommendation', 'Wait')
                        return ("vcs", res)
                return ("vcs", None)

            def run_vpa_worker(sym, df):
                if len(df) >= 60:
                    res = scan_vpa_trend(sym, df)
                    if res:
                        res['market_cap_cr'] = 0
                        return ("vpa", res)
                return ("vpa", None)

            def run_vp_worker(sym, df):
                if len(df) >= 120:
                    res = scan_volume_profile(sym, df, 0)
                    if res:
                        return ("vp", res)
                return ("vp", None)

            def run_vpa_sq_worker(sym, df):
                results = {"daily": None, "weekly": None, "monthly": None}
                if len(df) >= 50:
                    results["daily"] = scan_vpa_ma_squeeze(sym, df)
                    from local_cache_manager import resample_ohlcv
                    w_df = resample_ohlcv(df, "W")
                    if len(w_df) >= 50:
                        results["weekly"] = scan_vpa_ma_squeeze(sym, w_df)
                    m_df = resample_ohlcv(df, "M")
                    if len(m_df) >= 50:
                        results["monthly"] = scan_vpa_ma_squeeze(sym, m_df)
                return ("vpa_sq", results)

            def run_near_30sma_worker(sym, df):
                from scanner import scan_near_30sma
                from local_cache_manager import resample_ohlcv
                results = {"daily": None, "weekly": None, "monthly": None}
                results["daily"] = scan_near_30sma(sym, df)
                w_df = resample_ohlcv(df, "W")
                results["weekly"] = scan_near_30sma(sym, w_df)
                m_df = resample_ohlcv(df, "M")
                results["monthly"] = scan_near_30sma(sym, m_df)
                return ("near_30sma", results)

            # Phase 3: Parallel Execution
            wt_tf_results, custom_vcs_results, vpa_list, vp_list, vpa_sq_list, near_30sma_list = [], [], [], [], [], []
            vpa_sq_weekly_list, vpa_sq_monthly_list = [], []
            near_30sma_weekly_list = []
            near_30sma_monthly_list = []
            
            if shared_daily_data:
                st.session_state.ALL_TAB_SCAN_STATUS["current_scanner"] = "Executing Scans"
                st.session_state.ALL_TAB_SCAN_STATUS["status_text"] = "Running concurrent daily scans..."
                
                tasks_to_run = []
                for sym, df in shared_daily_data.items():
                    if run_wt: tasks_to_run.append((run_wt_worker, sym, df))
                    if run_vcs: tasks_to_run.append((run_vcs_worker, sym, df))
                    if run_vpa: tasks_to_run.append((run_vpa_worker, sym, df))
                    if run_vp: tasks_to_run.append((run_vp_worker, sym, df))
                    if run_vpa_sq: tasks_to_run.append((run_vpa_sq_worker, sym, df))
                    if run_near_30sma: tasks_to_run.append((run_near_30sma_worker, sym, df))
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(func, sym, df) for func, sym, df in tasks_to_run]
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        if i % 100 == 0:
                            st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 0.30 + (i / len(tasks_to_run)) * 0.45
                        try:
                            scan_type, result = future.result()
                            if result:
                                if scan_type == "wt": wt_tf_results.append(result)
                                elif scan_type == "vcs": custom_vcs_results.append(result)
                                elif scan_type == "vpa": vpa_list.append(result)
                                elif scan_type == "vp": vp_list.append(result)
                                elif scan_type == "vpa_sq":
                                    if result["daily"]: vpa_sq_list.append(result["daily"])
                                    if result["weekly"]: vpa_sq_weekly_list.append(result["weekly"])
                                    if result["monthly"]: vpa_sq_monthly_list.append(result["monthly"])
                                elif scan_type == "near_30sma":
                                    if result["daily"]: near_30sma_list.append(result["daily"])
                                    if result["weekly"]: near_30sma_weekly_list.append(result["weekly"])
                                    if result["monthly"]: near_30sma_monthly_list.append(result["monthly"])
                        except Exception:
                            pass
                
                # Save results
                if run_wt:
                    st.session_state.ALL_TAB_SCAN_STATUS["wt_results"] = wt_tf_results
                    try: database.save_wt_cross_only(today_str, wt_tf_results)
                    except: pass
                if run_vcs:
                    st.session_state.ALL_TAB_SCAN_STATUS["vcs_results"] = custom_vcs_results
                    try: database.save_vcs_only(today_str, custom_vcs_results)
                    except: pass
                if run_vpa:
                    st.session_state.ALL_TAB_SCAN_STATUS["vpa_results"] = vpa_list
                    try: database.save_vpa_only(today_str, vpa_list)
                    except: pass
                if run_vp:
                    st.session_state.ALL_TAB_SCAN_STATUS["volume_profile_results"] = vp_list
                    try: database.save_volume_profile_only(today_str, vp_list)
                    except: pass
                if run_vpa_sq:
                    st.session_state.ALL_TAB_SCAN_STATUS["vpa_squeeze_results"] = vpa_sq_list
                    try:
                        database.save_vpa_squeeze_only(today_str, vpa_sq_list)
                        database.save_vpa_squeeze_weekly_only(today_str, vpa_sq_weekly_list)
                        database.save_vpa_squeeze_monthly_only(today_str, vpa_sq_monthly_list)
                    except Exception as e: print(f"Save VPA SQ failed: {e}")
                if run_near_30sma:
                    st.session_state.ALL_TAB_SCAN_STATUS["near_30sma_results"] = near_30sma_list
                    try:
                        database.save_near_30sma_only(today_str, near_30sma_list)
                        database.save_near_30sma_weekly_only(today_str, near_30sma_weekly_list)
                        database.save_near_30sma_monthly_only(today_str, near_30sma_monthly_list)
                    except Exception as e: print(f"Save Near 30SMA failed: {e}")

            # Phase 4: Stage-2 (Monthly)
            if run_s2:
                st.session_state.ALL_TAB_SCAN_STATUS["current_scanner"] = "Stage-2"
                st.session_state.ALL_TAB_SCAN_STATUS["status_text"] = "Running Stage-2 monthly scan..."
                st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 0.75
                from data_fetcher import get_top1000_nse_symbols
                s2_cands = get_top1000_nse_symbols()
                s2_res = []
                
                from local_cache_manager import bulk_get_cached_ohlcv, resample_ohlcv
                s2_bulk = bulk_get_cached_ohlcv([s.strip().upper() for s in s2_cands], "1d")
                
                for c_idx, (sym, t_df) in enumerate(s2_bulk.items()):
                    if c_idx % 20 == 0:
                        st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 0.75 + (c_idx / max(len(s2_bulk), 1)) * 0.20
                        
                    if t_df is None or t_df.empty:
                        continue
                        
                    try:
                        m_df = resample_ohlcv(t_df, 'ME')
                        if not m_df.empty and len(m_df) >= 24:
                            res = scan_monthly_early_stage2(sym, m_df, max_run_up_pct=20.0)
                            if res: s2_res.append(res)
                    except Exception: pass
                    
                s2_res = sorted(s2_res, key=lambda x: x.get('score', 0), reverse=True)
                st.session_state.ALL_TAB_SCAN_STATUS["stage2_results"] = s2_res
                try: database.save_stage2_only(today_str, s2_res)
                except Exception: pass

            st.session_state.ALL_TAB_SCAN_STATUS["status_text"] = "All background tab scans complete!"
            st.session_state.ALL_TAB_SCAN_STATUS["progress"] = 1.0
            st.session_state.ALL_TAB_SCAN_STATUS["current_scanner"] = "Complete"
            st.session_state.ALL_TAB_SCAN_STATUS["is_running"] = False
            print("[BG All-Tab] All background tab scans complete!")

        except Exception as err:
            st.session_state.ALL_TAB_SCAN_STATUS["status_text"] = f"Background scan error: {err}"
            st.session_state.ALL_TAB_SCAN_STATUS["is_running"] = False
            print(f"[BG All-Tab] Fatal error: {err}")

    t = threading.Thread(target=thread_runner, name="Background_All_Tab_Scans", daemon=True)
    t.start()


if 'monthly_momentum_results' not in st.session_state:
    st.session_state.monthly_momentum_results = None
if 'weekly_momentum_results' not in st.session_state:
    st.session_state.weekly_momentum_results = None

today_str_check = get_market_date(for_display=True)

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


# --- Authentication System ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    with st.sidebar.expander("🔐 Admin Login", expanded=False):
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if user == "admin" and pwd == "vdu123":
                    st.session_state['authenticated'] = True
                    st.toast("Logged in successfully!", icon="✅")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
else:
    if st.sidebar.button("🔓 Logout", use_container_width=True):
        st.session_state['authenticated'] = False
        st.rerun()

is_admin = st.session_state['authenticated']
# -----------------------------

st.sidebar.markdown('### ⚡ Performance Settings')
enable_background_scans = st.sidebar.checkbox("Enable Auto-Background Scans", value=False, help="Disable this on Streamlit Cloud to prevent UI freezing due to heavy thread execution.")

# Automatically trigger scanning in background if results are missing for today
if enable_background_scans:
    if (st.session_state.monthly_momentum_results is None or st.session_state.weekly_momentum_results is None) and not MOMENTUM_SCAN_STATUS["is_running"]:
        run_background_momentum_scans()
    # Auto-trigger all remaining tab scans (WaveTrend, VCS, Stage-2, VPA, Volume Profile)
    if not st.session_state.ALL_TAB_SCAN_STATUS["is_running"]:
        run_background_all_tab_scans()
    # Auto-trigger BB Squeeze
    if not st.session_state.ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
        run_background_ema_support_scan()

# --- Automatic Daily Database Cache Loader ---
# Runs ONCE per browser session (db_cache_checked stays False until first load).
# Each scanner loads independently from its own DB table/date — NOT gated behind scan_logs.
if not st.session_state.get('db_cache_checked', False):
    st.session_state['db_cache_checked'] = True
    try:
        # Fetch the max dates for ALL tables at once
        all_latest_dates = database.get_all_latest_scan_dates()
        best_scan_date = database.get_best_scan_date()

        def _load_latest(table, getter_fn, state_key, post_fn=None):
            """Helper: find own latest date for a table, then load and set session state.
            Returns the date string the data was loaded from (or None)."""
            try:
                # Prefer best_scan_date for breakouts to ensure we don't load a partial scan
                d = best_scan_date if table == "scanned_breakouts" and best_scan_date else all_latest_dates.get(table)
                
                if d:
                    data = getter_fn(d)
                    if data:
                        st.session_state[state_key] = post_fn(data) if post_fn else data
                    else:
                        st.session_state[state_key] = []
                    return d  # Return the date loaded
                else:
                    st.session_state[state_key] = []
                    return None
            except Exception as _e:
                print(f"Error loading {state_key} from {table}: {_e}")
                st.session_state[state_key] = []
                return None

        # ── Independent loaders: each scanner uses its OWN table's latest date ──────
        breakouts_date = _load_latest("scanned_breakouts", database.get_cached_breakouts, "scan_results")
        # Store the breakouts date so Results tab can show "data from" and correctly look up total_scanned
        st.session_state['scan_results_date'] = breakouts_date
        _load_latest("scanned_gapups", database.get_cached_gapups, "gapup_results")
        _load_latest("scanned_trend_setups", lambda d: database.get_cached_trend_setups(d, 'above_ma'), "above_ma_results")
        _load_latest("scanned_trend_setups", lambda d: database.get_cached_trend_setups(d, 'support_ma'), "support_ma_results")
        _load_latest("scanned_trend_setups", lambda d: database.get_cached_trend_setups(d, 'crossover_ma'), "crossover_ma_results")
        _load_latest("scanned_trend_setups", lambda d: database.get_cached_trend_setups(d, 'minervini'), "minervini_results", ensure_minervini_fields)
        _load_latest("scanned_stage_analysis", database.get_cached_stage_analysis, "stage_analysis_results")

        _load_latest("scanned_wt_cross", database.get_cached_wt_cross, "wt_results")
        if st.session_state.get("wt_results"):
            st.session_state.wt_results_by_tf = {"Daily_-40.0": st.session_state.wt_results, "Daily": st.session_state.wt_results}

        _load_latest("scanned_vcs", database.get_cached_vcs, "vcs_results")
        _load_latest("scanned_vpa", database.get_cached_vpa, "vpa_results")
        _load_latest("scanned_volume_profile", database.get_cached_volume_profile, "vp_results")
        _load_latest("scanned_ema_support", database.get_cached_ema_support, "ema_support_results")
        _load_latest("scanned_vpa_squeeze", database.get_cached_vpa_squeeze, "vpa_squeeze_results")
        _load_latest("scanned_near_30sma", database.get_cached_near_30sma, "near_30sma_results")  # FIX: was missing on reload
        _load_latest("scanned_stage2", database.get_cached_stage2, "stage2_results")
        _load_latest("scanned_support_rsi", database.get_cached_support_rsi, "support_rsi_results")
        _load_latest("scanned_monthly_momentum", database.get_cached_monthly_momentum, "monthly_momentum_results")
        _load_latest("scanned_weekly_momentum", database.get_cached_weekly_momentum, "weekly_momentum_results")

        # ── Dan Zanger and VCP+Minervini: auto-load from their own DB tables ─────────
        _load_latest("scanned_zanger", lambda d: database.get_cached_zanger(d, 'Daily'), "zanger_results")

        def _load_vcp_from_db(data):
            """Reconstruct Rank and Score columns for VCP data loaded from DB cache."""
            import pandas as pd
            if not data:
                return data
            vcp_df = pd.DataFrame(data)
            rs_proxy = pd.to_numeric(vcp_df.get('RS Proxy', 50), errors='coerce').fillna(50)
            vcp_range = pd.to_numeric(vcp_df.get('VCP range %', 100), errors='coerce').fillna(100)
            vcp_df['Score'] = rs_proxy - (vcp_range * 5)
            vcp_df = vcp_df.sort_values(by='Score', ascending=False)
            vcp_df.insert(0, 'Rank', range(1, len(vcp_df) + 1))
            return vcp_df.to_dict('records')
        _load_latest("scanned_vcp_minervini", database.get_cached_vcp_minervini, "vcp_minervini_results", _load_vcp_from_db)

        # Load scan_logs for UI metadata
        # IMPORTANT: Use the SAME date as the loaded breakouts (not scan_logs latest date)
        # to avoid a mismatch where total_scanned is from a different scan than the results.
        breakouts_scan_date = st.session_state.get('scan_results_date')
        if breakouts_scan_date:
            # First try to get total_scanned from the same date as the breakout data
            cached_log = database.has_scanned_today(breakouts_scan_date) or {}
            loaded_total = cached_log.get('total_scanned', 0)
            # If scan_logs doesn't have this date, derive from actual breakout data
            st.session_state.total_scanned = loaded_total or len(st.session_state.scan_results or [])
            st.session_state.failed_count = 0
            st.session_state.last_scanned = breakouts_scan_date + " (Loaded from DB Cache)"
        else:
            available_dates = database.get_available_scan_dates()
            if available_dates:
                latest_date_str = available_dates[0]
                cached_log = database.has_scanned_today(latest_date_str) or {}
                st.session_state.total_scanned = cached_log.get('total_scanned', 0)
                st.session_state.failed_count = 0
                st.session_state.last_scanned = latest_date_str + " (Loaded from DB Cache)"
            else:
                st.session_state.total_scanned = 0
                st.session_state.failed_count = 0
                st.session_state.last_scanned = "Never"

        # NOTE: Auto background AI scan on boot is DISABLED.
        # It was triggering Groq API calls for all 1000 stocks on every page refresh,
        # burning through API quota and significantly slowing down the app.
        # AI scans now only run from the AI Pattern tab (on demand).


    except Exception as cache_err:
        print(f"Error loading daily database scan cache on boot: {cache_err}")


# --- HEADER SECTION ---
st.markdown('<h1 class="gradient-title">📈 Volume Surge Scanner</h1>', unsafe_allow_html=True)
st.markdown('<p class="gradient-subtitle">Scan NSE-listed stocks for institutional Volume Dry-Up (VDU) breakouts & build a high-conviction swing trading watchlist.</p>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown('### ⚙️ Scan Universe')
universe_selection = "Top 1000 NSE Stocks (By Market Cap)"
st.sidebar.info("🔍 **Scan Universe:** Top 1000 NSE Stocks (By Market Cap)")

# =============================================================================
# MARKET CONDITION WIDGET — Nifty 50 Breadth Filter
# Shows live market health so you know whether it's safe to buy breakouts.
# =============================================================================
st.sidebar.markdown('---')

with st.sidebar.expander("📈 20/50 SMA Multi-Timeframe Strategy Settings", expanded=False):
    sma20_lower_bound = st.slider("SMA 20 Lower Bound", min_value=0.85, max_value=1.00, value=0.94, step=0.01, key="sma20_lower_bound")
    sma20_upper_bound = st.slider("SMA 20 Upper Bound", min_value=1.00, max_value=1.15, value=1.06, step=0.01, key="sma20_upper_bound")
    sma50_lower_bound = st.slider("SMA 50 Lower Bound", min_value=0.85, max_value=1.00, value=0.92, step=0.01, key="sma50_lower_bound")
    sma50_upper_bound = st.slider("SMA 50 Upper Bound", min_value=1.00, max_value=1.15, value=1.08, step=0.01, key="sma50_upper_bound")
    sma20_min_volume = st.number_input("Min Volume SMA 20", min_value=10000, max_value=10000000, value=100000, step=10000, key="sma20_min_volume")

st.sidebar.markdown('---')
st.sidebar.markdown('### 🌐 Market Condition')
try:
    _mc = get_market_condition()
    _mc_status  = _mc.get('status', 'Unknown')
    _mc_emoji   = _mc.get('emoji', '⚪')
    _mc_cmp     = _mc.get('cmp', 0.0)
    _mc_chg     = _mc.get('change_pct', 0.0)
    _mc_sma50   = _mc.get('sma50', 0.0)
    _mc_sma200  = _mc.get('sma200', 0.0)
    _chg_color  = '#00e676' if _mc_chg >= 0 else '#ef4444'
    _card_color = 'rgba(0,230,118,0.06)'  if _mc_status == 'Bullish' else \
                  'rgba(255,160,0,0.06)'   if _mc_status == 'Caution' else \
                  'rgba(239,68,68,0.06)'
    _border_color = 'rgba(0,230,118,0.35)'  if _mc_status == 'Bullish' else \
                    'rgba(255,160,0,0.35)'   if _mc_status == 'Caution' else \
                    'rgba(239,68,68,0.35)'
    st.sidebar.markdown(
        f"""
        <div style='padding:10px 14px; background:{_card_color}; border:1px solid {_border_color};
                    border-radius:10px; margin-bottom:12px;'>
          <div style='font-size:1rem; font-weight:700; color:#e2e8f0;'>
            {_mc_emoji} Nifty 50 — <span style='color:{_border_color.replace('0.35','1')};'>{_mc_status}</span>
          </div>
          <div style='font-size:0.85rem; color:#94a3b8; margin-top:4px;'>
            CMP: <b style='color:#e2e8f0;'>₹{_mc_cmp:,.2f}</b>
            &nbsp;<span style='color:{_chg_color};'>({'▲' if _mc_chg>=0 else '▼'} {abs(_mc_chg):.2f}%)</span>
          </div>
          <div style='font-size:0.78rem; color:#64748b; margin-top:2px;'>
            50-SMA: ₹{_mc_sma50:,.0f} &nbsp;|&nbsp; 200-SMA: ₹{_mc_sma200:,.0f}
          </div>
          {'<div style="font-size:0.75rem; color:#ffa000; margin-top:5px;">⚠️ Market in caution zone — be selective with new buys</div>' if _mc_status == 'Caution' else ''}
          {'<div style="font-size:0.75rem; color:#ef4444; margin-top:5px;">🚨 Bear market — avoid new breakout trades, prioritize capital preservation</div>' if _mc_status == 'Bearish' else ''}
        </div>
        """,
        unsafe_allow_html=True
    )
except Exception as _mc_err:
    st.sidebar.caption(f"⚪ Market condition unavailable ({_mc_err})")

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
    value=2.5,
    step=0.5,
    key="vdu_min_vol_ratio_v8",
    help="Breakout day volume compared to dry average volume (e.g., 2.0 = 2x surge)"
)

min_price_chg = st.sidebar.slider(
    "Min Price Change %",
    min_value=1.5,
    max_value=30.0,
    value=7.0,
    step=0.5,
    key="vdu_min_price_chg_v8",
    help="Minimum price percentage increase on the breakout day (Close vs Open)"
)

dry_zone_range = st.sidebar.slider(
    "Dry Zone Range (Trading Days)",
    min_value=0,
    max_value=150,
    value=(0, 50),
    step=5,
    key="vdu_dry_zone_range_v5",
    help="Configure the minimum and maximum duration of the dry zone consolidation period (up to 150 days)"
)

min_dry_spikes = st.sidebar.slider(
    "Min Spikes in Dry Zone",
    min_value=0,
    max_value=20,
    value=7,
    step=1,
    key="vdu_min_dry_spikes_v8",
    help="Requires at least this many volume accumulation spikes inside the dry zone window (up to 20 spikes)"
)

min_signal_str = st.sidebar.slider(
    "Min Signal Strength Score",
    min_value=0,
    max_value=100,
    value=30,
    step=5,
    key="vdu_min_signal_str_v5",
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
run_full = st.sidebar.button("🔍 Run Full Scanner", use_container_width=True)

col1, col2 = st.sidebar.columns(2)
if col1.button("💾 Save Results", help="Force save all current results to the DB for today.", disabled=not is_admin):
    import database
    from datetime import datetime
    import pytz
    ist = pytz.timezone('Asia/Kolkata')
    today_str = datetime.now(ist).strftime("%Y-%m-%d")
    
    with st.spinner("Saving all results to database..."):
        # Gather data from session state
        breakouts = st.session_state.get('scan_results', []) or []
        squeezes = st.session_state.get('vpa_squeeze_results', []) or []
        gapups = st.session_state.get('gapup_results', []) or []
        trend = []
        trend.extend(st.session_state.get('above_ma_results', []) or [])
        trend.extend(st.session_state.get('support_ma_results', []) or [])
        trend.extend(st.session_state.get('crossover_ma_results', []) or [])
        trend.extend(st.session_state.get('minervini_results', []) or [])
        wt = st.session_state.get('wt_results', []) or []
        tot = st.session_state.get('total_scanned', 0)
        vcs = st.session_state.get('vcs_results', []) or []
        vpa = st.session_state.get('vpa_results', []) or []
        near30 = st.session_state.get('near_30sma_results', []) or []
        
        # Save master results
        database.save_scan_results(today_str, breakouts, squeezes, gapups, trend, wt, tot, vcs_results=vcs, vpa_results=vpa, near_30sma_list=near30)
        
        # Save other specific results
        if st.session_state.get('monthly_momentum_results'): database.save_monthly_momentum_results(today_str, st.session_state['monthly_momentum_results'])
        if st.session_state.get('weekly_momentum_results'): database.save_weekly_momentum_results(today_str, st.session_state['weekly_momentum_results'])
        if st.session_state.get('stage2_results'): database.save_stage2_only(today_str, st.session_state['stage2_results'])
        if st.session_state.get('support_rsi_results'): database.save_support_rsi_only(today_str, st.session_state['support_rsi_results'])
        if st.session_state.get('stage_analysis_results'): database.save_stage_analysis_only(today_str, st.session_state['stage_analysis_results'])
        if st.session_state.get('ema_support_results'): database.save_ema_support_only(today_str, st.session_state['ema_support_results'])
        if st.session_state.get('zanger_results'): database.save_zanger_scan(today_str, "1d", st.session_state['zanger_results'])
        if st.session_state.get('vcp_minervini_results'): database.save_vcp_minervini_scan(today_str, st.session_state['vcp_minervini_results'])
        if st.session_state.get('vp_results'): database.save_volume_profile_only(today_str, st.session_state['vp_results'])
        
        # Multi-timeframe tables
        if st.session_state.get('near_30sma_weekly_results'): database.save_near_30sma_weekly_only(today_str, st.session_state['near_30sma_weekly_results'])
        if st.session_state.get('near_30sma_monthly_results'): database.save_near_30sma_monthly_only(today_str, st.session_state['near_30sma_monthly_results'])
        if st.session_state.get('vpa_squeeze_weekly_results'): database.save_vpa_squeeze_weekly_only(today_str, st.session_state['vpa_squeeze_weekly_results'])
        if st.session_state.get('vpa_squeeze_monthly_results'): database.save_vpa_squeeze_monthly_only(today_str, st.session_state['vpa_squeeze_monthly_results'])
        
    st.sidebar.success(f"Saved results for {today_str}!")

if col2.button("📥 Fetch Latest", help="Fetch the latest saved results from DB."):
    import database
    latest_dates = database.get_available_scan_dates()
    latest_date_str = latest_dates[0] if latest_dates else None
    
    if latest_date_str:
        with st.spinner(f"Loading data for {latest_date_str}..."):
            # Manually load all data from database to session state
            st.session_state.scan_results = database.get_cached_breakouts(latest_date_str)
            st.session_state.vpa_squeeze_results = database.get_cached_vpa_squeeze(latest_date_str)
            st.session_state.gapup_results = database.get_cached_gapups(latest_date_str)
            st.session_state.above_ma_results = database.get_cached_trend_setups(latest_date_str, 'above_ma')
            st.session_state.support_ma_results = database.get_cached_trend_setups(latest_date_str, 'support_ma')
            st.session_state.crossover_ma_results = database.get_cached_trend_setups(latest_date_str, 'crossover_ma')
            st.session_state.minervini_results = database.get_cached_trend_setups(latest_date_str, 'minervini')
            st.session_state.wt_results = database.get_cached_wt_cross(latest_date_str)
            st.session_state.vcs_results = database.get_cached_vcs(latest_date_str)
            st.session_state.vpa_results = database.get_cached_vpa(latest_date_str)
            
            st.session_state.monthly_momentum_results = database.get_cached_monthly_momentum(latest_date_str)
            st.session_state.weekly_momentum_results = database.get_cached_weekly_momentum(latest_date_str)
            st.session_state.stage2_results = database.get_cached_stage2(latest_date_str)
            st.session_state.near_30sma_results = database.get_cached_near_30sma(latest_date_str)
            st.session_state.near_30sma_weekly_results = database.get_cached_near_30sma_weekly(latest_date_str)
            st.session_state.near_30sma_monthly_results = database.get_cached_near_30sma_monthly(latest_date_str)
            st.session_state.support_rsi_results = database.get_cached_support_rsi(latest_date_str)
            st.session_state.stage_analysis_results = database.get_cached_stage_analysis(latest_date_str)
            st.session_state.ema_support_results = database.get_cached_ema_support(latest_date_str)
            st.session_state.zanger_results = database.get_cached_zanger(latest_date_str)
            st.session_state.vcp_minervini_results = database.get_cached_vcp_minervini(latest_date_str)
            st.session_state.vp_results = database.get_cached_volume_profile(latest_date_str)
            
            st.session_state.near_30sma_results = database.get_cached_near_30sma(latest_date_str) or []
            st.session_state.near_30sma_weekly_results = database.get_cached_near_30sma_weekly(latest_date_str) or []
            st.session_state.near_30sma_monthly_results = database.get_cached_near_30sma_monthly(latest_date_str) or []
            
            # Explicitly load missing multi-timeframe caches that aren't in load_previous_scan_results
            st.session_state.near_30sma_results = database.get_cached_near_30sma(latest_date_str) or []
            st.session_state.near_30sma_weekly_results = database.get_cached_near_30sma_weekly(latest_date_str)
            st.session_state.near_30sma_monthly_results = database.get_cached_near_30sma_monthly(latest_date_str)
            try:
                st.session_state.vpa_squeeze_weekly_results = database.get_cached_vpa_squeeze_weekly(latest_date_str)
                st.session_state.vpa_squeeze_monthly_results = database.get_cached_vpa_squeeze_monthly(latest_date_str)
            except Exception:
                pass
                
            st.session_state.scan_executed = True
            
            # Re-read total scanned securely
            log_entry = database.has_scanned_today(latest_date_str)
            st.session_state.total_scanned = log_entry.get('total_scanned', 0) if log_entry else 0
            
        st.sidebar.success(f"Loaded {latest_date_str}!")
        st.rerun()
    else:
        st.sidebar.warning("No saved data found.")
run_sma = False

if run_full or run_sma:
    scan_mode_flag = "sma_only" if run_sma else "full"
    
    if "1wk" in scan_timeframe:
        yf_period = "4y"
        yf_interval = "1wk"
    elif "1mo" in scan_timeframe:
        yf_period = "10y"
        yf_interval = "1mo"
    else:
        yf_period = "2y"
        yf_interval = "1d"

    # Universe is hardcoded to Top 1000 NSE stocks as per user preference
    universe_key = "TOP 1000"
    from data_fetcher import get_top1000_nse_symbols
    raw_symbols = get_top1000_nse_symbols()
        
    if not raw_symbols:
        st.sidebar.error("❌ No symbols found to scan.")
    else:
      try:
        with st.spinner("🚀 Running Full Scanner (1-2 mins)... Please wait!"):
            # UI Scanner Feedback
            status_box = st.empty()
            prog_bar = st.progress(0)
            
            # Known-delisted / permanently broken symbols — skip entirely to avoid wasted retries
            BLACKLISTED_SYMBOLS = {"GANGAFO-RE", "AMIRCHAND"}
            raw_symbols = [s for s in raw_symbols if s.strip().upper() not in BLACKLISTED_SYMBOLS]

            all_tickers_ns = []
            for s in raw_symbols:
                formatted = s.strip().upper()
                if not formatted.endswith(".NS"):
                    formatted = f"{formatted}.NS"
                all_tickers_ns.append(formatted)
                
            today_date_str = get_market_date()
            # NOTE: We do NOT use st.session_state cache for Phase 1 quotes.
            # Session state caches stale partial data (e.g. 201 stocks from a broken previous run).
            # The database is the single source of truth — it supports incremental saves.
            open_price_map = {}
            close_price_map = {}
            volume_map = {}
            high_price_map = {}
            low_price_map = {}

            # ── Smart Phase 1: Check Local DB first ─────────────────────────────
            _db_quotes = {}
            # Always check if today's data is already in the database to avoid redundant downloads
            status_box.text("Phase 1/3: Checking Local Database for today's cached quotes...")
            try:
                import concurrent.futures as _p1_cf
                with _p1_cf.ThreadPoolExecutor(max_workers=1) as _p1_tex:
                    _fut = _p1_tex.submit(database.get_today_quotes, raw_symbols, today_date_str)
                    try:
                        _db_quotes = _fut.result(timeout=10)  # 10s timeout
                    except _p1_cf.TimeoutError:
                        print("Phase 1 DB check timed out after 10s — falling back to Yahoo")
                        _db_quotes = {}
            except Exception as _dq_err:
                print(f"Phase 1 DB check error: {_dq_err}")
                _db_quotes = {}

            # ✅ Always load whatever we have in the local DB to save time!
            for _sym, _q in _db_quotes.items():
                if _q["close"] > 0:
                    close_price_map[_sym]  = _q["close"]
                    open_price_map[_sym]   = _q["open"]
                    high_price_map[_sym]   = _q["high"]
                    low_price_map[_sym]    = _q["low"]
                    volume_map[_sym]       = _q["volume"]

            # Find which symbols are STILL missing
            missing_ns = []
            for s in raw_symbols:
                clean_s = s.strip().upper()
                if clean_s not in close_price_map:
                    missing_ns.append(f"{clean_s}.NS")

            if len(missing_ns) == 0:
                status_box.text(f"Phase 1/3: ✅ Loaded {len(close_price_map)} quotes from Local Database (skipped Yahoo Finance!)")
                prog_bar.progress(1.0)
            else:
                # ⬇️ Download from Yahoo Finance ONLY for the missing symbols
                status_box.text(f"Phase 1/3: Downloading {len(missing_ns)} missing quotes from Yahoo...")
                import time
                chunk_size = 35  # 35 tickers per chunk
                ticker_chunks = [missing_ns[i:i + chunk_size] for i in range(0, len(missing_ns), chunk_size)]

                # Thread-safe accumulators for parallel quote downloads
                import threading as _p1_threading
                _p1_lock = _p1_threading.Lock()

                def _download_quote_chunk(idx_chunk_pair):
                    idx, chunk = idx_chunk_pair
                    _open = {}; _close = {}; _vol = {}; _high = {}; _low = {}
                    retries = 0
                    max_retries = 5
                    backoff = 3.0
                    while retries <= max_retries:
                        try:
                            # yfinance 1.x: auto_adjust=True by default, threads param removed
                            quotes_df = yf.download(tickers=chunk, period="1d", progress=False, threads=False, timeout=15)
                            if not quotes_df.empty:
                                # yfinance 1.x multi-ticker: MultiIndex (price_type, ticker)
                                if isinstance(quotes_df.columns, pd.MultiIndex):
                                    # Level 0 = price type (Close/Open/etc), Level 1 = ticker symbol
                                    price_types = quotes_df.columns.get_level_values(0).unique().tolist()
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
                                for k, v in close_series.items():
                                    clean_k = str(k).replace(".NS", "").upper()
                                    if not pd.isna(v) and float(v) > 0:
                                        _close[clean_k] = float(v)
                                        if k in open_series.index and not pd.isna(open_series[k]):
                                            _open[clean_k] = float(open_series[k])
                                        if k in volume_series.index and not pd.isna(volume_series[k]):
                                            _vol[clean_k] = int(volume_series[k])
                                        if k in high_series.index and not pd.isna(high_series[k]):
                                            _high[clean_k] = float(high_series[k])
                                        if k in low_series.index and not pd.isna(low_series[k]):
                                            _low[clean_k] = float(low_series[k])
                                # Successfully loaded chunk, add a tiny delay to respect rate limits
                                time.sleep(1.5)
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

                # Use a single worker to prevent aggressive yfinance rate-limiting
                p1_workers = 1 if len(ticker_chunks) > 10 else min(3, len(ticker_chunks))
                with concurrent.futures.ThreadPoolExecutor(max_workers=p1_workers) as p1_executor:
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

                        # ✅ Save this chunk to DB immediately (per-chunk progressive save)
                        if _c:
                            _chunk_c = dict(_c); _chunk_o = dict(_o)
                            _chunk_h = dict(_h); _chunk_l = dict(_l); _chunk_v = dict(_v)
                            _d_str = today_date_str
                            def _save_chunk_now(cc, co, ch, cl, cv, ds):
                                try:
                                    database.save_today_quotes(ds, cc, co, ch, cl, cv)
                                except Exception as _sce:
                                    print(f"Chunk {i+1} save error: {_sce}")
                            import threading as _p1_save_t
                            _p1_save_t.Thread(
                                target=_save_chunk_now,
                                args=(_chunk_c, _chunk_o, _chunk_h, _chunk_l, _chunk_v, _d_str),
                                daemon=True
                            ).start()

            # Fast filter Price > 0 (removes completely dead/invalid symbols)
            from local_cache_manager import get_cached_ohlcv
            scan_symbols = []
            for s in raw_symbols:
                clean_s = s.strip().upper()
                if close_price_map.get(clean_s, 0.0) > 0.0:
                    scan_symbols.append(s)
                else:
                    # Fallback to local cache if Phase 1 failed (e.g. rate limit)
                    cached_df = get_cached_ohlcv(clean_s, scan_timeframe, ignore_ttl=True)
                    if cached_df is not None and not cached_df.empty:
                        last_row = cached_df.iloc[-1]
                        close_price_map[clean_s] = float(last_row['Close'])
                        if 'Open' in last_row: open_price_map[clean_s] = float(last_row['Open'])
                        if 'Volume' in last_row: volume_map[clean_s] = float(last_row['Volume'])
                        if 'High' in last_row: high_price_map[clean_s] = float(last_row['High'])
                        if 'Low' in last_row: low_price_map[clean_s] = float(last_row['Low'])
                        scan_symbols.append(s)
            
            n_stocks = len(scan_symbols)
            if n_stocks == 0:
                status_box.error("❌ Failed to fetch live prices (Rate Limited by Yahoo Finance). Please try again in 5 minutes.")
                prog_bar.progress(1.0)
                st.stop()
                
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
            vpa_squeeze_list = []
            zanger_list = []
            vp_list = []
            support_rsi_list = []
            ema_support_list = []
            stage_analysis_list = []
            stage2_list = []
            monthly_momentum_list = []
            weekly_momentum_list = []
            near_30sma_list = []
            near_30sma_weekly_list = []
            near_30sma_monthly_list = []
            
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
    
                    status_box.text(f"Phase 2/3: Checking local cache for {scan_timeframe} historical data...")
                    prog_bar.progress(0)
                    
                    from local_cache_manager import get_cached_ohlcv, save_to_cache, bulk_get_cached_ohlcv
                    missing_symbols = []
    
                    # Normalize timeframe key to short form for consistent DB storage
                    # (DB always stores '1d', '1wk', '1mo' — not full label like 'Daily (1d)')
                    _tf_db_key = yf_interval  # '1d', '1wk', or '1mo' — set above from scan_timeframe
    
                    bulk_cached = bulk_get_cached_ohlcv(scan_symbols, _tf_db_key)
                    
                    for sym in scan_symbols:
                        clean_sym = sym.strip().upper().replace(".NS", "")
                        if clean_sym in bulk_cached and not bulk_cached[clean_sym].empty:
                            bulk_data[sym.strip().upper()] = bulk_cached[clean_sym]
                        else:
                            missing_symbols.append(sym)
                    
                    if len(missing_symbols) > 0:
                        status_box.text(f"Phase 2/3: Downloading {len(missing_symbols)} missing symbols...")
                        chunk_size = 30
                        sym_chunks = [missing_symbols[i:i + chunk_size] for i in range(0, len(missing_symbols), chunk_size)]
                        
                        def download_chunk(chunk_idx, chunk):
                            # Pre-fill with empty DataFrames so Phase 3 does not retry failed/delisted symbols individually
                            chunk_data = {s.strip().upper(): pd.DataFrame() for s in chunk}
                            chunk_ns = [f"{s.strip().upper()}.NS" for s in chunk]
    
                            retries = 0
                            max_retries = 2
                            backoff = 2.0
    
                            while retries <= max_retries:
                                try:
                                    # yfinance 1.x
                                    df_bulk = yf.download(
                                        tickers=chunk_ns,
                                        period=yf_period,
                                        interval=yf_interval,
                                        progress=False,
                                        threads=False,
                                        timeout=20
                                    )
    
                                    if df_bulk is None or df_bulk.empty:
                                        raise ValueError("Empty DataFrame returned from yfinance")
    
                                    for sym in chunk:
                                        sym_ns = f"{sym.strip().upper()}.NS"
    
                                        try:
                                            if isinstance(df_bulk.columns, pd.MultiIndex):
                                                all_tickers_bulk = df_bulk.columns.get_level_values(1).unique().tolist()
                                                matched = next((t for t in all_tickers_bulk if t.upper() == sym_ns.upper()), None)
    
                                                if matched is None:
                                                    continue
    
                                                ticker_df = df_bulk.xs(matched, axis=1, level=1).copy()
    
                                            else:
                                                if len(chunk_ns) == 1:
                                                    ticker_df = df_bulk.copy()
    
                                                    if isinstance(ticker_df.columns, pd.MultiIndex):
                                                        ticker_df.columns = ticker_df.columns.droplevel(1)
                                                else:
                                                    continue
    
                                            required_cols = ["Open", "High", "Low", "Close", "Volume"]
    
                                            if all(col in ticker_df.columns for col in required_cols):
                                                ticker_df = ticker_df[required_cols].dropna(subset=["Close"])
                                                ticker_df = ticker_df[ticker_df["Volume"] > 0]
    
                                                if not ticker_df.empty:
                                                    ticker_df = ticker_df.reset_index()
                                                    ticker_df.rename(columns={ticker_df.columns[0]: "Date"}, inplace=True)
                                                    ticker_df["Date"] = pd.to_datetime(ticker_df["Date"], utc=True).dt.tz_localize(None)
    
                                                    chunk_data[sym.strip().upper()] = ticker_df
    
                                        except Exception:
                                            pass
    
                                    # Upload successfully downloaded data to Turso DB in a background thread
                                    # This prevents the 60-second database upload from blocking the next Yahoo download!
                                    valid_data = {k: v.copy() for k, v in chunk_data.items() if not v.empty}
                                    if valid_data:
                                        def _upload_chunk(data_dict, tf):
                                            for sym_k, df_v in data_dict.items():
                                                try:
                                                    save_to_cache(sym_k, df_v, tf)
                                                except Exception:
                                                    pass
                                        import threading
                                        threading.Thread(target=_upload_chunk, args=(valid_data, _tf_db_key), daemon=True).start()
    
                                    return chunk_data
    
                                except Exception as chunk_ex:
                                    retries += 1
    
                                    if retries > max_retries:
                                        print(f"Error downloading chunk {chunk_idx+1}: {chunk_ex}")
                                        break
    
                                    import time
                                    time.sleep(backoff)
                                    backoff *= 2.0
    
                            return chunk_data
    
                        import time
                        # Process sequentially and add a tiny sleep to be polite to the Yahoo Finance API.
                        # This prevents Streamlit from dropping the WebSocket connection (blank screen) due to excessive rate-limit backoff blocking.
                        for chunk_idx, chunk in enumerate(sym_chunks):
                            res = download_chunk(chunk_idx, chunk)
                            bulk_data.update(res)
                            
                            prog_bar.progress((chunk_idx + 1) / len(sym_chunks))
                            status_box.text(f"Phase 2/3: Downloading historical data (Chunk {chunk_idx+1}/{len(sym_chunks)})...")
                            
                            # Add a proactive sleep between chunks to avoid Yahoo rate limits
                            if chunk_idx < len(sym_chunks) - 1:
                                time.sleep(1)
    
                    if len(bulk_data) > 0:
                        st.session_state[cache_key_p2] = bulk_data
            
            mcap_cache = {}
            status_box.text(f"Phase 3/3: Scanning {n_stocks} active NSE listed equities...")
            prog_bar.progress(0)
            
    
            import joblib
            import os
            
            status_box.text(f"Phase 3/3: Fetching Benchmark Data (^NSEI)...")
            try:
                nifty_benchmark_df = yf.download("^NSEI", period=yf_period, interval=yf_interval, progress=False, threads=False, timeout=15)
            except Exception as _nsei_err:
                print(f"Warning: Failed to download Nifty benchmark data: {_nsei_err}")
                nifty_benchmark_df = pd.DataFrame()  # Use empty DF as fallback
            
            status_box.text(f"Phase 3/3: Scanning {n_stocks} active NSE listed equities...")
            prog_bar.progress(0)
            
            # Parallel Execution Core
            def process_and_fetch_if_needed(sym, df, benchmark_df, *args):
                try:
                    if df is None or len(df) == 0:
                        return {"failed": True, "error": "No historical data available"}
                    return process_single_symbol(sym, df, benchmark_df, *args)
                except Exception as e:
                    print(f"Internal error processing {sym}: {e}")
                    return {"failed": True, "error": str(e)}
    
            n_workers = 4
            sma_timeframe_val = st.session_state.get("sma_timeframe_tab", "All (Multi-Timeframe Convergence)")
            generator = joblib.Parallel(n_jobs=n_workers, backend="threading", return_as="generator_unordered")(
                joblib.delayed(process_and_fetch_if_needed)(
                    sym, bulk_data.get(sym.strip().upper()), nifty_benchmark_df, open_price_map, close_price_map, high_price_map, low_price_map, volume_map, min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes, min_signal_str, above_50dma_only, above_200dma_only, vcp_max_tightness, sma20_lower_bound, sma20_upper_bound, sma50_lower_bound, sma50_upper_bound, sma20_min_volume, sma_timeframe_val, scan_mode_flag
                ) for sym in scan_symbols
            )
            
            for i, res in enumerate(generator):
                try:
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
                    if res.get("vpa_squeeze"): vpa_squeeze_list.append(res["vpa_squeeze"])
                    if res.get("zanger"): zanger_list.append(res["zanger"])
                    if res.get("volume_profile"): vp_list.append(res["volume_profile"])
                    if res.get("support_rsi"): support_rsi_list.append(res["support_rsi"])
                    if res.get("ema_support"): ema_support_list.append(res["ema_support"])
                    if res.get("stage_analysis"): stage_analysis_list.append(res["stage_analysis"])
                    if res.get("stage2"): stage2_list.append(res["stage2"])
                    if res.get("monthly_momentum"): monthly_momentum_list.append(res["monthly_momentum"])
                    if res.get("weekly_momentum"): weekly_momentum_list.append(res["weekly_momentum"])
                    if res.get("near_30sma"): near_30sma_list.append(res["near_30sma"])  # FIX: was missing from main pass
                    if res.get("near_30sma_weekly"): near_30sma_weekly_list.append(res["near_30sma_weekly"])
                    if res.get("near_30sma_monthly"): near_30sma_monthly_list.append(res["near_30sma_monthly"])
                except Exception as exc:
                    print(f"Error processing result: {exc}")
                    failed_count += 1
                    
                # Throttle UI Updates (every 25 iterations or at the end)
                if (i + 1) % 25 == 0 or i + 1 == n_stocks:
                    status_box.text(f"Phase 3/3: Scanning ({i+1}/{n_stocks})")
                    prog_bar.progress((i + 1) / n_stocks)
    
            # Clean progress assets
            prog_bar.empty()
            status_box.empty()
            # Retry pass for symbols that failed the first time (often just rate-limited, not actually bad)
            def _is_empty_data(sym_key):
                """Safely check if a symbol has no usable data in bulk_data."""
                v = bulk_data.get(sym_key)
                if v is None: return True
                if isinstance(v, pd.DataFrame): return v.empty
                return True
    
            failed_syms_retry = [s for s in scan_symbols if _is_empty_data(s.strip().upper())]
            if failed_syms_retry and len(failed_syms_retry) < n_stocks * 0.6:
                status_box.text(f"Retrying {len(failed_syms_retry)} failed symbols after cool-down...")
                time.sleep(5)
                retry_chunk_size = 15
                retry_chunks = [failed_syms_retry[i:i+retry_chunk_size] for i in range(0, len(failed_syms_retry), retry_chunk_size)]
                for rc_idx, rc in enumerate(retry_chunks):
                    rc_ns = [f"{s.strip().upper()}.NS" for s in rc]
                    try:
                        df_retry = yf.download(tickers=rc_ns, period=yf_period, interval=yf_interval, progress=False, threads=False, timeout=20)
                        if df_retry is not None and not df_retry.empty:
                            for sym in rc:
                                sym_ns = f"{sym.strip().upper()}.NS"
                                try:
                                    if isinstance(df_retry.columns, pd.MultiIndex):
                                        all_t = df_retry.columns.get_level_values(1).unique().tolist()
                                        matched = next((t for t in all_t if t.upper() == sym_ns.upper()), None)
                                        if matched is None:
                                            continue
                                        t_df = df_retry.xs(matched, axis=1, level=1).copy()
                                    else:
                                        if len(rc_ns) == 1:
                                            t_df = df_retry.copy()
                                        else:
                                            continue
                                    req = ["Open", "High", "Low", "Close", "Volume"]
                                    if all(c in t_df.columns for c in req):
                                        t_df = t_df[req].dropna(subset=["Close"])
                                        t_df = t_df[t_df["Volume"] > 0]
                                        if not t_df.empty:
                                            t_df = t_df.reset_index()
                                            t_df.rename(columns={t_df.columns[0]: "Date"}, inplace=True)
                                            t_df["Date"] = pd.to_datetime(t_df["Date"], utc=True).dt.tz_localize(None)
                                            bulk_data[sym.strip().upper()] = t_df
                                            # Save to cache in background
                                            _t_df_copy = t_df.copy()
                                            _sym_key = sym.strip().upper()
                                            from local_cache_manager import save_to_cache
                                            import threading as _rt
                                            _rt.Thread(target=save_to_cache, args=(_sym_key, _t_df_copy, _tf_db_key), daemon=True).start()
                                except Exception:
                                    pass
                    except Exception as retry_ex:
                        print(f"Retry chunk {rc_idx+1} failed: {retry_ex}")
                    import random
                    time.sleep(random.uniform(1.5, 2.5))
                status_box.text(f"Retry complete. Re-scanning recovered symbols...")
                # Re-run process_single_symbol only for the symbols that now have data
                recovered = [s for s in failed_syms_retry if bulk_data.get(s.strip().upper()) is not None and not bulk_data.get(s.strip().upper()).empty]
                for sym in recovered:
                    try:
                        res = process_single_symbol(sym, bulk_data.get(sym.strip().upper()), nifty_benchmark_df, open_price_map, close_price_map, high_price_map, low_price_map, volume_map, min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes, min_signal_str, above_50dma_only, above_200dma_only, vcp_max_tightness, sma20_lower_bound, sma20_upper_bound, sma50_lower_bound, sma50_upper_bound, sma20_min_volume, sma_timeframe_val, scan_mode_flag)
                        if not res.get("failed"):
                            failed_count -= 1
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
                            if res.get("vpa_squeeze"): vpa_squeeze_list.append(res["vpa_squeeze"])
                            if res.get("zanger"): zanger_list.append(res["zanger"])
                            if res.get("volume_profile"): vp_list.append(res["volume_profile"])
                            if res.get("support_rsi"): support_rsi_list.append(res["support_rsi"])
                            if res.get("ema_support"): ema_support_list.append(res["ema_support"])
                            if res.get("stage_analysis"): stage_analysis_list.append(res["stage_analysis"])
                            if res.get("stage2"): stage2_list.append(res["stage2"])
                            if res.get("monthly_momentum"): monthly_momentum_list.append(res["monthly_momentum"])
                            if res.get("weekly_momentum"): weekly_momentum_list.append(res["weekly_momentum"])
                            if res.get("near_30sma"): near_30sma_list.append(res["near_30sma"])
                    except Exception:
                        pass
            
            # Cache results in state to allow seamless widget interactions
            # Safety net: if fresh scan returned 0 VDU results, fall back to last good DB results
            if len(flagged_list) == 0:
                try:
                    import database as _db_fb
                    _fb_dates = _db_fb.get_available_scan_dates()
                    if _fb_dates:
                        _fb_results = _db_fb.get_cached_breakouts(_fb_dates[0])
                        if _fb_results and len(_fb_results) > 0:
                            flagged_list = _fb_results
                            st.toast(f"⚠️ Fresh scan found 0 VDU results. Loaded {len(flagged_list)} cached results from {_fb_dates[0]}.", icon="📅")
                except Exception:
                    pass
            st.session_state.scan_results = flagged_list
            st.session_state.gapup_results = gapup_list
            st.session_state.above_ma_results = above_ma_list
            st.session_state.support_ma_results = support_ma_list
            st.session_state.crossover_ma_results = crossover_ma_list
            st.session_state.minervini_results = minervini_list
            st.session_state.vcs_results = vcs_list
            st.session_state.structural_vcp_results = structural_vcp_list
            
            # Populate VCP+Minervini tab results and filter them
            if len(structural_vcp_list) > 0:
                import pandas as pd
                vcp_df = pd.DataFrame(structural_vcp_list)
                # Filter for true setups (SQUEEZE or Entry Signal)
                if 'Entry Signal' in vcp_df.columns:
                    # Add score for ranking: Higher RS Proxy is better, lower VCP Range % is better
                    rs_proxy = pd.to_numeric(vcp_df.get('RS Proxy', 50), errors='coerce').fillna(50)
                    vcp_range = pd.to_numeric(vcp_df.get('VCP range %', 100), errors='coerce').fillna(100)
                    vcp_df['Score'] = rs_proxy - (vcp_range * 5)
                    
                    # Sort everything by Score
                    vcp_df = vcp_df.sort_values(by='Score', ascending=False)
                    
                    # Insert Rank column
                    vcp_df.insert(0, 'Rank', range(1, len(vcp_df) + 1))
                    st.session_state.vcp_minervini_results = vcp_df.to_dict('records')
                else:
                    st.session_state.vcp_minervini_results = structural_vcp_list
            else:
                st.session_state.vcp_minervini_results = []
    
            st.session_state.vpa_results = vpa_list
            st.session_state.vpa_squeeze_results = vpa_squeeze_list
            st.session_state.near_30sma_results = near_30sma_list
            st.session_state.near_30sma_weekly_results = near_30sma_weekly_list
            st.session_state.near_30sma_monthly_results = near_30sma_monthly_list
            st.session_state.wt_results = wt_list
            st.session_state.wt_results_by_tf = {"Daily_-40.0": wt_list, "Daily": wt_list}
            
            # Rank Dan Zanger signals
            if len(zanger_list) > 0:
                import pandas as pd
                from zanger_scanner import rank_signals, ZangerConfig
                hits_df = pd.DataFrame(zanger_list)
                ranked_df = rank_signals(hits_df, ZangerConfig())
                st.session_state.zanger_results = ranked_df.to_dict('records')
            else:
                st.session_state.zanger_results = zanger_list
                
            st.session_state.vp_results = vp_list
            st.session_state.support_rsi_results = support_rsi_list
            st.session_state.ema_support_results = ema_support_list
            st.session_state.stage_analysis_results = stage_analysis_list
            st.session_state.stage2_results = stage2_list
            st.session_state.monthly_momentum_results = monthly_momentum_list
            st.session_state.weekly_momentum_results = weekly_momentum_list
            
            st.session_state.total_scanned = n_stocks
            st.session_state.failed_count = failed_count
            st.session_state.last_scanned = datetime.now(IST_TIMEZONE).strftime("%Y-%m-%d %I:%M:%S %p")
            
            # Save to database cache daily
            db_save_succeeded = False
            try:
                today_ist_str = get_market_date()
                trend_setups_list = above_ma_list + support_ma_list + crossover_ma_list + minervini_list
                
                if scan_mode_flag == "sma_only":
                    database.save_sma_scan_results(
                        date_str=today_ist_str,
                        trend_setups=trend_setups_list,
                        total_scanned=n_stocks
                    )
                    st.toast("💾 Today's SMA scan results cached!", icon="✅")
                    db_save_succeeded = True
                else:
                    db_save_succeeded = database.save_scan_results(
                        date_str=today_ist_str,
                        breakouts=flagged_list,
                        squeezes=[],
                        gapups=gapup_list,
                        trend_setups=trend_setups_list,
                        wt_cross=wt_list,
                        total_scanned=n_stocks,
                        vcs_results=vcs_list,
                        vpa_results=vpa_list,
                        near_30sma_list=near_30sma_list
                    )
                    try: database.save_zanger_scan(today_ist_str, "Daily", zanger_list)
                    except Exception as _ze: print(f"Zanger save error: {_ze}")
                    try: database.save_volume_profile_only(today_ist_str, vp_list)
                    except Exception as _vpe: print(f"Volume profile save error: {_vpe}")
                    try: database.save_support_rsi_only(today_ist_str, support_rsi_list)
                    except Exception as _sre: print(f"Support RSI save error: {_sre}")
                    try: database.save_ema_support_only(today_ist_str, ema_support_list)
                    except Exception as _ese: print(f"EMA support save error: {_ese}")
                    try: database.save_vpa_squeeze_only(today_ist_str, vpa_squeeze_list)
                    except Exception as _vse: print(f"VPA Squeeze save error: {_vse}")
                    try: database.save_stage_analysis_only(today_ist_str, stage_analysis_list)
                    except Exception as _sae: print(f"Stage analysis save error: {_sae}")
                    try: database.save_stage2_only(today_ist_str, stage2_list)
                    except Exception as _s2e: print(f"Stage2 save error: {_s2e}")
                    try: database.save_monthly_momentum_results(today_ist_str, monthly_momentum_list)
                    except Exception as _mme: print(f"Monthly momentum save error: {_mme}")
                    try: database.save_weekly_momentum_results(today_ist_str, weekly_momentum_list)
                    except Exception as _wme: print(f"Weekly momentum save error: {_wme}")
                    # Save near 30 SMA weekly and monthly (were missing)
                    try: database.save_near_30sma_weekly_only(today_ist_str, near_30sma_weekly_list)
                    except Exception as _n30we: print(f"Near 30 SMA weekly save error: {_n30we}")
                    try: database.save_near_30sma_monthly_only(today_ist_str, near_30sma_monthly_list)
                    except Exception as _n30me: print(f"Near 30 SMA monthly save error: {_n30me}")
                    # Save VCP
                    try: database.save_vcp_minervini_scan(today_ist_str, st.session_state.get('vcp_minervini_results', []))
                    except Exception as _vcpe: print(f"VCP save error: {_vcpe}")
                    
                    if db_save_succeeded:
                        st.toast(f"💾 Scan saved: {len(flagged_list)} VDU, {len(trend_setups_list)} SMA setups, {len(zanger_list)} Zanger signals!", icon="✅")
                    else:
                        st.toast("⚠️ Scan complete but DB save had issues. Results visible in this session.", icon="⚠️")
                        print(f"[DB SAVE] save_scan_results returned False for {today_ist_str}. Results in session state only.")
                
                # Trigger background AI scans automatically in the backend!
                all_flagged_syms = [r['symbol'] for r in flagged_list]
                if len(all_flagged_syms) > 0:
                    run_background_ai_scan(all_flagged_syms, today_ist_str)
            except Exception as db_err:
                import traceback
                print(f"[DB SAVE ERROR] Failed to cache scan results: {db_err}")
                print(traceback.format_exc())
                st.sidebar.warning(f"⚠️ Results shown but DB save failed: {db_err}")
            
            # Highlight large failure rate
            if n_stocks > 0 and (failed_count / n_stocks) > 0.20:
                st.sidebar.warning(f"⚠️ Failed to fetch {failed_count}/{n_stocks} symbols ({failed_count/n_stocks*100:.1f}%). Check internet connection.")
                
            st.success("✅ Scanner complete! Results have been updated.")
            import time
            time.sleep(1.5)
            # Only reset db_cache_checked if DB save succeeded so results reload from DB.
            # If save failed, keep session state results as-is (user can still see them).
            if db_save_succeeded:
                st.session_state['db_cache_checked'] = False
            st.rerun()

      except Exception as _scan_top_err:
          import traceback
          st.error(f"❌ Scanner crashed: {_scan_top_err}")
          st.code(traceback.format_exc())
          print(f"[SCAN TOP-LEVEL ERROR] {traceback.format_exc()}")  


# Display Last Scanned Timestamp
if st.session_state.last_scanned:
    st.sidebar.markdown("---")
    

    st.sidebar.markdown("### ⚙️ Scanner Settings")
    st.sidebar.markdown("<p style='font-size:0.8rem; color:#94a3b8; margin-bottom:10px;'>Settings for VDU, Zanger, & Minervini Scans</p>", unsafe_allow_html=True)
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
st.markdown("---")

# Get scan cache (used by multiple tabs)
scan_data = st.session_state.scan_results

(tab_results, tab_detail, tab_watchlist, tab_ai, tab_sma, tab_sma65,
 tab_macross, tab_wave, tab_minervini, tab_monthly, tab_weekly, tab_history,
 tab_vcs, tab_vcp, tab_stage2, tab_vpa, tab_alerts, tab_volprofile, tab_support, tab_rsi_wt, tab_ema_support, tab_stage_analysis, tab_vpa_squeeze, tab_near_30sma) = st.tabs([
    "📊 Results", "📈 Detail", "📋 Watchlist", "🤖 AI Pattern",
    "📈 20&50 SMA", "🛡️ 65 SMA", "🔄 MA Cross",
    "🌊 Wave", "🏆 Minervini", "📅 Monthly", "📈 Weekly",
    "📅 History", "📉 Dan Zanger Scanner", "🎯 VCP+Minervini", "🚀 Stage2 Brk",
    "🚥 VPA", "🔄 Alerts", "📊 Vol Profile", "🛡️ Support", "🎯 RSI Oversold", "📈 9/21 EMA Support", "🏆 Stage Analysis", "📉 VPA Squeeze", "📉 Near 30 SMA"
])

# ==============================================================================
# TAB 1: SCANNER RESULTS
# ==============================================================================
with tab_results:
    import tabs.tab_results
    tabs.tab_results.render()

# ==============================================================================
# TAB 2: STOCK DETAIL
# ==============================================================================
with tab_detail:
    import tabs.tab_detail
    tabs.tab_detail.render()

# ==============================================================================
# TAB 3: WATCHLIST
# ==============================================================================
with tab_watchlist:
    tabs.tab_watchlist.render()

with tab_ai:
    import tabs.tab_ai
    tabs.tab_ai.render()


# ==============================================================================
# TAB 6: GAP-UP SETUPS
# ==============================================================================
if False: # Removed tab_gapup
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
        csv_g_data = export_g_df.to_csv(index=False).encode('utf-8-sig')
        
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
with tab_sma:
    import tabs.tab_sma
    tabs.tab_sma.render()

# ==============================================================================
# TAB 8: 65 SMA SUPPORT
# ==============================================================================
with tab_sma65:
    import tabs.tab_sma65
    tabs.tab_sma65.render()

# ==============================================================================
# TAB 9: MA CROSSOVERS
# ==============================================================================
with tab_macross:
    import tabs.tab_macross
    tabs.tab_macross.render()

# ==============================================================================
# TAB 10: WAVE TREND (LazyBear)
# ==============================================================================
with tab_wave:
    import tabs.tab_wave
    tabs.tab_wave.render()


# ==============================================================================
# TAB 11: MARK MINERVINI STAGE-2 TREND TEMPLATE
# ==============================================================================
with tab_minervini:
    import tabs.tab_minervini
    tabs.tab_minervini.render()


# ==============================================================================
# TAB 12: SCAN HISTORY VIEWER
# ==============================================================================
with tab_history:
    import tabs.tab_history
    tabs.tab_history.render()


# ==============================================================================
# TAB: MONTHLY MOMENTUM SCANNER (EMA Stack + ROC + RSI + Volume > Vol SMA)
# ==============================================================================
with tab_monthly:
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
        run_mm_scan = st.button("🔍 Run Monthly Momentum Scan", width="stretch", key="run_monthly_mom_btn")
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
            from scanner import run_monthly_momentum_update
            
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
        from data_fetcher import get_index_stocks, get_all_nse_symbols, get_top1000_nse_symbols
        if "Top 1000" in universe_selection or "1000" in universe_selection:
            mm_universe = get_top1000_nse_symbols()
        elif "NIFTY 500" in universe_selection:
            mm_universe = get_index_stocks("NIFTY 500")
        elif "NIFTY 100" in universe_selection:
            mm_universe = get_index_stocks("NIFTY 100")
        elif "NIFTY 50" in universe_selection:
            mm_universe = get_index_stocks("NIFTY 50")
        else:
            mm_universe = get_all_nse_symbols()

        mm_results = []
        mm_prog = st.progress(0)
        mm_status = st.empty()

        # ---- Fetch all from Master Cache & Resample ----
        mm_status.text("Step 1/3 — Fetching data from Master Cache...")
        from local_cache_manager import bulk_get_cached_ohlcv, resample_ohlcv
        
        symbols_to_scan = [s.strip().upper() for s in mm_universe]
        bulk_cached = bulk_get_cached_ohlcv(symbols_to_scan, "1d")
        
        # --- Fallback to yfinance for missing symbols ---
        missing_syms = [s for s in symbols_to_scan if s.replace(".NS", "") not in bulk_cached]
        if missing_syms:
            mm_status.text(f"Step 1.5/3 — Fetching {len(missing_syms)} missing stocks from yfinance...")
            import yfinance as yf
            import pandas as pd
            chunk_size = 100
            missing_chunks = [missing_syms[i:i+chunk_size] for i in range(0, len(missing_syms), chunk_size)]
            for c_idx, chunk in enumerate(missing_chunks):
                tkrs = [f"{s}.NS" if not s.endswith(".NS") else s for s in chunk]
                try:
                    df_yf = yf.download(tickers=tkrs, period="5y", interval="1d", progress=False, threads=False, timeout=15)
                    if not df_yf.empty:
                        for sym in chunk:
                            try:
                                if isinstance(df_yf.columns, pd.MultiIndex):
                                    all_tkrs = df_yf.columns.get_level_values(1).unique().tolist()
                                    matched_t = next((t for t in all_tkrs if t.upper() == f"{sym}.NS".upper()), None)
                                    if not matched_t:
                                        continue
                                    t_df = df_yf.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                                else:
                                    t_df = df_yf.dropna(subset=['Close'])
                                    
                                if not t_df.empty:
                                    t_df = t_df.reset_index()
                                    if 'Date' not in t_df.columns:
                                        t_df.rename(columns={t_df.columns[0]: 'Date'}, inplace=True)
                                    t_df['Date'] = pd.to_datetime(t_df['Date'])
                                    bulk_cached[sym.replace(".NS", "")] = t_df
                            except Exception:
                                pass
                except Exception:
                    pass
        
        mm_status.text(f"Step 2/3 — Resampling to Monthly & Scanning {len(bulk_cached)} stocks...")
        mm_total = len(bulk_cached)
        
        for idx, (sym, d_df) in enumerate(bulk_cached.items()):
            if idx % 50 == 0:
                mm_status.text(f"Step 3/3 — Scanning: {idx+1}/{mm_total} ({len(mm_results)} matches so far)...")
                mm_prog.progress((idx + 1) / max(mm_total, 1))
                
            if d_df is None or d_df.empty:
                continue
                
            cmp = d_df['Close'].iloc[-1]
            if cmp < 100:
                continue
                
            m_df = resample_ohlcv(d_df, 'ME')  # 'ME' is month-end frequency
            if len(m_df) < 22:
                continue
                
            res_m = scan_monthly_momentum(sym, m_df, market_cap_cr=0.0)
            if res_m:
                mm_results.append(res_m)

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
        mm_csv = pd.DataFrame(mm_export).to_csv(index=False).encode('utf-8-sig')
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
with tab_weekly:
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
        run_wm_scan = st.button("🔍 Run Weekly Momentum Scan", width="stretch", key="run_weekly_mom_btn")
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
        if "NIFTY 500" in universe_selection:
            wm_universe = get_index_stocks("NIFTY 500")
        elif "NIFTY 100" in universe_selection:
            wm_universe = get_index_stocks("NIFTY 100")
        elif "NIFTY 50" in universe_selection:
            wm_universe = get_index_stocks("NIFTY 50")
        else:
            wm_universe = get_all_nse_symbols()

        wm_results = []
        wm_prog    = st.progress(0)
        wm_status  = st.empty()

        # ---- Fetch all from Master Cache & Resample ----
        wm_status.text("Step 1/3 — Fetching data from Master Cache...")
        from local_cache_manager import bulk_get_cached_ohlcv, resample_ohlcv
        
        symbols_to_scan = [s.strip().upper() for s in wm_universe]
        bulk_cached = bulk_get_cached_ohlcv(symbols_to_scan, "1d")
        
        wm_status.text(f"Step 2/3 — Resampling to Weekly & Scanning {len(bulk_cached)} stocks...")
        wm_total = len(bulk_cached)
        
        for idx, (sym, d_df) in enumerate(bulk_cached.items()):
            if idx % 50 == 0:
                wm_status.text(f"Step 3/3 — Scanning: {idx+1}/{wm_total} ({len(wm_results)} matches so far)...")
                wm_prog.progress((idx + 1) / max(wm_total, 1))
                
            if d_df is None or d_df.empty:
                continue
                
            cmp = d_df['Close'].iloc[-1]
            if cmp < 200:
                continue
                
            w_df = resample_ohlcv(d_df, 'W')
            if len(w_df) < 22:
                continue
                
            res_w = scan_weekly_momentum(sym, w_df, market_cap_cr=0.0)
            if res_w:
                wm_results.append(res_w)

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
        wm_csv = pd.DataFrame(wm_export).to_csv(index=False).encode('utf-8-sig')
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
# TAB: DAN ZANGER SCANNER
# ==============================================================================
with tab_vcs:
    import tabs.tab_vcs
    tabs.tab_vcs.render()

# ==============================================================================
# TAB: VCP+Minervini
# ==============================================================================
with tab_vcp:
    import tabs.tab_vcp
    tabs.tab_vcp.render()

# ==============================================================================
# TAB: EARLY STAGE 2 BREAKOUT
# ==============================================================================
with tab_stage2:
    import tabs.tab_stage2
    tabs.tab_stage2.render()



# ==============================================================================
# TAB 17: VPA TREND
# ==============================================================================
with tab_vpa:
    import tabs.tab_vpa
    tabs.tab_vpa.render()

# TAB: VPA SQUEEZE
with tab_vpa_squeeze:
    import tabs.tab_vpa_squeeze
    tabs.tab_vpa_squeeze.render()

# --- NEAR 30 SMA TAB ---
with tab_near_30sma:
    import tabs.tab_near_30sma
    tabs.tab_near_30sma.render()

# TAB: FREQUENT FLYERS (CONSISTENT ALERTS)
with tab_alerts:
    import tabs.tab_alerts
    tabs.tab_alerts.render()


# --- VOLUME PROFILE SCANNER TAB ---
with tab_volprofile:
    import tabs.tab_volprofile
    tabs.tab_volprofile.render()

# ==============================================================================
# TAB 20: CONFLUENCE — WaveTrend Buy Signal + Volume Profile Daily Support
# ==============================================================================
if False: # Removed tab_confluence
    try:
        st.markdown("### 💎 Confluence Scanner — WaveTrend Buy + Volume Profile Support")
        st.markdown(
            "<p style='font-size:0.9rem; color:#94a3b8; margin-top:-8px; line-height:1.5;'>"
            "Stocks appearing <b style='color:#00e676;'>simultaneously</b> in two independent scanners: "
            "<b style='color:#29b6f6;'>🌊 WaveTrend</b> (WT1 crossed above WT2 = Buy Signal) "
            "<b>AND</b> <b style='color:#ffa000;'>📊 Volume Profile</b> (Daily timeframe = ✅ Can Buy Near Support). "
            "This dual-confirmation dramatically increases trade conviction."
            "</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")

        today_str = get_market_date(for_display=True)
        confluence_data = database.get_wt_vp_confluence(today_str)

        # Metrics row
        cf_m1, cf_m2, cf_m3 = st.columns(3)
        cf_count = len(confluence_data) if confluence_data else 0
        cf_deepest_wt = min(r['wt_value'] for r in confluence_data) if confluence_data else 0.0
        cf_avg_va = (sum(r['daily_pos'] for r in confluence_data) / cf_count) if cf_count > 0 else 0.0

        cf_m1.markdown(
            f'<div class="glass-card metric-glow-green">'
            f'<p style="font-size:0.85rem; color:#94a3b8; margin:0;">💎 Confluence Matches</p>'
            f'<h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#00e676;">{cf_count}</h3>'
            f'</div>', unsafe_allow_html=True
        )
        cf_m2.markdown(
            f'<div class="glass-card metric-glow-blue">'
            f'<p style="font-size:0.85rem; color:#94a3b8; margin:0;">🌊 Deepest WT1 Value</p>'
            f'<h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#29b6f6;">{cf_deepest_wt:.1f}</h3>'
            f'</div>', unsafe_allow_html=True
        )
        cf_m3.markdown(
            f'<div class="glass-card metric-glow-purple">'
            f'<p style="font-size:0.85rem; color:#94a3b8; margin:0;">📊 Avg Daily VA Position</p>'
            f'<h3 style="font-size:1.8rem; margin:5px 0 0 0; color:#ce93d8;">{cf_avg_va:.1f}%</h3>'
            f'</div>', unsafe_allow_html=True
        )

        if not confluence_data:
            st.info(
                f"No confluence matches found for {today_str}. "
                "This requires both **WaveTrend** and **Volume Profile** scans to have run today. "
                "Run them from the 🌊 Wave and 📊 Vol Profile tabs first."
            )
        else:
            # Build table rows
            rows_html = []
            for idx, r in enumerate(confluence_data, 1):
                sym = r.get('symbol', '')
                cmp = r.get('cmp', 0.0)
                chg = r.get('day_change_pct', 0.0)
                wt1 = r.get('wt_value', 0.0)
                wt2 = r.get('wt2_value', 0.0)
                wt_d = r.get('wt_diff', 0.0)
                vol = r.get('volume', 0)
                va_pos = r.get('daily_pos', 0.0)
                poc = r.get('daily_poc', 0.0)
                val_price = r.get('daily_val', 0.0)
                vah_price = r.get('daily_vah', 0.0)

                # SMA badges
                sma_badges = []
                if r.get('above_200sma'): sma_badges.append('<span style="background:rgba(0,230,118,0.12); color:#00e676; padding:1px 5px; border-radius:3px; font-size:0.72rem; font-weight:600;">200</span>')
                if r.get('above_50sma'): sma_badges.append('<span style="background:rgba(41,182,246,0.12); color:#29b6f6; padding:1px 5px; border-radius:3px; font-size:0.72rem; font-weight:600;">50</span>')
                if r.get('above_20sma'): sma_badges.append('<span style="background:rgba(206,147,216,0.12); color:#ce93d8; padding:1px 5px; border-radius:3px; font-size:0.72rem; font-weight:600;">20</span>')
                sma_html = " ".join(sma_badges) if sma_badges else '<span style="color:#64748b; font-size:0.72rem;">—</span>'

                # Day change color
                chg_color = "#00e676" if chg >= 0 else "#ef4444"
                chg_sign = "+" if chg >= 0 else ""

                row = (
                    f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.2s;">'
                    f'<td style="padding:10px 12px; color:#64748b; font-weight:500;">{idx}</td>'
                    f'<td style="padding:10px 12px; font-weight:bold;">'
                    f'<a href="https://in.tradingview.com/chart/?symbol=NSE:{sym}" target="_blank" style="color:#29b6f6; text-decoration:none;">{sym}</a>'
                    f'</td>'
                    f'<td style="padding:10px 12px; color:#e2e8f0; font-weight:500;">₹{cmp:,.2f}</td>'
                    f'<td style="padding:10px 12px; color:{chg_color}; font-weight:600;">{chg_sign}{chg:.2f}%</td>'
                    f'<td style="padding:10px 12px; color:#00e676; font-weight:bold;">{wt1:.1f}</td>'
                    f'<td style="padding:10px 12px; color:#94a3b8;">{wt2:.1f}</td>'
                    f'<td style="padding:10px 12px; color:#ffa000; font-weight:500;">{va_pos:.1f}%</td>'
                    f'<td style="padding:10px 12px; color:#00e676;">₹{val_price:,.2f}</td>'
                    f'<td style="padding:10px 12px; color:#ce93d8;">₹{poc:,.2f}</td>'
                    f'<td style="padding:10px 12px; color:#ef4444;">₹{vah_price:,.2f}</td>'
                    f'<td style="padding:10px 12px;">{sma_html}</td>'
                    f'<td style="padding:10px 12px; color:#94a3b8; text-align:right;">{vol:,}</td>'
                    f'</tr>'
                )
                rows_html.append(row)

            table_body = "".join(rows_html)

            st.markdown(
                f'<div class="glass-card" style="padding:18px; margin-bottom:22px; border:1px solid rgba(0,230,118,0.2); background:rgba(9,13,22,0.55); border-radius:12px;">'
                f'<h3 style="margin-top:0; color:#00e676; font-size:1.15rem; display:flex; align-items:center; gap:8px; font-family:Outfit,sans-serif;">'
                f'💎 Dual-Confirmation Buy Candidates — {today_str}'
                f'</h3>'
                f'<p style="font-size:0.82rem; color:#94a3b8; margin-top:-8px; margin-bottom:15px; font-family:Outfit,sans-serif;">'
                f'Each stock below has a <b style="color:#00e676;">🟢 WaveTrend Buy Signal</b> (WT1 crossing above WT2 in oversold zone) '
                f'<b>AND</b> is in the <b style="color:#ffa000;">Volume Profile daily ✅ Can Buy zone</b> (near VAL support). '
                f'<b>Buy Range (VAL)</b> = support entry | <b>Target (POC)</b> = fair value target | <b>Resistance (VAH)</b> = upper limit.'
                f'</p>'
                f'<div style="overflow-x:auto;">'
                f'<table style="width:100%; border-collapse:collapse; text-align:left; font-size:0.85rem; color:#cbd5e1; font-family:Outfit,sans-serif;">'
                f'<thead>'
                f'<tr style="border-bottom:1px solid rgba(255,255,255,0.1); color:#38bdf8; font-weight:bold; background:rgba(0,230,118,0.04); font-size:0.78rem; text-transform:uppercase;">'
                f'<th style="padding:8px 12px;">#</th>'
                f'<th style="padding:8px 12px;">Symbol</th>'
                f'<th style="padding:8px 12px;">CMP</th>'
                f'<th style="padding:8px 12px;">Change</th>'
                f'<th style="padding:8px 12px; color:#00e676;">WT1</th>'
                f'<th style="padding:8px 12px;">WT2</th>'
                f'<th style="padding:8px 12px; color:#ffa000;">VA Pos %</th>'
                f'<th style="padding:8px 12px; color:#00e676;">Buy Range (VAL)</th>'
                f'<th style="padding:8px 12px; color:#ce93d8;">Target (POC)</th>'
                f'<th style="padding:8px 12px; color:#ef4444;">Resistance (VAH)</th>'
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

            # Quick Trade Board for confluence results
            render_quick_trade_board(confluence_data, key_prefix="confluence")

            # CSV Download
            df_confluence = pd.DataFrame([{
                'Symbol': r['symbol'],
                'CMP': r['cmp'],
                'Change %': round(r['day_change_pct'], 2),
                'WT1': round(r['wt_value'], 2),
                'WT2': round(r['wt2_value'], 2),
                'WT Diff': round(r['wt_diff'], 2),
                'VA Position %': round(r['daily_pos'], 2),
                'Buy Range (VAL)': round(r['daily_val'], 2),
                'Target (POC)': round(r['daily_poc'], 2),
                'Resistance (VAH)': round(r['daily_vah'], 2),
                'Above 20 SMA': r['above_20sma'],
                'Above 50 SMA': r['above_50sma'],
                'Above 200 SMA': r['above_200sma'],
                'Volume': r['volume'],
                'Buy Price': r.get('buy_price'),
                'Exit Price': r.get('exit_price'),
                'Target Price': r.get('target_price'),
                'Confidence': r.get('confidence'),
            } for r in confluence_data])

            csv_cf = df_confluence.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Confluence Results (CSV)",
                data=csv_cf,
                file_name=f"WT_VP_Confluence_{today_str}.csv",
                mime="text/csv",
                key="dl_confluence_csv"
            )

            # How it works
            with st.expander("ℹ️ How Confluence Works", expanded=False):
                st.markdown("""
                **This tab shows stocks that pass TWO independent filters simultaneously:**

                1. **🌊 WaveTrend Buy Signal** — The WT1 line has crossed ABOVE WT2 in the oversold zone (below -40), generating a bullish crossover buy signal (green dot).

                2. **📊 Volume Profile Daily Support** — The stock's current price is in the "✅ Can Buy (Near Support)" zone on the daily timeframe, meaning it's near the Value Area Low (VAL) — a strong institutional support level.

                **Why this matters:**
                - WaveTrend identifies **momentum reversal timing** (when to buy)
                - Volume Profile identifies **price support levels** (where to buy)
                - Together, they confirm both **timing AND price level**, dramatically increasing trade success probability

                **How to trade these signals:**
                - **Entry:** Near VAL (Buy Range) — this is the support level
                - **Target:** POC (Point of Control) — the high-volume fair value zone
                - **Stop Loss:** Below VAL by 1-2%
                """)

    except Exception as e:
        st.error(f"Error rendering Confluence tab: {e}")

# ==============================================================================
# TAB 21: SUPPORT BOUNCE — Stocks at Historical Support + RSI Oversold
# ==============================================================================
with tab_support:
    import tabs.tab_support
    tabs.tab_support.render()

# ==============================================================================
# TAB 22: RSI OVERSOLD SCANNER
# ==============================================================================
with tab_rsi_wt:
    import tabs.tab_rsi_wt
    tabs.tab_rsi_wt.render()



# ==============================================================================
# TAB: BB SQUEEZE
# ==============================================================================
with tab_ema_support:
    import tabs.tab_ema_support
    tabs.tab_ema_support.render()


# ==============================================================================
# TAB: STAGE ANALYSIS (MINERVINI TREND TEMPLATE)
# ==============================================================================
with tab_stage_analysis:
    import tabs.tab_stage_analysis
    tabs.tab_stage_analysis.render()


