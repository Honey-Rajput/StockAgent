
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



# --- Initialize Session State ---
if 'ema_support_results' not in st.session_state:
    st.session_state.ema_support_results = None
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
if 'vpa_results' not in st.session_state:
    st.session_state.vpa_results = []
if 'vp_results' not in st.session_state:
    st.session_state.vp_results = []
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
if 'zanger_results' not in st.session_state:
    st.session_state.zanger_results = None
if 'vcp_minervini_results' not in st.session_state:
    st.session_state.vcp_minervini_results = None
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
if "ALL_TAB_SCAN_STATUS" not in globals():
    ALL_TAB_SCAN_STATUS = {
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
    if ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
        return
        
    is_already_running = any(t.name == "Background_BB_Squeeze" for t in __import__('threading').enumerate())
    if is_already_running:
        return
        
    ALL_TAB_SCAN_STATUS["ema_support_running"] = True
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
                    ALL_TAB_SCAN_STATUS["ema_support_results"] = cached_bb
                    st.session_state.ema_support_results = cached_bb
                    ALL_TAB_SCAN_STATUS["ema_support_running"] = False
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
                    
            ALL_TAB_SCAN_STATUS["ema_support_results"] = bb_results
            st.session_state.ema_support_results = bb_results
            try:
                database.save_ema_support_only(today_str, bb_results)
            except:
                pass
            
        except Exception as e:
            print(f"BB Squeeze background thread crashed: {e}")
        finally:
            ALL_TAB_SCAN_STATUS["ema_support_running"] = False
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
    global ALL_TAB_SCAN_STATUS
    if ALL_TAB_SCAN_STATUS["is_running"]:
        return

    # Guard to prevent duplicate concurrent background scanning threads
    import threading
    is_already_running = any(t.name == "Background_All_Tab_Scans" for t in threading.enumerate())
    if is_already_running:
        print("Background all-tab scan thread is already active. Skipping duplicate thread launch.")
        return

    ALL_TAB_SCAN_STATUS["is_running"] = True
    ALL_TAB_SCAN_STATUS["status_text"] = "Initializing all-tab background scans..."
    ALL_TAB_SCAN_STATUS["progress"] = 0.0

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
                ALL_TAB_SCAN_STATUS["status_text"] = "All background tab scans already cached!"
                ALL_TAB_SCAN_STATUS["progress"] = 1.0
                ALL_TAB_SCAN_STATUS["current_scanner"] = "Complete"
                ALL_TAB_SCAN_STATUS["is_running"] = False
                print("[BG All-Tab] All background tab scans already cached. Skipping.")
                return

            raw_symbols = get_all_nse_symbols()
            all_symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
            # Phase 2: Shared Daily Download (1y or 2y)
            shared_daily_data = {}
            if run_wt or run_vcs or run_vpa or run_vp or run_vpa_sq or run_near_30sma:
                ALL_TAB_SCAN_STATUS["current_scanner"] = "Downloading Shared Data"
                ALL_TAB_SCAN_STATUS["status_text"] = "Downloading shared daily data for NSE symbols..."
                ALL_TAB_SCAN_STATUS["progress"] = 0.05
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
                            ALL_TAB_SCAN_STATUS["progress"] = 0.05 + (processed_count / len(all_symbols)) * 0.25
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
                if len(df) >= 200:
                    res = scan_vpa_ma_squeeze(sym, df)
                    if res:
                        return ("vpa_sq", res)
                return ("vpa_sq", None)

            def run_near_30sma_worker(sym, df):
                from scanner import scan_near_30sma
                res = scan_near_30sma(sym, df)
                if res:
                    return ("near_30sma", res)
                return ("near_30sma", None)

            # Phase 3: Parallel Execution
            wt_tf_results, custom_vcs_results, vpa_list, vp_list, vpa_sq_list, near_30sma_list = [], [], [], [], [], []
            
            if shared_daily_data:
                ALL_TAB_SCAN_STATUS["current_scanner"] = "Executing Scans"
                ALL_TAB_SCAN_STATUS["status_text"] = "Running concurrent daily scans..."
                
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
                            ALL_TAB_SCAN_STATUS["progress"] = 0.30 + (i / len(tasks_to_run)) * 0.45
                        try:
                            scan_type, result = future.result()
                            if result:
                                if scan_type == "wt": wt_tf_results.append(result)
                                elif scan_type == "vcs": custom_vcs_results.append(result)
                                elif scan_type == "vpa": vpa_list.append(result)
                                elif scan_type == "vp": vp_list.append(result)
                                elif scan_type == "vpa_sq": vpa_sq_list.append(result)
                                elif scan_type == "near_30sma": near_30sma_list.append(result)
                        except Exception:
                            pass
                
                # Save results
                if run_wt:
                    ALL_TAB_SCAN_STATUS["wt_results"] = wt_tf_results
                    try: database.save_wt_cross_only(today_str, wt_tf_results)
                    except: pass
                if run_vcs:
                    ALL_TAB_SCAN_STATUS["vcs_results"] = custom_vcs_results
                    try: database.save_vcs_only(today_str, custom_vcs_results)
                    except: pass
                if run_vpa:
                    ALL_TAB_SCAN_STATUS["vpa_results"] = vpa_list
                    try: database.save_vpa_only(today_str, vpa_list)
                    except: pass
                if run_vp:
                    ALL_TAB_SCAN_STATUS["volume_profile_results"] = vp_list
                    try: database.save_volume_profile_only(today_str, vp_list)
                    except: pass
                if run_vpa_sq:
                    ALL_TAB_SCAN_STATUS["vpa_squeeze_results"] = vpa_sq_list
                    try: database.save_vpa_squeeze_only(today_str, vpa_sq_list)
                    except: pass
                if run_near_30sma:
                    ALL_TAB_SCAN_STATUS["near_30sma_results"] = near_30sma_list
                    try: database.save_near_30sma_only(today_str, near_30sma_list)
                    except: pass

            # Phase 4: Stage-2 (Monthly)
            if run_s2:
                ALL_TAB_SCAN_STATUS["current_scanner"] = "Stage-2"
                ALL_TAB_SCAN_STATUS["status_text"] = "Running Stage-2 monthly scan..."
                ALL_TAB_SCAN_STATUS["progress"] = 0.75
                s2_cands = get_index_stocks("NIFTY 500")
                s2_res = []
                
                from local_cache_manager import bulk_get_cached_ohlcv, resample_ohlcv
                s2_bulk = bulk_get_cached_ohlcv([s.strip().upper() for s in s2_cands], "1d")
                
                for c_idx, (sym, t_df) in enumerate(s2_bulk.items()):
                    if c_idx % 20 == 0:
                        ALL_TAB_SCAN_STATUS["progress"] = 0.75 + (c_idx / max(len(s2_bulk), 1)) * 0.20
                        
                    if t_df is None or t_df.empty:
                        continue
                        
                    try:
                        m_df = resample_ohlcv(t_df, 'ME')
                        if not m_df.empty and len(m_df) >= 24:
                            res = scan_monthly_early_stage2(sym, m_df, max_run_up_pct=20.0)
                            if res: s2_res.append(res)
                    except Exception: pass
                    
                s2_res = sorted(s2_res, key=lambda x: x.get('score', 0), reverse=True)
                ALL_TAB_SCAN_STATUS["stage2_results"] = s2_res
                try: database.save_stage2_only(today_str, s2_res)
                except Exception: pass

            ALL_TAB_SCAN_STATUS["status_text"] = "All background tab scans complete!"
            ALL_TAB_SCAN_STATUS["progress"] = 1.0
            ALL_TAB_SCAN_STATUS["current_scanner"] = "Complete"
            ALL_TAB_SCAN_STATUS["is_running"] = False
            print("[BG All-Tab] All background tab scans complete!")

        except Exception as err:
            ALL_TAB_SCAN_STATUS["status_text"] = f"Background scan error: {err}"
            ALL_TAB_SCAN_STATUS["is_running"] = False
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

st.sidebar.markdown('### ⚡ Performance Settings')
enable_background_scans = st.sidebar.checkbox("Enable Auto-Background Scans", value=False, help="Disable this on Streamlit Cloud to prevent UI freezing due to heavy thread execution.")

# Automatically trigger scanning in background if results are missing for today
if enable_background_scans:
    if (st.session_state.monthly_momentum_results is None or st.session_state.weekly_momentum_results is None) and not MOMENTUM_SCAN_STATUS["is_running"]:
        run_background_momentum_scans()
    # Auto-trigger all remaining tab scans (WaveTrend, VCS, Stage-2, VPA, Volume Profile)
    if not ALL_TAB_SCAN_STATUS["is_running"]:
        run_background_all_tab_scans()
    # Auto-trigger BB Squeeze
    if not ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
        run_background_ema_support_scan()

# --- Automatic Daily Database Cache Loader ---
# Runs ONCE per browser session (db_cache_checked stays False until first load).
# Each scanner loads independently from its own DB table/date — NOT gated behind scan_logs.
if not st.session_state.get('db_cache_checked', False):
    st.session_state['db_cache_checked'] = True
    try:
        # Fetch the max dates for ALL tables at once
        all_latest_dates = database.get_all_latest_scan_dates()

        def _load_latest(table, getter_fn, state_key, post_fn=None):
            """Helper: find own latest date for a table, then load and set session state.
            Returns the date string the data was loaded from (or None)."""
            try:
                d = all_latest_dates.get(table)
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

        # Auto-resume background AI scan for flagged symbols
        all_syms = []
        if st.session_state.scan_results:
            all_syms.extend([r['symbol'] for r in st.session_state.scan_results])
        all_syms = list(set(all_syms))
        if all_syms and enable_background_scans:
            try:
                run_background_ai_scan(all_syms, latest_date_str if available_dates else get_market_date())
            except Exception as auto_scan_err:
                print(f"Failed to auto-resume background AI scan on boot: {auto_scan_err}")

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
    min_value=45,
    max_value=100,
    value=55,
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

    # Universe is hardcoded to Top 1000 NSE stocks
    universe_key = "TOP 1000"
    from data_fetcher import get_top1000_nse_symbols
    raw_symbols = get_top1000_nse_symbols()
        
    if not raw_symbols:
        st.sidebar.error("❌ No symbols found to scan.")
    else:
      try:
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
            
        today_date_str = get_market_date()
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

            # ── Smart Phase 1: Check Turso DB first ─────────────────────────────
            # NSE market hours: 9:15 AM – 3:30 PM IST on weekdays
            from datetime import time as _time
            _now_ist = datetime.now(IST_TIMEZONE)
            _market_open = (
                _now_ist.weekday() < 5
                and _time(9, 15) <= _now_ist.time() <= _time(15, 30)
            )

            _db_quotes = {}
            if not _market_open:
                # Market is closed — check if today's data is already in Turso
                status_box.text("Phase 1/3: Checking Turso DB for today's cached quotes...")
                try:
                    import concurrent.futures as _p1_cf
                    with _p1_cf.ThreadPoolExecutor(max_workers=1) as _p1_tex:
                        _fut = _p1_tex.submit(database.get_today_quotes, raw_symbols, today_date_str)
                        try:
                            _db_quotes = _fut.result(timeout=10)  # 10s timeout — don't hang the UI
                        except _p1_cf.TimeoutError:
                            print("Phase 1 DB check timed out after 10s — falling back to Yahoo")
                            _db_quotes = {}
                except Exception as _dq_err:
                    print(f"Phase 1 DB check error: {_dq_err}")
                    _db_quotes = {}

            _coverage = len(_db_quotes) / max(len(raw_symbols), 1)

            if _coverage >= 0.90:
                # ✅ Turso has today's data — skip Yahoo entirely!
                for _sym, _q in _db_quotes.items():
                    if _q["close"] > 0:
                        close_price_map[_sym]  = _q["close"]
                        open_price_map[_sym]   = _q["open"]
                        high_price_map[_sym]   = _q["high"]
                        low_price_map[_sym]    = _q["low"]
                        volume_map[_sym]       = _q["volume"]
                status_box.text(f"Phase 1/3: ✅ Loaded {len(close_price_map)} quotes from Turso DB (skipped Yahoo Finance!)")
                prog_bar.progress(1.0)
            else:
                # ⬇️ Download from Yahoo Finance (first scan of the day / market open)
                status_box.text("Phase 1/3: Downloading real-time quotes for selected universe...")
                import time
                chunk_size = 35  # 35 tickers per chunk → ~51 chunks for 1795 symbols (was 72)
                ticker_chunks = [all_tickers_ns[i:i + chunk_size] for i in range(0, len(all_tickers_ns), chunk_size)]
                
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
                
                # Use a single worker for large universes to prevent aggressive yfinance rate-limiting
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

                        # ✅ Save this chunk to Turso DB immediately (per-chunk progressive save)
                        # This means if the user closes the app mid-scan, all completed chunks
                        # are already in the DB and won't need to be downloaded again!
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

            st.session_state[cache_key_p1] = (open_price_map, close_price_map, volume_map, high_price_map, low_price_map)

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
        zanger_list = []
        vp_list = []
        support_rsi_list = []
        ema_support_list = []
        stage_analysis_list = []
        stage2_list = []
        monthly_momentum_list = []
        weekly_momentum_list = []
        near_30sma_list = []
        
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
                if res.get("zanger"): zanger_list.append(res["zanger"])
                if res.get("volume_profile"): vp_list.append(res["volume_profile"])
                if res.get("support_rsi"): support_rsi_list.append(res["support_rsi"])
                if res.get("ema_support"): ema_support_list.append(res["ema_support"])
                if res.get("stage_analysis"): stage_analysis_list.append(res["stage_analysis"])
                if res.get("stage2"): stage2_list.append(res["stage2"])
                if res.get("monthly_momentum"): monthly_momentum_list.append(res["monthly_momentum"])
                if res.get("weekly_momentum"): weekly_momentum_list.append(res["weekly_momentum"])
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
        st.session_state.near_30sma_results = near_30sma_list
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
        try:
            today_ist_str = get_market_date()
            trend_setups_list = above_ma_list + support_ma_list + crossover_ma_list + minervini_list
            
            if scan_mode_flag == "sma_only":
                database.save_sma_scan_results(
                    date_str=today_ist_str,
                    trend_setups=trend_setups_list,
                    total_scanned=n_stocks
                )
                st.toast("💾 Today's SMA scan results cached in Neon PostgreSQL!", icon="✅")
            else:
                database.save_scan_results(
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
                except Exception: pass
                try: database.save_volume_profile_only(today_ist_str, vp_list)
                except Exception: pass
                try: database.save_support_rsi_only(today_ist_str, support_rsi_list)
                except Exception: pass
                try: database.save_ema_support_only(today_ist_str, ema_support_list)
                except Exception: pass
                try: database.save_stage_analysis_only(today_ist_str, stage_analysis_list)
                except Exception: pass
                try: database.save_stage2_only(today_ist_str, stage2_list)
                except Exception: pass
                try: database.save_monthly_momentum_results(today_ist_str, monthly_momentum_list)
                except Exception: pass
                try: database.save_weekly_momentum_results(today_ist_str, weekly_momentum_list)
                except Exception: pass
                
                # Save VCP
                try: database.save_vcp_minervini_scan(today_ist_str, st.session_state.get('vcp_minervini_results', []))
                except Exception as e: print(f"Error saving VCP DB: {e}")
                
                st.toast("💾 Today's scan results cached in Neon PostgreSQL!", icon="✅")
            
            # Trigger background AI scans automatically in the backend!
            all_flagged_syms = [r['symbol'] for r in flagged_list]
            if len(all_flagged_syms) > 0:
                run_background_ai_scan(all_flagged_syms, today_ist_str)
        except Exception as db_err:
            print(f"Failed to cache daily scan results to database: {db_err}")
        
        # Highlight large failure rate
        if n_stocks > 0 and (failed_count / n_stocks) > 0.20:
            st.sidebar.warning(f"⚠️ Failed to fetch {failed_count}/{n_stocks} symbols ({failed_count/n_stocks*100:.1f}%). Check internet connection.")
            
        st.success("✅ Scanner complete! Results have been updated.")

      except Exception as _scan_top_err:
          import traceback
          st.error(f"❌ Scanner crashed: {_scan_top_err}")
          st.code(traceback.format_exc())
          print(f"[SCAN TOP-LEVEL ERROR] {traceback.format_exc()}")  


# Display Last Scanned Timestamp
if st.session_state.last_scanned:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Sync with Database", use_container_width=True, help="Force reload all results from the database to reflect background scans."):
        st.session_state['db_cache_checked'] = False
        st.rerun()

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
    try:
        # 1. Premium Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        
        if scan_data:
            total_scanned = st.session_state.total_scanned
            # If total_scanned is 0 or less than actual breakout count, use breakout count as floor
            # (scan_logs may not have the entry for this date)
            if total_scanned < len(scan_data):
                total_scanned = len(scan_data)
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
        # Show which date the data is from (critical transparency for cached results)
        _scan_date_display = st.session_state.get('scan_results_date') or st.session_state.get('last_scanned', '')
        if _scan_date_display and scan_data:
            st.info(f"📅 **Showing scan results from: {_scan_date_display}** — Refresh auto-loaded these from the database. Click **'Run Scanner'** in the sidebar to get today's fresh results.")

        # 2. Main Scan Table
        # NOTE: Use len(scan_data) not total_scanned==0 because DB-loaded results
        # have scan_data populated but total_scanned may be 0 from scan_logs.
        if not scan_data:
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
                try:
                    if pd.isnull(v):
                        return ""
                except (TypeError, ValueError):
                    pass
                if hasattr(v, 'strftime'):
                    return v.strftime("%Y-%m-%d")
                return str(v)

            export_rows = []
            for r in sorted_scan:
                export_rows.append({
                    "Symbol": r['symbol'],
                    "Sector": get_stock_sector(r['symbol']),
                    "CMP (₹)": r['cmp'],
                    "Setup": r.get('setup_type', 'VDU Breakout'),
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
            csv_data = export_df.to_csv(index=False).encode('utf-8-sig')
            
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
                prices_df = yf.download(tickers=tickers_list, period="1d", progress=False, threads=False, timeout=15)
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
            width="stretch",
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
                del_clicked = c_del2.button("Remove Ticker", type="secondary", key="del_action", width="stretch")
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
                width="stretch",
                key="dl_watchlist"
            )
            
            # Clear all database
            clear_btn = st.button("🗑️ Clear Entire Watchlist", type="secondary", width="stretch", key="clear_watchlist_btn")
            if clear_btn:
                st.session_state.confirm_clear = True
                
            if st.session_state.confirm_clear:
                st.markdown("<p style='color:#ef4444; font-weight:600;'>⚠️ Are you absolutely sure? This deletes watchlist.csv entries forever.</p>", unsafe_allow_html=True)
                col_yes, col_no = st.columns(2)
                
                if col_yes.button("Yes, Clear All", type="primary", width="stretch", key="clr_yes"):
                    # Clear CSV
                    empty_df = pd.DataFrame(columns=watchlist.COLUMNS)
                    watchlist.save_watchlist(empty_df)
                    st.session_state.confirm_clear = False
                    st.toast("🗑️ Watchlist fully cleared.")
                    st.rerun()
                    
                if col_no.button("Cancel", width="stretch", key="clr_no"):
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
                btn_force_batch_scan = btn_cols[1].button(f"🔄 Force Re-scan All ({len(active_flagged_symbols)} Flagged Candidates)", key="force_batch_ai_scan_action_btn", width="stretch")
                
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
    st.markdown("### 📈 Stocks Trading Above 20 SMA & 50 SMA")
    st.markdown("<p style='font-size:0.9rem; color:#94a3b8;'>Identify stocks in a strong medium-term uptrend where price is trading comfortably above both their 20-day and 50-day Simple Moving Averages.</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        sma_timeframe_tab = st.selectbox("SMA Strategy Timeframe", ["Daily", "Weekly", "All (Daily + Weekly Convergence)"], index=0, key="sma_timeframe_tab")
    
    above_ma_data = st.session_state.above_ma_results
    
    if above_ma_data is None:
        st.info("💡 Run the scanner from the sidebar to identify stocks trading above their 20 SMA and 50 SMA.")
    else:
        # Dynamically filter based on the UI dropdown
        filtered_above = []
        for r in above_ma_data:
            if matches_sma_timeframe_filter(r, sma_timeframe_tab):
                filtered_above.append(r)
                
        if len(filtered_above) == 0:
            st.info(f"ℹ️ No stocks found today matching the '{sma_timeframe_tab}' 20 & 50 SMA uptrend criteria.")
        else:
            # Sort by day change descending
            sorted_above = sorted(filtered_above, key=lambda x: x.get('day_change_pct', 0.0), reverse=True)
            
            # Download results option
            export_above = []
            for r in sorted_above:
                export_above.append({
                    "Symbol": r['symbol'],
                    "Sector": get_stock_sector(r['symbol']),
                    "CMP (₹)": r['cmp'],
                    "Day Change %": r['day_change_pct'],
                    "Setup Type": r['setup_type'],
                    "Dist to 20 SMA (%)": float(r.get('dist_20sma_pct') or 0.0),
                    "Dist to 50 SMA (%)": float(r.get('dist_50sma_pct') or 0.0),
                    "Dist to 200 SMA (%)": float(r.get('dist_200sma_pct') or 0.0),
                    "Suggested Buy (₹)": r['buy_price'],
                    "Suggested Exit/SL (₹)": r['exit_price'],
                    "Suggested Target (₹)": r['target_price'],
                    "Confidence": r['confidence'],
                    "Recommendation": extract_clean_recommendation(r.get('recommendation', ''))
                })
            export_a_df = pd.DataFrame(export_above)
            csv_a_data = export_a_df.to_csv(index=False).encode('utf-8-sig')
            
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
with tab_sma65:
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
        csv_s_data = export_s_df.to_csv(index=False).encode('utf-8-sig')
        
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
with tab_macross:
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
        csv_x_data = export_x_df.to_csv(index=False).encode('utf-8-sig')
        
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
with tab_wave:
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
        default_wt_thresh = -20.0 if wt_timeframe in ["Weekly", "Monthly"] else -40.0
        wt_oversold_threshold = st.number_input(
            "📉 Oversold Threshold:",
            min_value=-100.0,
            max_value=0.0,
            value=default_wt_thresh,
            step=5.0,
            key=f"wt_oversold_threshold_{wt_timeframe}",
            help="Define the WT1 value below which a stock is considered oversold. Default is -40.0."
        )
        
    wt_cache_key = f"{wt_timeframe}_{wt_oversold_threshold}"
    
    # Reactive Loader
    if 'wt_results_by_tf' not in st.session_state:
        st.session_state.wt_results_by_tf = {}
        
    run_wt_btn = st.button("🌊 Run Advanced WaveTrend Scan", key="run_wt_scan_btn", width="stretch")
    
    if run_wt_btn:
        # Universe is hardcoded to Top 1000 NSE stocks
        universe_key = "TOP 1000"
        from data_fetcher import get_top1000_nse_symbols
        raw_symbols = get_top1000_nse_symbols()
            
        symbols_to_scan = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
        with st.spinner(f"Running Advanced WaveTrend {wt_timeframe} scan on Top 1000 ({len(symbols_to_scan)} stocks)..."):
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
                chunk_ns = [s if s.endswith('.NS') else f"{s}.NS" for s in chunk]
                try:
                    # yfinance 1.x: group_by, threads, auto_adjust=False removed
                    df_bulk = yf.download(tickers=chunk_ns, period=period, interval=interval, progress=False, threads=False, timeout=15)
                    for sym in chunk:
                        sym_ns = sym if sym.endswith('.NS') else f"{sym}.NS"
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
                                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)

                                    wt_res = scan_wt_cross(sym, ticker_df, wt_oversold_threshold=wt_oversold_threshold)
                                    if wt_res is not None:
                                        # Market cap filter: exclude stocks below ₹2000 Cr
                                        try:
                                            mcap = getattr(yf.Ticker(sym_ns).fast_info, 'market_cap', None) or 0
                                            mcap_cr = mcap / 1e7  # Convert to Crore
                                            if mcap_cr < 2000.0:
                                                continue
                                            wt_res['market_cap_cr'] = round(mcap_cr, 1)
                                        except Exception:
                                            pass  # Allow through if market cap fetch fails
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
            
    # Pick up background scan results if available
    if not st.session_state.wt_results_by_tf.get(wt_cache_key) and ALL_TAB_SCAN_STATUS["wt_results"] is not None:
        st.session_state.wt_results_by_tf["Daily_-40.0"] = ALL_TAB_SCAN_STATUS["wt_results"]
        st.session_state.wt_results = ALL_TAB_SCAN_STATUS["wt_results"]

    wt_data = st.session_state.wt_results_by_tf.get(wt_cache_key, None)
    
    st.markdown(f"### 🌊 WaveTrend Oversold Buy Signals ({wt_timeframe} Timeframe)")
    st.markdown(f"<p style='font-size:0.9rem; color:#94a3b8;'>Scan for stocks in the WaveTrend oversold zone (WT1 below {wt_oversold_threshold}) using LazyBear's WaveTrend with Crosses indicator. <span style=\"color:#ffa000; font-weight:600;\">Filters: Price ≥ ₹100 | Market Cap ≥ ₹2000 Cr</span>. Stocks showing a <b style=\"color:#00e676;\">green dot 🟢 buy signal</b> (WT1 crossing above WT2) in oversold territory are prime mean-reversion candidates.</p>", unsafe_allow_html=True)
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
    
    # Background scan progress indicator
    if wt_data is None and ALL_TAB_SCAN_STATUS["is_running"]:
        _bg_scanner = ALL_TAB_SCAN_STATUS["current_scanner"]
        _bg_status = ALL_TAB_SCAN_STATUS["status_text"]
        _bg_progress = ALL_TAB_SCAN_STATUS["progress"]
        st.markdown(f"""
        <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
            <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
            </h4>
            <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. WaveTrend results will appear here when ready!</p>
            <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
        </div>
        <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
        """, unsafe_allow_html=True)
        st.progress(_bg_progress)
        if st.button("🔄 Refresh Scanner Status", key="refresh_bg_wt_status_btn"):
            st.rerun()

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
            csv_wt_data = export_wt_df.to_csv(index=False).encode('utf-8-sig')
            
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
        sub_breakout, sub_gapup, sub_above_ma, sub_support_ma, sub_crossover_ma, sub_wt, sub_vp = st.tabs([
            "📊 VDU Breakouts",
            "🚀 Gap-Ups",
            "📈 Above 20 & 50 SMA",
            "🛡️ 65 SMA Support",
            "🔄 MA Crossovers",
            "🌊 Wave Trend",
            "📊 Volume Profile"
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
    st.markdown("### 📉 Dan Zanger Breakout Scanner")
    st.markdown("Identifies stocks meeting Dan Zanger's criteria: Uptrend stack, prior massive run, shallow base, and high-volume breakout.")
    
    st.markdown("#### Scanner Parameters (Tweak if 0 results)")
    col_z1, col_z2, col_z3 = st.columns(3)
    z_min_run_pct = col_z1.number_input("Min Prior Run (%)", value=25.0, step=5.0)
    z_hft_run_pct = col_z2.number_input("High-Tight Flag Run (%)", value=90.0, step=10.0)
    z_max_base_depth = col_z3.number_input("Max Base Depth (%)", value=25.0, step=5.0)
    
    col_z4, col_z5, col_z6 = st.columns(3)
    z_vol_mult = col_z4.number_input("Breakout Vol Multiplier", value=2.0, step=0.5)
    z_base_lookback = col_z5.number_input("Base Lookback (Bars)", value=15, step=5)
    z_hft_max_base = col_z6.number_input("HFT Max Base Depth (%)", value=20.0, step=5.0)

    st.markdown("---")
    z_require_uptrend = st.checkbox("Require Strict Uptrend (Close > 50MA > 150MA > 200MA)", value=True, help="Uncheck this if you want to find setups even when the stock is in a broad market correction/downtrend.")
    
    st.markdown("---")
    
    def on_zanger_tf_change():
        import database
        zanger_dates = database.get_zanger_scan_dates()
        if zanger_dates:
            try:
                st.session_state.zanger_results = database.get_cached_zanger(zanger_dates[0], timeframe=st.session_state.zanger_tf)
            except Exception:
                st.session_state.zanger_results = []
        else:
            st.session_state.zanger_results = []

    col_z7, _ = st.columns([2, 8])
    with col_z7:
        zanger_tf = st.selectbox("Scan Timeframe", ["Daily", "Weekly"], index=0, key="zanger_tf", on_change=on_zanger_tf_change)
        
    st.markdown("---")
    
    col_btn, col_note = st.columns([1, 2])
    run_zanger_btn = col_btn.button("🔍 Run Dan Zanger Scan", type="primary", use_container_width=True)
    
    if run_zanger_btn:
        with st.spinner("Running Dan Zanger scan..."):
            import yfinance as yf
            from zanger_scanner import ZangerConfig, scan_zanger, get_latest_signal, rank_signals
            
            # Universe is hardcoded to Top 1000 NSE stocks
            from data_fetcher import get_top1000_nse_symbols
            zanger_candidates = get_top1000_nse_symbols()
            
            zanger_results = []
            chunk_size = 50
            chunks = [zanger_candidates[i:i+chunk_size] for i in range(0, len(zanger_candidates), chunk_size)]
            cfg = ZangerConfig(
                min_run_pct=float(z_min_run_pct),
                hft_run_pct=float(z_hft_run_pct),
                max_base_depth_pct=float(z_max_base_depth),
                breakout_vol_mult=float(z_vol_mult),
                base_lookback=int(z_base_lookback),
                hft_max_base_depth_pct=float(z_hft_max_base),
                require_uptrend=z_require_uptrend
            )
            
            yf_interval = "1wk" if zanger_tf == "Weekly" else "1d"
            yf_period = "10y" if zanger_tf == "Weekly" else "1100d"
            
            if zanger_tf == "Weekly":
                cfg.ma_fast = 10     # 50 days = 10 weeks
                cfg.ma_slow = 30     # 150 days = 30 weeks
                cfg.ma_slowest = 40  # 200 days = 40 weeks
                cfg.base_lookback = max(3, cfg.base_lookback // 5)
            
            for chunk in chunks:
                tkrs = [f"{s}.NS" for s in chunk]
                try:
                    df_zanger = yf.download(tickers=tkrs, period=yf_period, interval=yf_interval, progress=False, threads=False, timeout=15)
                    if not df_zanger.empty:
                        for sym in chunk:
                            try:
                                if isinstance(df_zanger.columns, pd.MultiIndex):
                                    all_tkrs = df_zanger.columns.get_level_values(1).unique().tolist()
                                    matched_t = next((t for t in all_tkrs if t.upper() == f"{sym}.NS".upper()), None)
                                    if not matched_t:
                                        continue
                                    t_df = df_zanger.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                                else:
                                    t_df = df_zanger.dropna(subset=['Close'])
                                    
                                if not t_df.empty and len(t_df) > cfg.ma_slowest + 5:
                                    t_df = t_df.reset_index()
                                    if 'Date' not in t_df.columns:
                                        t_df.rename(columns={t_df.columns[0]: 'Date'}, inplace=True)
                                    t_df['Date'] = pd.to_datetime(t_df['Date'])
                                    t_df.set_index('Date', inplace=True)
                                    
                                    res_df = scan_zanger(t_df, cfg)
                                    latest = get_latest_signal(res_df)
                                    
                                    if latest.get("zanger_signal", False):
                                        from data_fetcher import get_stock_sector
                                        latest["symbol"] = sym
                                        latest["sector"] = get_stock_sector(sym)
                                        zanger_results.append(latest)
                            except Exception as e:
                                pass
                except Exception as e:
                    pass
            
            if len(zanger_results) > 0:
                import pandas as pd
                hits_df = pd.DataFrame(zanger_results)
                ranked_df = rank_signals(hits_df, cfg)
                # Convert back to dicts for session_state to be consistent
                st.session_state.zanger_results = ranked_df.to_dict('records')
                try:
                    database.save_zanger_scan(get_market_date(), zanger_tf, st.session_state.zanger_results)
                except Exception as e:
                    print(f"Error saving Dan Zanger scan: {e}")
                st.success(f"Dan Zanger Scan Complete! Found {len(zanger_results)} setups ({zanger_tf}).")
            else:
                st.session_state.zanger_results = []
                st.info("No Dan Zanger setups found today.")
                
    if st.session_state.get('zanger_results') is not None:
        if len(st.session_state.zanger_results) > 0:
            import pandas as pd
            z_df = pd.DataFrame(st.session_state.zanger_results)
            # Reorder columns to put rank and symbol first
            cols = list(z_df.columns)
            if 'date' in cols:
                # keep as string, Streamlit will be told it's text
                z_df['date'] = z_df['date'].astype(str).str[:10]
            
            # Clean up unwanted columns (like company_name if sector is there)
            if 'company_name' in cols and 'sector' in cols:
                cols.remove('company_name')
                z_df = z_df.drop(columns=['company_name'])
                
            if 'rank' in cols and 'symbol' in cols:
                cols.insert(0, cols.pop(cols.index('rank')))
                cols.insert(1, cols.pop(cols.index('symbol')))
                if 'sector' in cols:
                    cols.insert(2, cols.pop(cols.index('sector')))
                if 'score' in cols:
                    cols.insert(3, cols.pop(cols.index('score')))
                if 'confidence_level' in cols:
                    cols.insert(4, cols.pop(cols.index('confidence_level')))
                if 'breakout_status' in cols:
                    cols.insert(5, cols.pop(cols.index('breakout_status')))
                if 'target_price' in cols:
                    risk_idx = cols.index('risk_pct') if 'risk_pct' in cols else len(cols)
                    cols.insert(risk_idx + 1, cols.pop(cols.index('target_price')))
                z_df = z_df[cols]
                
            if 'symbol' in z_df.columns:
                z_df['symbol'] = z_df['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            
            st.dataframe(z_df, use_container_width=True, column_config={
                "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)"),
                "date": st.column_config.TextColumn("Date"),
                "score": st.column_config.NumberColumn("Score (Out of 100)", format="%.1f"),
                "confidence_level": st.column_config.TextColumn("Confidence Level"),
                "breakout_status": st.column_config.TextColumn("Breakout Status")
            })
        else:
            st.info("No Dan Zanger setups found.")
    else:
        st.info("💡 Click 'Run Dan Zanger Scan' to find breakouts.")

# ==============================================================================
# TAB: VCP+Minervini
# ==============================================================================
with tab_vcp:
    st.markdown("### 🎯 Minervini Ultimate +VCP")
    st.markdown("Scans for stocks passing the Minervini Trend Template with Volatility Contraction Pattern (VCP) squeeze.")

    col_v1, col_v2, col_v3 = st.columns(3)
    vcp_thresh = col_v1.number_input("Max VCP Range (%)", value=2.5, step=0.5, help="Maximum percentage range over lookback to be considered a squeeze")
    vcp_lookback = col_v2.number_input("VCP Lookback (Bars)", value=5, step=1)
    risk_low = col_v3.number_input("Max Low Risk (%)", value=15.0, step=1.0, help="Maximum distance above 50SMA to be considered Low Risk")

    st.markdown("---")
    
    col_v_btn, col_v_note = st.columns([1, 2])
    run_vcp_btn = col_v_btn.button("🔍 Run VCP+Minervini Scan", type="primary", use_container_width=True)
    
    if run_vcp_btn:
        with st.spinner("Running Minervini VCP scan across all NSE stocks..."):
            from vcp_minervini import VCPConfig, MinerviniVCPAnalyzer
            from data_fetcher import get_all_nse_symbols, get_stock_sector
            import yfinance as yf
            import pandas as pd
            
            # Use all NSE symbols (capped at 1800)
            raw_symbols = get_all_nse_symbols()
            symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
            
            cfg = VCPConfig(
                vcp_thresh=vcp_thresh,
                vcp_lookback=int(vcp_lookback),
                risk_low=risk_low
            )
            
            # Download benchmark for RS Proxy calculation
            try:
                benchmark_df = yf.download("^NSEI", period="2y", interval="1d", progress=False, threads=False, timeout=15)
            except Exception:
                benchmark_df = None
                
            # Batch download data for speed, then run analyzer per-stock
            vcp_results = []
            chunk_size = 100
            sym_chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]
            progress_bar = st.progress(0, text="Downloading data...")
            
            for c_idx, chunk in enumerate(sym_chunks):
                progress_bar.progress(
                    (c_idx + 1) / len(sym_chunks),
                    text=f"Processing chunk {c_idx+1}/{len(sym_chunks)} ({len(vcp_results)} setups found)..."
                )
                try:
                    bulk_df = yf.download(tickers=chunk, period="2y", interval="1d", progress=False, threads=False, timeout=15)
                    if bulk_df.empty:
                        continue
                    
                    for sym in chunk:
                        try:
                            # Extract single-stock data from bulk download
                            if isinstance(bulk_df.columns, pd.MultiIndex):
                                all_tkrs = bulk_df.columns.get_level_values(1).unique().tolist()
                                matched_t = next((t for t in all_tkrs if t.upper() == sym.upper()), None)
                                if not matched_t:
                                    continue
                                t_df = bulk_df.xs(matched_t, axis=1, level=1).dropna(subset=['Close'])
                            else:
                                t_df = bulk_df.dropna(subset=['Close'])
                            
                            if t_df.empty or len(t_df) < 250:
                                continue
                            
                            # Run the Minervini analyzer with pre-fetched data
                            analyzer = MinerviniVCPAnalyzer(sym, cfg, benchmark_df=benchmark_df)
                            analyzer.df = t_df.copy()
                            analyzer._moving_averages()
                            analyzer._trend_template()
                            analyzer._buy_risk()
                            analyzer._pressure()
                            analyzer._relative_price_strength()
                            analyzer._vcp()
                            analyzer._entry_signals()
                            
                            last = analyzer.df.iloc[-1]
                            result = {
                                "symbol": sym,
                                "date": analyzer.df.index[-1].strftime("%Y-%m-%d"),
                                "close": round(float(last["Close"]), 2),
                                "Pressure": last["pressure_txt"],
                                "Risk (50d)": last["risk_status"],
                                "Trend (TPR)": last["tpr_txt"],
                                "RS Proxy": round(float(last["rpr_proxy"]), 1) if pd.notna(last["rpr_proxy"]) else None,
                                "VCP (5d)": last["vcp_txt"],
                                "VCP range %": round(float(last["vcp_range_pct"]), 2) if pd.notna(last["vcp_range_pct"]) else None,
                                "VCP (10d)": last.get("vcp10_txt", "Normal"),
                                "VCP 10d range %": round(float(last["vcp10_range_pct"]), 2) if "vcp10_range_pct" in last and pd.notna(last["vcp10_range_pct"]) else None,
                                "VCP (15d)": last.get("vcp15_txt", "Normal"),
                                "VCP 15d range %": round(float(last["vcp15_range_pct"]), 2) if "vcp15_range_pct" in last and pd.notna(last["vcp15_range_pct"]) else None,
                                "Entry Signal": last["entry_signal"],
                            }
                            
                            result["Sector"] = get_stock_sector(sym)
                            vcp_results.append(result)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            progress_bar.empty()
            
            if vcp_results:
                # Add score and rank
                vcp_df = pd.DataFrame(vcp_results)
                rs_proxy = pd.to_numeric(vcp_df.get('RS Proxy', 50), errors='coerce').fillna(50)
                vcp_range = pd.to_numeric(vcp_df.get('VCP range %', 100), errors='coerce').fillna(100)
                vcp_df['Score'] = rs_proxy - (vcp_range * 5)
                vcp_df = vcp_df.sort_values(by='Score', ascending=False)
                vcp_df.insert(0, 'Rank', range(1, len(vcp_df) + 1))
                st.session_state.vcp_minervini_results = vcp_df.to_dict('records')
                
                # Save to database
                try:
                    today_str = get_market_date()
                    database.save_vcp_minervini_scan(today_str, st.session_state.vcp_minervini_results)
                except Exception as e:
                    print(f"Error saving VCP+Minervini scan: {e}")
            else:
                st.session_state.vcp_minervini_results = []
            
            st.rerun()

    # Display results
    if st.session_state.get('vcp_minervini_results'):
        import pandas as pd
        v_df = pd.DataFrame(st.session_state.vcp_minervini_results)
        
        vcp_count = len(v_df)
        st.success(f"Found {vcp_count} VCP+Minervini setups!")
        
        # CSV Download
        col_btn, _ = st.columns([2, 8])
        with col_btn:
            vcp_csv = v_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ Download CSV",
                data=vcp_csv,
                file_name="vcp_minervini_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        col1, col2 = st.columns(2)
        with col1:
            show_buyable = st.checkbox("Show Buyable Only (Buying Pressure, Low Risk, PASSED Trend)", value=False)
        with col2:
            show_squeeze = st.checkbox("Show 'Squeeze' / Entry Signals Only", value=True)

        # Reorder columns for display
        display_cols = ['Rank', 'Score', 'symbol', 'Sector', 'close', 'Entry Signal', 'Trend (TPR)', 
                       'Pressure', 'Risk (50d)', 'RS Proxy', 'VCP (5d)', 'VCP range %', 'VCP (10d)', 'VCP 10d range %', 'VCP (15d)', 'VCP 15d range %', 'date']
        available_cols = [c for c in display_cols if c in v_df.columns]
        v_df = v_df[available_cols]
        
        if show_buyable:
            if 'Pressure' in v_df.columns:
                v_df = v_df[v_df['Pressure'].str.contains('Buying', case=False, na=False)]
            if 'Risk (50d)' in v_df.columns:
                v_df = v_df[v_df['Risk (50d)'].str.contains('Low Risk', case=False, na=False)]
            if 'Trend (TPR)' in v_df.columns:
                v_df = v_df[v_df['Trend (TPR)'].str.contains('PASSED', case=False, na=False)]
                
        if show_squeeze:
            if 'Entry Signal' in v_df.columns:
                mask = v_df['Entry Signal'].isin(["BREAKOUT", "EARLY ENTRY"])
                if 'VCP (5d)' in v_df.columns: mask = mask | (v_df['VCP (5d)'] == 'SQUEEZE')
                if 'VCP (10d)' in v_df.columns: mask = mask | (v_df['VCP (10d)'] == 'SQUEEZE')
                if 'VCP (15d)' in v_df.columns: mask = mask | (v_df['VCP (15d)'] == 'SQUEEZE')
                v_df = v_df[mask]
        
        if 'symbol' in v_df.columns:
            v_df['symbol'] = v_df['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            
        st.dataframe(v_df, use_container_width=True, column_config={
            "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)"),
            "date": st.column_config.TextColumn("Date"),
            "close": st.column_config.NumberColumn("CMP", format="%.2f"),
            "Score": st.column_config.NumberColumn("Score", format="%.1f"),
            "RS Proxy": st.column_config.NumberColumn("RS Proxy (vs Nifty)", format="%.1f"),
            "VCP range %": st.column_config.NumberColumn("VCP Range 5d %", format="%.2f"),
            "VCP 10d range %": st.column_config.NumberColumn("VCP Range 10d %", format="%.2f"),
            "VCP 15d range %": st.column_config.NumberColumn("VCP Range 15d %", format="%.2f"),
            "Entry Signal": st.column_config.TextColumn("Entry Signal"),
            "Trend (TPR)": st.column_config.TextColumn("Trend Template"),
        })
    else:
        st.info("💡 Click 'Run VCP+Minervini Scan' to find setups.")

# ==============================================================================
# TAB: EARLY STAGE 2 BREAKOUT
# ==============================================================================
with tab_stage2:
    st.markdown("### 🚀 Early Stage 2 Base Breakout Scanner")
    st.markdown("Identifies stocks moving out of a long-term Stage 1 base on the monthly timeframe.")
    
    if 'stage2_results' not in st.session_state:
        st.session_state.stage2_results = None

    # Pick up background scan results if available
    if st.session_state.stage2_results is None and ALL_TAB_SCAN_STATUS["stage2_results"] is not None:
        st.session_state.stage2_results = ALL_TAB_SCAN_STATUS["stage2_results"]
        # Try loading from DB
        today_str = get_market_date(for_display=True)
        try:
            cached_stage2 = database.get_cached_stage2(today_str)
            if cached_stage2 is not None:
                st.session_state.stage2_results = cached_stage2
                # Note: No need to show success message on silent load, just let the table render
        except Exception as e:
            print(f"Failed to load cached stage2: {e}")
        
    s2_col1, s2_col2 = st.columns([2, 8])
    with s2_col1:
        s2_max_runup = st.number_input("Max Run-Up (%)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
        run_stage2_btn = st.button("🔍 Run Stage 2 Scan", width="stretch", type="primary")
        
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
                    df_s2 = yf.download(tickers=tkrs, period="5y", interval="1mo", progress=False, threads=False, timeout=15)
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
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
                today_ist_str = get_market_date()
                database.save_stage2_only(today_ist_str, s2_res)
            except Exception as e:
                print(f"Failed to cache stage2 scan: {e}")
            st.success(f"Stage 2 Scan Complete! Found {len(s2_res)} setups.")
            
    st.markdown("---")
    
    if st.session_state.stage2_results is None:
        # Background scan progress indicator
        if ALL_TAB_SCAN_STATUS["is_running"]:
            _bg_scanner = ALL_TAB_SCAN_STATUS["current_scanner"]
            _bg_status = ALL_TAB_SCAN_STATUS["status_text"]
            _bg_progress = ALL_TAB_SCAN_STATUS["progress"]
            st.markdown(f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
                <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. Stage-2 results will appear here when ready!</p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """, unsafe_allow_html=True)
            st.progress(_bg_progress)
            if st.button("🔄 Refresh Scanner Status", key="refresh_bg_s2_status_btn"):
                st.rerun()
                
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
            csv_data = s2_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name="stage2_scan_results.csv",
                mime="text/csv",
                width="stretch"
            )
        render_unified_strategy_table(st.session_state.stage2_results, "stage2", "stage2_tab")



# ==============================================================================
# TAB 17: VPA TREND
# ==============================================================================
with tab_vpa:
    st.markdown("### 🚥 VPA Trend Indicator (Daily, Weekly, Monthly)")
    st.info("Scans ALL NSE listed stocks. Filters: Price > ₹100. Shows Major, Mid, and Minor trends across timeframes.")
    
    # Pick up background scan results if available
    if not st.session_state.get('vpa_results') and ALL_TAB_SCAN_STATUS["vpa_results"] is not None:
        st.session_state.vpa_results = ALL_TAB_SCAN_STATUS["vpa_results"]

    col1, col2 = st.columns([3, 7])
    with col1:
        run_vpa_btn = st.button("🚀 Run Advanced VPA Scan", width="stretch")
    
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
                    today_ist_str = get_market_date()
                    database.save_vpa_only(today_ist_str, vpa_list)
                except Exception as e:
                    print(f"Failed to cache custom VPA scan: {e}")
                st.success(f"VPA Scan complete! Found {len(vpa_list)} stocks meeting all criteria and saved to database.")
                
            except Exception as e:
                st.error(f"Scan failed: {e}")
                
    if not st.session_state.get('vpa_results'):
        # Background scan progress indicator
        if ALL_TAB_SCAN_STATUS["is_running"]:
            _bg_scanner = ALL_TAB_SCAN_STATUS["current_scanner"]
            _bg_status = ALL_TAB_SCAN_STATUS["status_text"]
            _bg_progress = ALL_TAB_SCAN_STATUS["progress"]
            st.markdown(f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
                <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. VPA results will appear here when ready!</p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """, unsafe_allow_html=True)
            st.progress(_bg_progress)
            if st.button("🔄 Refresh Scanner Status", key="refresh_bg_vpa_status_btn"):
                st.rerun()
                
        st.info("No VPA data available. Click 'Run Advanced VPA Scan' to process.")
    else:
        vpa_data = st.session_state.vpa_results
        
        # Sort by score
        vpa_data = sorted(vpa_data, key=lambda x: x.get('score', 0), reverse=True)
        
        # Download Button
        import pandas as pd
        
        def get_action_signal_text(short, mid, max_t, max_val, rsi=0, cci=0):
            # Issue #3: RSI/CCI overbought guard
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

        only_buy_signals = st.checkbox("🟢 Show Only 'Buy' Signals", value=False)
        
        # Issue #2: Move timeframe selection BEFORE filter so we can filter by the correct timeframe
        st.markdown("### Select Timeframe")
        selected_tf = st.selectbox("Timeframe to display", ["Daily", "Weekly", "Monthly"], key="vpa_tf_select")
        
        daily_export = []
        weekly_export = []
        monthly_export = []
        
        rank = 1
        filtered_vpa_data = []
        for r in vpa_data:
            d = r['daily']; w = r['weekly']; m = r['monthly']
            
            # Issue #2: Filter by the SELECTED timeframe signal, not always daily
            if selected_tf == "Weekly":
                tf_data = w
            elif selected_tf == "Monthly":
                tf_data = m
            else:
                tf_data = d
            
            tf_sig = get_signal(tf_data['minor'], tf_data['mid'], tf_data['major'], tf_data.get('major_val', 0))
            if only_buy_signals and tf_sig != "Buy":
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
                'Action': get_action_signal_text(d['minor'], d['mid'], d['major'], d.get('major_val', 0), rsi=d.get('rsi', 0), cci=d.get('cci', 0)),
                'Signal': get_signal(d['minor'], d['mid'], d['major'], d.get('major_val', 0)),
                'Score': r.get('score', 0),
                'Confidence': r.get('confidence', 'N/A')
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
                'Action': get_action_signal_text(w['minor'], w['mid'], w['major'], w.get('major_val', 0), rsi=w.get('rsi', 0), cci=w.get('cci', 0)),
                'Signal': get_signal(w['minor'], w['mid'], w['major'], w.get('major_val', 0)),
                'Score': r.get('score', 0),
                'Confidence': r.get('confidence', 'N/A')
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
                'Action': get_action_signal_text(m['minor'], m['mid'], m['major'], m.get('major_val', 0), rsi=m.get('rsi', 0), cci=m.get('cci', 0)),
                'Signal': get_signal(m['minor'], m['mid'], m['major'], m.get('major_val', 0)),
                'Score': r.get('score', 0),
                'Confidence': r.get('confidence', 'N/A')
            })
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download Daily VPA (CSV)",
                data=pd.DataFrame(daily_export).to_csv(index=False).encode('utf-8-sig'),
                file_name="vpa_daily_trend.csv",
                mime="text/csv",
                width="stretch"
            )
        with col2:
            st.download_button(
                label="📥 Download Weekly VPA (CSV)",
                data=pd.DataFrame(weekly_export).to_csv(index=False).encode('utf-8-sig'),
                file_name="vpa_weekly_trend.csv",
                mime="text/csv",
                width="stretch"
            )
        with col3:
            st.download_button(
                label="📥 Download Monthly VPA (CSV)",
                data=pd.DataFrame(monthly_export).to_csv(index=False).encode('utf-8-sig'),
                file_name="vpa_monthly_trend.csv",
                mime="text/csv",
                width="stretch"
            )
        
        # Timeframe selection was moved above the filter loop
        # selected_tf is already defined above
        
        def trend_to_badge(t_val):
            if t_val == 1:
                return "<span style='color: #00e676; font-weight: bold;'>Up (1)</span>"
            elif t_val == -1:
                return "<span style='color: #ef4444; font-weight: bold;'>Dn (-1)</span>"
            return "<span style='color: #fbbf24; font-weight: bold;'>Neu (0)</span>"
            
        def get_action_signal(short, mid, max_t, max_val, rsi=0, cci=0):
            text = get_action_signal_text(short, mid, max_t, max_val, rsi=rsi, cci=cci)
            if "Perfect Buy" in text:
                return f"<span style='color: #00e676; font-weight: bold;'>🟢 {text}</span>"
            elif "Counter Trend Buy" in text:
                return f"<span style='color: #4ade80; font-weight: bold;'>🟢 {text}</span>"
            elif "Early Breakout" in text:
                return f"<span style='color: #3b82f6; font-weight: bold;'>🔵 {text}</span>"
            elif "Pullback" in text:
                return f"<span style='color: #fbbf24; font-weight: bold;'>🟡 {text}</span>"
            elif "Warning" in text:
                return f"<span style='color: #f97316; font-weight: bold;'>🟠 {text}</span>"
            elif "Avoid" in text:
                return f"<span style='color: #ef4444; font-weight: bold;'>🔴 {text}</span>"
            elif "Parabolic" in text or "Overextended" in text:
                return f"<span style='color: #d946ef; font-weight: bold;'>🟣 {text}</span>"
            else:
                return f"<span style='color: #9ca3af; font-weight: bold;'>⚪ {text}</span>"
            
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
            
            action = get_action_signal(tf_data['minor'], tf_data['mid'], tf_data['major'], tf_data.get('major_val', 0), rsi=tf_data.get('rsi', 0), cci=tf_data.get('cci', 0))
            
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
<td style="padding: 10px; font-weight: bold; color: #a3e635;">{r.get('score', 0)}</td>
<td style="padding: 10px;">{r.get('confidence', 'N/A')}</td>
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
<th style="padding: 10px;">Score</th>
<th style="padding: 10px;">Confidence</th>
</tr>
</thead>
<tbody>
{rows_str}
</tbody>
</table>
</div>"""
        st.markdown(table_html, unsafe_allow_html=True)

# TAB: VPA SQUEEZE
with tab_vpa_squeeze:
    st.markdown("### 📉 VPA Green + MA Squeeze")
    st.info("Finds stocks where VPA is Green (Minor, Mid, Major) and the 10/21/50 SMA are tightly clustered (<6% gap).")
    
    if "vpa_squeeze_results" not in st.session_state:
        st.session_state.vpa_squeeze_results = []
        
    run_vpa_squeeze_btn = st.button("🚀 Run VPA Squeeze Scan", width="stretch")
    if run_vpa_squeeze_btn:
        st.session_state.vpa_squeeze_results = []
        with st.spinner("Running VPA Squeeze Scan..."):
            try:
                from data_fetcher import get_all_nse_symbols
                from scanner import scan_vpa_ma_squeeze
                from local_cache_manager import bulk_get_cached_ohlcv
                import pandas as pd

                import database
                
                raw_symbols = get_all_nse_symbols()
                symbols_to_scan = [s.strip().upper().replace('.NS', '') for s in raw_symbols if str(s).strip()]
                
                st.info(f"Step 1/2 — Fetching history from Master Cache...")
                prog = st.progress(0)
                status = st.empty()
                
                # Fetch all cached data at once
                bulk_cached = bulk_get_cached_ohlcv(symbols_to_scan, "1d")
                
                st.info("Step 2/2 — Calculating MA Squeeze (Instant)...")
                status.empty()
                prog.progress(1.0)
                
                results = []
                for sym, df in bulk_cached.items():
                    res = scan_vpa_ma_squeeze(sym, df)
                    if res is not None:
                        results.append(res)
                            
                prog.empty()
                status.empty()
                st.session_state.vpa_squeeze_results = results
                
                # Save to database
                today_str = get_market_date()
                database.save_vpa_squeeze_only(today_str, results)
                
                st.success(f"Scan complete! Found {len(results)} matches.")
                
            except Exception as e:
                st.error(f"Error running scan: {e}")

    if st.session_state.get('vpa_squeeze_results'):
        results = st.session_state.vpa_squeeze_results
        df_res = pd.DataFrame(results)
        df_res['symbol'] = df_res['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
        st.write(f"### Found {len(results)} stocks")
        # Sort by Compression Score (Gap %) ascending so tightest squeezes appear first
        if 'ma_gap_pct' in df_res.columns:
            df_res = df_res.sort_values('ma_gap_pct', ascending=True)
        display_cols = ['symbol', 'cmp', 'day_change_pct', 'sma10', 'sma21', 'sma50', 'ma_gap_pct', 'dist_to_200_pct']
        # Keep only columns that actually exist (handles older cached results without the new field)
        display_cols = [c for c in display_cols if c in df_res.columns]
        st.dataframe(
            df_res[display_cols],
            use_container_width=True,
            column_config={
                "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)", width="small"),
                "cmp": st.column_config.NumberColumn("CMP (₹)", width="small"),
                "day_change_pct": st.column_config.NumberColumn("Chg %", width="small"),
                "sma10": st.column_config.NumberColumn("10 SMA", width="small"),
                "sma21": st.column_config.NumberColumn("21 SMA", width="small"),
                "sma50": st.column_config.NumberColumn("50 SMA", width="small"),
                "ma_gap_pct": st.column_config.NumberColumn("Gap %", help="(Max SMA − Min SMA) / Min SMA × 100. Lower = tighter.", width="small"),
                "compression_score": st.column_config.NumberColumn(
                    "Compression Score",
                    help="Max(10,21,50 SMA) − Min(10,21,50 SMA) / Min(10,21,50 SMA). Lower = tighter squeeze = stronger breakout candidate.",
                    format="%.2f %%",
                    width="small"
                ),
                "dist_to_200_pct": st.column_config.NumberColumn("200 Dist %", width="small")
            },
            hide_index=True
        )

# --- NEAR 30 SMA TAB ---
with tab_near_30sma:
    st.markdown("### 📉 Near 30 SMA")
    st.info("Finds stocks where the price is just above the 30-day SMA, but not more than 3% above it.")
    
    if not st.session_state.get('near_30sma_results') and ALL_TAB_SCAN_STATUS.get("near_30sma_results") is not None:
        st.session_state.near_30sma_results = ALL_TAB_SCAN_STATUS["near_30sma_results"]
        
    # Auto load from DB if missing in session
    if 'near_30sma_results' not in st.session_state or not st.session_state.near_30sma_results:
        try:
            today_str = get_market_date()
            import database
            db_res = database.get_cached_near_30sma(today_str)
            if db_res:
                st.session_state.near_30sma_results = db_res
        except Exception as e:
            st.error(f"Failed to load DB results: {e}")
            
    if st.session_state.get('near_30sma_results'):
        results = st.session_state.near_30sma_results
        df_res = pd.DataFrame(results)
        df_res['symbol'] = df_res['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
        st.write(f"### Found {len(results)} stocks")
        st.dataframe(
            df_res,
            use_container_width=True,
            column_config={
                "symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)", width="small"),
                "company_name": "Company",
                "cmp": st.column_config.NumberColumn("CMP (₹)", width="small"),
                "day_change_pct": st.column_config.NumberColumn("Chg %", width="small"),
                "sma30": st.column_config.NumberColumn("30 SMA (₹)", width="small"),
                "dist_pct": st.column_config.NumberColumn("Dist to SMA %", format="%.2f %%", width="small"),
                "volume": st.column_config.NumberColumn("Volume", width="small")
            },
            hide_index=True
        )
    else:
        st.info("No stocks found matching the Near 30 SMA criteria. Run the scanner.")

# TAB: FREQUENT FLYERS (CONSISTENT ALERTS)
with tab_alerts:
    import tabs.tab_frequent as tab_freq_mod
    tab_freq_mod.render()


# --- VOLUME PROFILE SCANNER TAB ---
with tab_volprofile:
    st.markdown("### 📊 Volume Profile Zones (Daily, Weekly, Monthly)")
    st.info("Scans ALL NSE listed stocks for POC, VAH, VAL levels. Filters: Price > ₹100, Market Cap > 2000 Cr.")
    
    # Auto-load cached results from database on first visit
    # Pick up background scan results if available
    if not st.session_state.get('vp_results') and ALL_TAB_SCAN_STATUS["vp_results"] is not None:
        st.session_state.vp_results = ALL_TAB_SCAN_STATUS["vp_results"]

    if 'vp_results' not in st.session_state or not st.session_state.vp_results:
        try:
            # Try loading today's cached results first, then search last 10 days
            from datetime import timedelta
            for days_back in range(10):
                check_date = (datetime.now(IST_TIMEZONE) - timedelta(days=days_back)).strftime("%Y-%m-%d")
                cached = database.get_cached_volume_profile(check_date)
                if cached:
                    st.session_state.vp_results = cached
                    st.caption(f"📅 Loaded cached results from {check_date}")
                    break
        except Exception as e:
            print(f"Failed to auto-load VP cache: {e}")
    
    col1, col2 = st.columns([3, 7])
    with col1:
        run_vp_btn = st.button("🚀 Run Advanced Volume Profile Scan", width="stretch")
    
    if run_vp_btn:
        st.session_state.vp_results = []
        with st.spinner("Initializing Volume Profile Scan on ALL NSE Stocks..."):
            try:
                vp_list = []
                scan_progress = st.progress(0)
                status_text = st.empty()
                
                from data_fetcher import get_all_nse_symbols
                import yfinance as yf
                import pandas as pd
                import concurrent.futures
                from scanner import scan_volume_profile
                
                raw_symbols = get_all_nse_symbols()
                all_symbols = [s if s.endswith('.NS') else f"{s}.NS" for s in raw_symbols if str(s).strip()]
                
                total_symbols = len(all_symbols)
                
                # Phase 1: Bulk OHLCV Download (10 workers, 200 per chunk)
                status_text.text(f"Phase 1: Downloading 2 years of history for {total_symbols} stocks...")
                chunk_size = 200
                sym_chunks = [all_symbols[i:i + chunk_size] for i in range(0, len(all_symbols), chunk_size)]
                
                valid_data = {}
                
                def download_vp_chunk(chunk_idx, chunk):
                    chunk_data = {}
                    try:
                        df_bulk = yf.download(tickers=chunk, period="2y", interval="1d", progress=False, threads=False, timeout=15)
                        if isinstance(df_bulk.columns, pd.MultiIndex):
                            for sym in chunk:
                                try:
                                    if 'Close' in df_bulk.columns.levels[0]:
                                        ticker_df = df_bulk.xs(sym, axis=1, level=1).copy()
                                        ticker_df = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
                                        if len(ticker_df) >= 100:
                                            ticker_df = ticker_df.reset_index()
                                            ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                            ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)
                                            chunk_data[sym] = ticker_df
                                except Exception:
                                    pass
                        else:
                            if len(chunk) == 1 and not df_bulk.empty and 'Close' in df_bulk:
                                ticker_df = df_bulk[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
                                if len(ticker_df) >= 100:
                                    ticker_df = ticker_df.reset_index()
                                    ticker_df.rename(columns={ticker_df.columns[0]: 'Date'}, inplace=True)
                                    ticker_df['Date'] = pd.to_datetime(ticker_df['Date'], utc=True).dt.tz_localize(None)
                                    chunk_data[chunk[0]] = ticker_df
                    except Exception:
                        pass
                    return chunk_data
                    
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                    futures = []
                    for chunk_idx, chunk in enumerate(sym_chunks):
                        futures.append(executor.submit(download_vp_chunk, chunk_idx, chunk))
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            res_data = future.result(timeout=120)
                            valid_data.update(res_data)
                        except Exception:
                            pass
                        scan_progress.progress((i + 1) / len(sym_chunks) * 0.5)
                        status_text.text(f"Phase 1: Downloading history... ({i+1}/{len(sym_chunks)} chunks, {len(valid_data)} stocks loaded)")

                # Phase 2: Compute Volume Profile (simple sequential — fast after numpy optimization)
                status_text.text(f"Phase 2: Computing Volume Profiles for {len(valid_data)} stocks...")
                
                total_to_process = len(valid_data)
                done_count = 0
                if total_to_process == 0:
                    status_text.text("Scan Complete! Found 0 matches.")
                    scan_progress.progress(1.0)
                else:
                    for sym, df in valid_data.items():
                        done_count += 1
                        try:
                            res = scan_volume_profile(sym, df, 0)
                            if res:
                                vp_list.append(res)
                        except Exception:
                            pass
                        
                        if done_count % 50 == 0 or done_count == total_to_process:
                            scan_progress.progress(0.5 + (done_count / total_to_process) * 0.5)
                            status_text.text(f"Scanning Profiles: {done_count}/{total_to_process} | Found: {len(vp_list)}")
                    
                    scan_progress.progress(1.0)
                    status_text.text(f"Scan Complete! Found {len(vp_list)} matches.")
                
                if vp_list:
                    st.session_state.vp_results = vp_list
                    try:
                        today_ist_str = get_market_date()
                        database.save_volume_profile_only(today_ist_str, vp_list)
                    except Exception as e:
                        print(f"Failed to cache Volume Profile scan: {e}")
                    st.success(f"Volume Profile Scan complete! Found {len(vp_list)} stocks.")
                    
            except Exception as e:
                st.error(f"Scan failed: {e}")
                
    if not st.session_state.get('vp_results'):
        # Background scan progress indicator
        if ALL_TAB_SCAN_STATUS["is_running"]:
            _bg_scanner = ALL_TAB_SCAN_STATUS["current_scanner"]
            _bg_status = ALL_TAB_SCAN_STATUS["status_text"]
            _bg_progress = ALL_TAB_SCAN_STATUS["progress"]
            st.markdown(f"""
            <div class="glass-card" style="padding:22px; border:1px solid rgba(0,229,255,0.25); background:rgba(9,13,22,0.6); border-radius:12px; margin-bottom:20px; box-shadow:0 8px 32px 0 rgba(0,0,0,0.37);">
                <h4 style="color:#00e5ff; margin:0 0 10px 0; display:flex; align-items:center; gap:8px;">
                    <span style="display:inline-block; animation: spin 2s linear infinite;">🔄</span> Background All-Tab Scan Active...
                </h4>
                <p style="font-size:0.9rem; color:#94a3b8; margin:0 0 15px 0;">All scanners are running automatically in the background. Volume Profile results will appear here when ready!</p>
                <div style="font-size:0.85rem; color:#e2e8f0; font-weight:600; margin-bottom:8px;">Current: <span style="color:#00e5ff;">{_bg_status}</span></div>
            </div>
            <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
            """, unsafe_allow_html=True)
            st.progress(_bg_progress)
            if st.button("🔄 Refresh Scanner Status", key="refresh_bg_vp_status_btn"):
                st.rerun()
                
        st.info("No Volume Profile data available. Click 'Run Advanced Volume Profile Scan' to process.")
    else:
        vp_data = st.session_state.vp_results
        
        # Helper to safely extract VP level data from a timeframe dict
        def _get_tf(r, tf_key):
            tf = r.get(tf_key)
            if isinstance(tf, dict) and tf:
                return {
                    'zone': tf.get('zone', ''),
                    'va_pct': tf.get('position_pct') if tf.get('position_pct') is not None and tf.get('position_pct') != '' else None,
                    'poc': round(tf['poc'], 2) if tf.get('poc') is not None else None,
                    'val': round(tf['val'], 2) if tf.get('val') is not None else None,
                    'vah': round(tf['vah'], 2) if tf.get('vah') is not None else None
                }
            return {'zone': '', 'va_pct': None, 'poc': None, 'val': None, 'vah': None}
        
        # Format for Dataframe
        import pandas as pd
        vp_export = []
        rank = 1
        
        for r in vp_data:
            d = _get_tf(r, 'daily')
            w = _get_tf(r, 'weekly')
            m = _get_tf(r, 'monthly')
            
            clean_sym = str(r.get('symbol', '')).replace('.NS', '').strip().upper()
            
            vp_export.append({
                'Rank': rank,
                'Symbol': clean_sym,
                'CMP': r.get('cmp', 0),
                'Market Cap (Cr)': round(r.get('market_cap_cr', 0), 2),
                # Daily levels
                'D Zone': d['zone'],
                'D Buy Range (VAL)': d['val'],
                'D Target (POC)': d['poc'],
                'D Resistance (VAH)': d['vah'],
                'D VA%': d['va_pct'],
                # Weekly levels
                'W Zone': w['zone'],
                'W Buy Range (VAL)': w['val'],
                'W Target (POC)': w['poc'],
                'W Resistance (VAH)': w['vah'],
                'W VA%': w['va_pct'],
                # Monthly levels
                'M Zone': m['zone'],
                'M Buy Range (VAL)': m['val'],
                'M Target (POC)': m['poc'],
                'M Resistance (VAH)': m['vah'],
                'M VA%': m['va_pct']
            })
            rank += 1
            
        df_vp = pd.DataFrame(vp_export)
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Scanned", len(df_vp))
        with col2:
            st.metric("Daily Buy Zone", len(df_vp[df_vp['D Zone'] == '✅ Can Buy (Near Support)']) if not df_vp.empty else 0)
        with col3:
            st.metric("Weekly Buy Zone", len(df_vp[df_vp['W Zone'] == '✅ Can Buy (Near Support)']) if not df_vp.empty else 0)
        with col4:
            st.metric("Monthly Buy Zone", len(df_vp[df_vp['M Zone'] == '✅ Can Buy (Near Support)']) if not df_vp.empty else 0)
        
        # Column groups per timeframe
        daily_cols = ['Rank', 'Symbol', 'CMP', 'D Zone', 'D Buy Range (VAL)', 'D Target (POC)', 'D Resistance (VAH)', 'D VA%']
        weekly_cols = ['Rank', 'Symbol', 'CMP', 'W Zone', 'W Buy Range (VAL)', 'W Target (POC)', 'W Resistance (VAH)', 'W VA%']
        monthly_cols = ['Rank', 'Symbol', 'CMP', 'M Zone', 'M Buy Range (VAL)', 'M Target (POC)', 'M Resistance (VAH)', 'M VA%']
        
        # Timeframe Tabs
        tab_all, tab_daily, tab_weekly, tab_monthly = st.tabs(["📊 All Stocks", "📅 Daily", "📅 Weekly", "📅 Monthly"])
        
        with tab_all:
            disp_vp = df_vp.copy()
            if 'symbol' in disp_vp.columns:
                disp_vp['symbol'] = disp_vp['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_vp, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_all = df_vp.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download All Stocks (CSV)",
                data=csv_all,
                file_name=f"VP_All_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_all"
            )
        
        with tab_daily:
            df_daily = df_vp[df_vp['D Zone'] != ''][daily_cols].copy()
            df_daily = df_daily.sort_values('D VA%', ascending=True)
            df_daily['Rank'] = range(1, len(df_daily) + 1)
            
            buy_daily = df_daily[df_daily['D Zone'] == '✅ Can Buy (Near Support)']
            st.markdown(f"**{len(buy_daily)}** stocks in Daily Buy Zone | **{len(df_daily)}** total with daily data")
            st.caption("💡 **Buy Range (VAL)** = Support level to buy near | **Target (POC)** = High-volume fair value | **Resistance (VAH)** = Upper boundary")
            disp_daily = df_daily.copy()
            if 'symbol' in disp_daily.columns:
                disp_daily['symbol'] = disp_daily['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_daily, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_daily = df_daily.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Daily Timeframe (CSV)",
                data=csv_daily,
                file_name=f"VP_Daily_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_daily"
            )
        
        with tab_weekly:
            df_weekly = df_vp[df_vp['W Zone'] != ''][weekly_cols].copy()
            df_weekly = df_weekly.sort_values('W VA%', ascending=True)
            df_weekly['Rank'] = range(1, len(df_weekly) + 1)
            
            buy_weekly = df_weekly[df_weekly['W Zone'] == '✅ Can Buy (Near Support)']
            st.markdown(f"**{len(buy_weekly)}** stocks in Weekly Buy Zone | **{len(df_weekly)}** total with weekly data")
            st.caption("💡 **Buy Range (VAL)** = Support level to buy near | **Target (POC)** = High-volume fair value | **Resistance (VAH)** = Upper boundary")
            disp_weekly = df_weekly.copy()
            if 'symbol' in disp_weekly.columns:
                disp_weekly['symbol'] = disp_weekly['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_weekly, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_weekly = df_weekly.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Weekly Timeframe (CSV)",
                data=csv_weekly,
                file_name=f"VP_Weekly_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_weekly"
            )
        
        with tab_monthly:
            df_monthly = df_vp[df_vp['M Zone'] != ''][monthly_cols].copy()
            df_monthly = df_monthly.sort_values('M VA%', ascending=True)
            df_monthly['Rank'] = range(1, len(df_monthly) + 1)
            
            buy_monthly = df_monthly[df_monthly['M Zone'] == '✅ Can Buy (Near Support)']
            st.markdown(f"**{len(buy_monthly)}** stocks in Monthly Buy Zone | **{len(df_monthly)}** total with monthly data")
            st.caption("💡 **Buy Range (VAL)** = Support level to buy near | **Target (POC)** = High-volume fair value | **Resistance (VAH)** = Upper boundary")
            disp_monthly = df_monthly.copy()
            if 'symbol' in disp_monthly.columns:
                disp_monthly['symbol'] = disp_monthly['symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
            st.dataframe(disp_monthly, width="stretch", hide_index=True, column_config={"symbol": st.column_config.LinkColumn("Symbol", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)")})
            csv_monthly = df_monthly.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Monthly Timeframe (CSV)",
                data=csv_monthly,
                file_name=f"VP_Monthly_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_vp_monthly"
            )
        
        # Combined Excel download with all sheets
        try:
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df_vp.to_excel(writer, sheet_name='All Stocks', index=False)
                if not df_vp.empty:
                    df_d_buy = df_vp[df_vp['D Zone'] == '✅ Can Buy (Near Support)']
                    df_w_buy = df_vp[df_vp['W Zone'] == '✅ Can Buy (Near Support)']
                    df_m_buy = df_vp[df_vp['M Zone'] == '✅ Can Buy (Near Support)']
                    
                    if not df_d_buy.empty:
                        df_d_buy[daily_cols].to_excel(writer, sheet_name='Daily Buy Zone', index=False)
                    if not df_w_buy.empty:
                        df_w_buy[weekly_cols].to_excel(writer, sheet_name='Weekly Buy Zone', index=False)
                    if not df_m_buy.empty:
                        df_m_buy[monthly_cols].to_excel(writer, sheet_name='Monthly Buy Zone', index=False)
            
            st.download_button(
                label="📥 Download Complete Report (Excel - All Sheets)",
                data=excel_buffer.getvalue(),
                file_name=f"Volume_Profile_Scan_{datetime.now(IST_TIMEZONE).strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_vp_excel"
            )
        except ImportError:
            st.caption("ℹ️ Excel export unavailable — use CSV downloads above instead.")

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

# ==============================================================================
# TAB 22: RSI OVERSOLD SCANNER
# ==============================================================================
with tab_rsi_wt:
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



# ==============================================================================
# TAB: BB SQUEEZE
# ==============================================================================
with tab_ema_support:
    st.markdown("### 📈 9/21 EMA Support")
    st.markdown("Stocks taking support at their 9 or 21 EMA with tight proximity, plus crossover signals.")
    
    col_btn, col_note = st.columns([1, 2])
    run_bb_btn = col_btn.button("🔍 Run EMA Support Scan", type="primary", use_container_width=True)
    
    if run_bb_btn:
        st.session_state.ema_support_results = None
        ALL_TAB_SCAN_STATUS["ema_support_results"] = None
        run_background_ema_support_scan(force=True)
        st.rerun()
        
    if st.session_state.get('ema_support_results') is None and ALL_TAB_SCAN_STATUS.get("ema_support_results") is not None:
        st.session_state.ema_support_results = ALL_TAB_SCAN_STATUS["ema_support_results"]
        
    if st.session_state.get('ema_support_results') is not None:
        ema_list = st.session_state.ema_support_results
        
        # Apply Universe Filter
        if "ALL NSE" not in universe_selection.upper() and len(ema_list) > 0:
            from data_fetcher import get_index_stocks
            resolved_univ = "ALL NSE"
            if "NIFTY 500" in universe_selection: resolved_univ = "NIFTY 500"
            elif "NIFTY 100" in universe_selection: resolved_univ = "NIFTY 100"
            elif "NIFTY 50" in universe_selection: resolved_univ = "NIFTY 50"
            elif "WATCHLIST" in universe_selection.upper(): resolved_univ = "WATCHLIST"
            if resolved_univ != "ALL NSE":
                raw_symbols = get_index_stocks(resolved_univ)
                valid_set = set([str(s).replace('.NS', '').strip().upper() for s in raw_symbols if str(s).strip()])
                ema_list = [r for r in ema_list if r['symbol'] in valid_set]
                
        if len(ema_list) > 0:
            st.markdown("### 📊 9/21 EMA Support Setups")
            
            # --- Download Button Logic ---
            import pandas as pd
            from datetime import datetime
            today_str = get_market_date()
            df_ema = pd.DataFrame([{
                'Symbol': r.get('symbol', ''),
                'CMP': r.get('cmp', 0.0),
                'Change %': round(float(r.get('day_change_pct') or 0.0), 2),
                'Dist to 9 EMA %': round(float(r.get('dist_9ema') or 0.0), 2),
                'Dist to 21 EMA %': round(float(r.get('dist_21ema') or 0.0), 2),
                'Crossover': r.get('crossover', False),
                'Setup': r.get('setup', ''),
                'Buy Price': r.get('buy_price', 0.0),
                'Exit Price': r.get('exit_price', 0.0),
                'Target Price': r.get('target_price', 0.0),
                'Confidence': r.get('confidence', ''),
                'Recommendation': r.get('recommendation', '')
            } for r in ema_list])
            
            csv_ema = df_ema.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download EMA Support Results (CSV)",
                data=csv_ema,
                file_name=f"EMA_Support_{today_str}.csv",
                mime="text/csv",
                key="dl_ema_support_csv"
            )
            
            render_unified_strategy_table(ema_list, "ema_support", "ema_support_tab")
        else:
            if st.session_state.get("ema_support_running", False) or ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
                st.info("⏳ Background scanner is analyzing EMA Support... Please wait.")
            else:
                st.info("✅ Scan completed — no EMA Support setups found for the selected universe.")
    else:
        if st.session_state.get("ema_support_running", False) or ALL_TAB_SCAN_STATUS.get("ema_support_running", False):
            st.info("⏳ Background scanner is analyzing EMA Support... Please wait.")
        else:
            st.warning("⚠️ Scan has not been run yet. Click **'Run EMA Support Scan'** above to start, or enable **Auto-Background Scans** in the sidebar.")


# ==============================================================================
# TAB: STAGE ANALYSIS (MINERVINI TREND TEMPLATE)
# ==============================================================================
with tab_stage_analysis:
    st.markdown("### 🏆 Minervini Trend Template — Stage Analyzer")
    st.markdown("Analyzes Minervini's 8 Trend Template criteria to classify stocks into Stage 1, Stage 2 (Uptrend), Stage 3 (Topping), or Stage 4 (Decline).")
    
    col_sa1, col_sa2 = st.columns([1, 2])
    run_sa_btn = col_sa1.button("🔍 Run Stage Analysis Scan", type="primary", use_container_width=True)
    
    if run_sa_btn:
        st.session_state.stage_analysis_results = None
        
        with st.spinner(f"Running Stage Analysis Scan on {universe_selection}..."):
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
            if "NIFTY 500" in universe_selection: sa_universe = "NIFTY 500"
            elif "NIFTY 100" in universe_selection: sa_universe = "NIFTY 100"
            elif "NIFTY 50" in universe_selection: sa_universe = "NIFTY 50"
            elif "WATCHLIST" in universe_selection.upper(): sa_universe = "WATCHLIST"
            
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
                            if len(df) >= 200:
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
        if "ALL NSE" not in universe_selection.upper() and len(sa_list) > 0:
            from data_fetcher import get_index_stocks
            resolved_univ = "ALL NSE"
            if "NIFTY 500" in universe_selection: resolved_univ = "NIFTY 500"
            elif "NIFTY 100" in universe_selection: resolved_univ = "NIFTY 100"
            elif "NIFTY 50" in universe_selection: resolved_univ = "NIFTY 50"
            elif "WATCHLIST" in universe_selection.upper(): resolved_univ = "WATCHLIST"
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


