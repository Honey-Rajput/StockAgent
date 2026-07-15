import os
import sys
import time
import random
import concurrent.futures
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import joblib

from dotenv import load_dotenv
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, ".env")
load_dotenv(env_path)

# Set up environment variables if not present (handled by GitHub Actions usually, but safe to check)
if not os.environ.get('TURSO_DATABASE_URL') or not os.environ.get('TURSO_AUTH_TOKEN'):
    print("❌ ERROR: TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set as environment variables.")
    sys.exit(1)

# Import local modules
import database
from local_cache_manager import get_cached_ohlcv, save_to_cache, bulk_get_cached_ohlcv
from config import LOOKBACK_DAYS, IST_TIMEZONE
from data_fetcher import get_index_stocks
from scan_orchestrator import process_single_symbol

def get_market_date(for_display=False):
    today = datetime.now(IST_TIMEZONE)
    if today.isoweekday() == 7:
        target_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
    elif today.isoweekday() == 6:
        target_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        target_date = today.strftime('%Y-%m-%d')
    return target_date
import ai_detector

def run_background_ai_scan(symbols_list, date_str, force=False):
    print(f"Starting AI analysis for {len(symbols_list)} breakouts...")
    def scan_and_save(sym):
        try:
            if not force:
                existing = database.get_pattern_by_date(sym, date_str)
                if existing and existing.get('pattern_name') not in ["Error", "Pending"]:
                    return sym, True
            
            # Simple fallback fetch since cron_scanner already downloaded data, but we can hit Yahoo here too
            df_hist = yf.download(f"{sym}.NS", period="2y", progress=False, threads=False, timeout=20)
            if df_hist is not None and not df_hist.empty:
                if isinstance(df_hist.columns, pd.MultiIndex):
                    df_hist = df_hist.xs(df_hist.columns.get_level_values(1)[0], axis=1, level=1)
                
                req = ["Open", "High", "Low", "Close", "Volume"]
                df_hist = df_hist[req].dropna()
                df_hist = df_hist.reset_index().rename(columns={"index": "Date", "Datetime": "Date"})
                
                ans_dict = ai_detector.detect_chart_pattern(sym, df_hist)
                if ans_dict:
                    pattern_name = ans_dict.get("pattern_name", "None")
                    if pattern_name == "Error": pattern_name = "None Detected"
                    
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for i, (sym, success) in enumerate(executor.map(scan_and_save, symbols_list)):
            if success:
                print(f"AI Scanned ({i+1}/{len(symbols_list)}): {sym}")

def run_headless_scan():
    print("=" * 50)
    print("🚀 Starting Automated Headless Volume Surge Scan")
    print("=" * 50)
    
    universe_key = "NIFTY 500"
    scan_timeframe = "Daily (1d)"
    yf_period = f"{LOOKBACK_DAYS}d"
    yf_interval = "1d"
    
    print(f"Universe: {universe_key} | Timeframe: {scan_timeframe}")
    
    raw_symbols = get_index_stocks(universe_key)
    if not raw_symbols:
        print("❌ No symbols found to scan.")
        sys.exit(1)
        
    all_tickers_ns = []
    for s in raw_symbols:
        formatted = s.strip().upper()
        if not formatted.endswith(".NS"):
            formatted = f"{formatted}.NS"
        all_tickers_ns.append(formatted)
        
    today_date_str = datetime.now(IST_TIMEZONE).strftime('%Y-%m-%d')
    today_ist_str = get_market_date()
    
    # --------------------------------------------------------------------------
    # PHASE 1: Real-time Quotes
    # --------------------------------------------------------------------------
    print("\n--- Phase 1/3: Checking / Downloading Real-Time Quotes ---")
    open_price_map = {}
    close_price_map = {}
    high_price_map = {}
    low_price_map = {}
    volume_map = {}
    
    _now_ist = datetime.now(IST_TIMEZONE)
    from datetime import time as _time
    _market_open = (_now_ist.weekday() < 5 and _time(9, 15) <= _now_ist.time() <= _time(15, 30))
    
    _db_quotes = {}
    if not _market_open:
        print("Market is closed — checking Turso DB for today's cached quotes...")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _p1_tex:
                _fut = _p1_tex.submit(database.get_today_quotes, raw_symbols, today_date_str)
                try:
                    _db_quotes = _fut.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    print("Phase 1 DB check timed out after 10s — falling back to Yahoo")
                    _db_quotes = {}
        except Exception as _dq_err:
            print(f"Phase 1 DB check error: {_dq_err}")
            _db_quotes = {}

    _coverage = len(_db_quotes) / max(len(raw_symbols), 1)

    if _coverage >= 0.90:
        for _sym, _q in _db_quotes.items():
            if _q["close"] > 0:
                close_price_map[_sym] = _q["close"]
                open_price_map[_sym] = _q["open"]
                high_price_map[_sym] = _q["high"]
                low_price_map[_sym] = _q["low"]
                volume_map[_sym] = _q["volume"]
        print(f"✅ Loaded {len(close_price_map)} quotes from Turso DB (skipped Yahoo Finance!)")
    else:
        print("Downloading real-time quotes from Yahoo Finance...")
        chunk_size = 35
        ticker_chunks = [all_tickers_ns[i:i + chunk_size] for i in range(0, len(all_tickers_ns), chunk_size)]
        
        def _download_quote_chunk(idx_chunk_pair):
            idx, chunk = idx_chunk_pair
            _open = {}; _close = {}; _vol = {}; _high = {}; _low = {}
            retries = 0
            max_retries = 2
            backoff = 2.0
            while retries <= max_retries:
                try:
                    quotes_df = yf.download(tickers=chunk, period="1d", progress=False, threads=False, timeout=20)
                    if not quotes_df.empty:
                        if isinstance(quotes_df.columns, pd.MultiIndex):
                            price_types = quotes_df.columns.get_level_values(0).unique().tolist()
                            def _get_field_series(field):
                                if field in price_types:
                                    return quotes_df[field].iloc[-1]
                                return pd.Series(dtype=float)
                            close_series = _get_field_series('Close')
                            open_series = _get_field_series('Open')
                            volume_series = _get_field_series('Volume')
                            high_series = _get_field_series('High')
                            low_series = _get_field_series('Low')
                        else:
                            ticker_key = chunk[0]
                            close_series = pd.Series({ticker_key: quotes_df['Close'].iloc[-1]})
                            open_series = pd.Series({ticker_key: quotes_df['Open'].iloc[-1]}) if 'Open' in quotes_df else close_series
                            volume_series = pd.Series({ticker_key: quotes_df['Volume'].iloc[-1]}) if 'Volume' in quotes_df else pd.Series({ticker_key: 0})
                            high_series = pd.Series({ticker_key: quotes_df['High'].iloc[-1]}) if 'High' in quotes_df else close_series
                            low_series = pd.Series({ticker_key: quotes_df['Low'].iloc[-1]}) if 'Low' in quotes_df else close_series

                        for k, v in close_series.items():
                            clean_k = str(k).replace(".NS", "").upper()
                            if not pd.isna(v) and float(v) > 0:
                                _close[clean_k] = float(v)
                                if k in open_series.index and not pd.isna(open_series[k]): _open[clean_k] = float(open_series[k])
                                if k in volume_series.index and not pd.isna(volume_series[k]): _vol[clean_k] = int(volume_series[k])
                                if k in high_series.index and not pd.isna(high_series[k]): _high[clean_k] = float(high_series[k])
                                if k in low_series.index and not pd.isna(low_series[k]): _low[clean_k] = float(low_series[k])
                        time.sleep(1.5)
                        return (_open, _close, _vol, _high, _low)
                    else:
                        raise ValueError("Empty DataFrame returned")
                except Exception as chunk_ex:
                    retries += 1
                    if retries > max_retries:
                        break
                    time.sleep(backoff)
                    backoff *= 2.0
            return ({}, {}, {}, {}, {})

        p1_workers = 1 if len(ticker_chunks) > 10 else min(3, len(ticker_chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=p1_workers) as p1_executor:
            for i, result in enumerate(p1_executor.map(_download_quote_chunk, list(enumerate(ticker_chunks)))):
                _o, _c, _v, _h, _l = result
                open_price_map.update(_o)
                close_price_map.update(_c)
                volume_map.update(_v)
                high_price_map.update(_h)
                low_price_map.update(_l)
                print(f"Phase 1: Downloaded chunk {i+1}/{len(ticker_chunks)}")
                
                if _c:
                    import threading
                    def _save_chunk_now(cc, co, ch, cl, cv, ds):
                        try:
                            database.save_today_quotes(ds, cc, co, ch, cl, cv)
                        except Exception as e:
                            print(f"Chunk save error: {e}")
                    threading.Thread(target=_save_chunk_now, args=(dict(_c), dict(_o), dict(_h), dict(_l), dict(_v), today_date_str), daemon=True).start()

    scan_symbols = [s for s in raw_symbols if close_price_map.get(s.strip().upper(), 0.0) > 0.0]
    print(f"Phase 1 complete. Found {len(scan_symbols)} active symbols to scan.")
    
    if not scan_symbols:
        print("❌ No active symbols found (Rate limited?). Exiting.")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # PHASE 2: Historical Data
    # --------------------------------------------------------------------------
    print("\n--- Phase 2/3: Checking / Downloading Historical Data ---")
    bulk_data = {}
    _tf_db_key = "1d"
    
    bulk_cached = bulk_get_cached_ohlcv(scan_symbols, _tf_db_key)
    missing_symbols = []
    
    for sym in scan_symbols:
        clean_sym = sym.strip().upper().replace(".NS", "")
        if clean_sym in bulk_cached and not bulk_cached[clean_sym].empty:
            bulk_data[sym.strip().upper()] = bulk_cached[clean_sym]
        else:
            missing_symbols.append(sym)
            
    if missing_symbols:
        print(f"Downloading historical data for {len(missing_symbols)} missing symbols...")
        chunk_size = 30
        sym_chunks = [missing_symbols[i:i + chunk_size] for i in range(0, len(missing_symbols), chunk_size)]
        
        def download_chunk(chunk_idx, chunk):
            chunk_data = {s.strip().upper(): pd.DataFrame() for s in chunk}
            chunk_ns = [f"{s.strip().upper()}.NS" for s in chunk]
            retries = 0
            max_retries = 2
            backoff = 2.0
            while retries <= max_retries:
                try:
                    df = yf.download(tickers=chunk_ns, period=yf_period, interval=yf_interval, progress=False, threads=False, timeout=20)
                    valid_data = {}
                    if df is not None and not df.empty:
                        for sym in chunk:
                            sym_ns = f"{sym.strip().upper()}.NS"
                            if isinstance(df.columns, pd.MultiIndex):
                                all_t = df.columns.get_level_values(1).unique().tolist()
                                matched = next((t for t in all_t if t.upper() == sym_ns.upper()), None)
                                if matched is None: continue
                                t_df = df.xs(matched, axis=1, level=1).copy()
                            else:
                                if len(chunk_ns) == 1:
                                    t_df = df.copy()
                                else:
                                    continue
                                    
                            req = ["Open", "High", "Low", "Close", "Volume"]
                            if all(c in t_df.columns for c in req):
                                t_df = t_df[req].dropna(subset=["Close"])
                                t_df = t_df[t_df["Volume"] > 0]
                                if not t_df.empty:
                                    t_df = t_df.reset_index()
                                    t_df.rename(columns={t_df.columns[0]: "Date"}, inplace=True)
                                    t_df["Date"] = pd.to_datetime(t_df["Date"]).dt.tz_localize(None)
                                    chunk_data[sym.strip().upper()] = t_df
                                    valid_data[sym.strip().upper()] = t_df
                        time.sleep(1.5)
                        if valid_data:
                            import threading
                            def _upload_chunk(data_dict, tf):
                                for sym_k, df_v in data_dict.items():
                                    try: save_to_cache(sym_k, df_v, tf)
                                    except: pass
                            threading.Thread(target=_upload_chunk, args=(valid_data, _tf_db_key), daemon=True).start()
                        return chunk_data
                    else:
                        raise ValueError("Empty DF")
                except Exception as chunk_ex:
                    retries += 1
                    if retries > max_retries:
                        break
                    time.sleep(backoff)
                    backoff *= 2.0
            return chunk_data

        p2_workers = 1 if len(sym_chunks) > 10 else min(3, len(sym_chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=p2_workers) as executor:
            chunk_args = [(i, chunk) for i, chunk in enumerate(sym_chunks)]
            for i, res_chunk in enumerate(executor.map(lambda args: download_chunk(*args), chunk_args)):
                for sym, df_res in res_chunk.items():
                    if df_res is not None and not df_res.empty:
                        bulk_data[sym] = df_res
                print(f"Phase 2: Processed historical chunk {i+1}/{len(sym_chunks)}")
    
    print(f"Phase 2 complete. Ready to scan {len(bulk_data)} symbols.")
    
    # --------------------------------------------------------------------------
    # PHASE 3: Indicator Scan
    # --------------------------------------------------------------------------
    print("\n--- Phase 3/3: Indicator Scanning ---")
    try:
        nifty_benchmark_df = yf.download("^NSEI", period=yf_period, interval=yf_interval, progress=False, threads=False, timeout=15)
    except Exception as e:
        print(f"Failed to fetch benchmark: {e}")
        nifty_benchmark_df = pd.DataFrame()
        
    def process_and_fetch_if_needed(sym, df, benchmark_df, *args):
        try:
            if df is None or len(df) == 0:
                return {"failed": True, "error": "No historical data available"}
            return process_single_symbol(sym, df, benchmark_df, *args)
        except Exception as e:
            return {"failed": True, "error": str(e)}

    # Scanner Settings (Same as Streamlit Defaults)
    min_dry = 30
    max_dry = 75
    min_vol_ratio = 1.5
    min_price_chg = 1.0
    min_dry_spikes = 2
    min_signal_str = 40
    above_50dma_only = True
    above_200dma_only = True
    vcp_max_tightness = 15.0
    sma20_lower_bound = -10.0
    sma20_upper_bound = 10.0
    sma50_lower_bound = -10.0
    sma50_upper_bound = 15.0
    sma20_min_volume = 15
    sma_timeframe_val = "All (Multi-Timeframe Convergence)"
    scan_mode_flag = "full"

    generator = joblib.Parallel(n_jobs=4, backend="threading", return_as="generator_unordered")(
        joblib.delayed(process_and_fetch_if_needed)(
            sym, bulk_data.get(sym.strip().upper()), nifty_benchmark_df,
            open_price_map, close_price_map, high_price_map, low_price_map, volume_map,
            min_dry, max_dry, min_vol_ratio, min_price_chg, min_dry_spikes, min_signal_str,
            above_50dma_only, above_200dma_only, vcp_max_tightness,
            sma20_lower_bound, sma20_upper_bound, sma50_lower_bound, sma50_upper_bound,
            sma20_min_volume, sma_timeframe_val, scan_mode_flag
        ) for sym in scan_symbols
    )

    failed_count = 0
    gapup_list, above_ma_list, support_ma_list, crossover_ma_list = [], [], [], []
    minervini_list, flagged_list, wt_list, vcs_list, structural_vcp_list, vpa_list = [], [], [], [], [], []
    
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
        except Exception:
            failed_count += 1
            
        if (i + 1) % 50 == 0 or i + 1 == len(scan_symbols):
            print(f"Phase 3: Scanned {i+1}/{len(scan_symbols)}...")

    trend_setups_list = above_ma_list + support_ma_list + crossover_ma_list + minervini_list
    
    print("\n--- Final Results & Saving to DB ---")
    print(f"Total Scanned: {len(scan_symbols)}")
    print(f"Breakouts Found: {len(flagged_list)}")
    print(f"Trend Setups: {len(trend_setups_list)}")
    print(f"Gap-Ups: {len(gapup_list)}")
    print(f"Failed Count: {failed_count}")
    
    try:
        database.save_scan_results(
            date_str=today_ist_str,
            breakouts=flagged_list,
            squeezes=[],
            gapups=gapup_list,
            trend_setups=trend_setups_list,
            wt_cross=wt_list,
            total_scanned=len(scan_symbols),
            vcs_results=vcs_list,
            vpa_results=vpa_list
        )
        print("✅ Scan results successfully cached in Turso PostgreSQL!")
        
        # Trigger background AI scans automatically
        all_flagged_syms = [r['symbol'] for r in flagged_list]
        if len(all_flagged_syms) > 0:
            print(f"Triggering AI Background Scan for {len(all_flagged_syms)} symbols...")
            run_background_ai_scan(all_flagged_syms, today_ist_str)
            
    except Exception as db_err:
        print(f"❌ Failed to cache daily scan results to database: {db_err}")

    print("🎉 Automated Headless Scan Complete!")

if __name__ == "__main__":
    try:
        run_headless_scan()
    except Exception as e:
        import traceback
        print(f"❌ SCAN CATASTROPHICALLY FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
