import yfinance as yf
import pandas as pd
from datetime import datetime
import joblib
import warnings

warnings.filterwarnings('ignore')

from data_fetcher import get_index_stocks
from scan_orchestrator import process_single_symbol
import database
from local_cache_manager import bulk_get_cached_ohlcv

def run_all_tabs_scan():
    print("Starting full scan of NIFTY 500 for all tabs...")
    symbols = get_index_stocks("NIFTY 500")
    symbols_ns = [f"{s.strip().upper()}.NS" for s in symbols]
    
    print("Downloading benchmark...")
    nifty_benchmark_df = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    
    print("Fetching local cache for stocks...")
    bulk_data = bulk_get_cached_ohlcv(symbols, "1d")
    
    missing = []
    for sym in symbols:
        if sym not in bulk_data or bulk_data[sym].empty:
            missing.append(sym)
            
    if missing:
        print(f"Downloading {len(missing)} missing symbols...")
        df_bulk = yf.download([f"{s}.NS" for s in missing], period="1y", interval="1d", progress=False)
        for sym in missing:
            sym_ns = f"{sym}.NS"
            try:
                if isinstance(df_bulk.columns, pd.MultiIndex):
                    t_df = df_bulk.xs(sym_ns, axis=1, level=1).copy()
                else:
                    t_df = df_bulk.copy()
                t_df = t_df.dropna(subset=["Close"]).reset_index()
                t_df.rename(columns={t_df.columns[0]: "Date"}, inplace=True)
                t_df["Date"] = pd.to_datetime(t_df["Date"], utc=True).dt.tz_localize(None)
                bulk_data[sym] = t_df
            except Exception as e:
                pass

    print("Analyzing stocks...")
    
    # All the lists from app.py
    flagged_list = []
    ema_support_list = []
    gapup_list = []
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
    stage_analysis_list = []
    stage2_list = []
    
    def process(sym):
        try:
            res = process_single_symbol(
                sym, bulk_data.get(sym), nifty_benchmark_df, 
                {}, {}, {}, {}, {}, 
                min_dry=3, max_dry=20, min_vol_ratio=1.5, min_price_chg=2.0, 
                min_dry_spikes=3, min_signal_str=5.0, above_50dma_only=False, 
                above_200dma_only=False, vcp_max_tightness=10.0,
                sma20_lower_bound=-2.0, sma20_upper_bound=2.0,
                sma50_lower_bound=-2.0, sma50_upper_bound=2.0,
                sma20_min_volume=50000, sma_timeframe="Daily", scan_mode="all"
            )
            return res
        except Exception:
            return {"failed": True}

    results = joblib.Parallel(n_jobs=4, backend="threading")(
        joblib.delayed(process)(sym) for sym in symbols
    )
    
    for res in results:
        if res.get("failed"): continue
        
        # Populate lists, injecting missing market_cap_cr
        def get_and_fix(key):
            item = res.get(key)
            if item: item['market_cap_cr'] = 0
            return item
            
        if get_and_fix("flagged"): flagged_list.append(res["flagged"])
        if get_and_fix("ema_support"): ema_support_list.append(res["ema_support"])
        if get_and_fix("gapup"): gapup_list.append(res["gapup"])
        if get_and_fix("above_ma"): above_ma_list.append(res["above_ma"])
        if get_and_fix("support_ma"): support_ma_list.append(res["support_ma"])
        if get_and_fix("crossover_ma"): crossover_ma_list.append(res["crossover_ma"])
        if get_and_fix("minervini"): minervini_list.append(res["minervini"])
        if get_and_fix("wt"): wt_list.append(res["wt"])
        if get_and_fix("vcs"): vcs_list.append(res["vcs"])
        if get_and_fix("vpa"): vpa_list.append(res["vpa"])
        if get_and_fix("zanger"): zanger_list.append(res["zanger"])
        if get_and_fix("volume_profile"): vp_list.append(res["volume_profile"])
        if get_and_fix("support_rsi"): support_rsi_list.append(res["support_rsi"])
        if get_and_fix("stage_analysis"): stage_analysis_list.append(res["stage_analysis"])
        if get_and_fix("stage2"): stage2_list.append(res["stage2"])

    print(f"Found {len(flagged_list)} breakouts, {len(above_ma_list)} above MA, {len(crossover_ma_list)} MA cross...")
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    trend_setups_list = above_ma_list + support_ma_list + crossover_ma_list + minervini_list
    
    database.save_scan_results(
        date_str=date_str,
        breakouts=flagged_list,
        squeezes=[], # legacy
        gapups=gapup_list,
        trend_setups=trend_setups_list,
        wt_cross=wt_list,
        total_scanned=len(symbols),
        vcs_results=vcs_list,
        vpa_results=vpa_list
    )
    try: database.save_zanger_scan(date_str, "Daily", zanger_list)
    except Exception: pass
    try: database.save_volume_profile_only(date_str, vp_list)
    except Exception: pass
    try: database.save_support_rsi_only(date_str, support_rsi_list)
    except Exception: pass
    try: database.save_ema_support_only(date_str, ema_support_list)
    except Exception: pass
    try: database.save_stage_analysis_only(date_str, stage_analysis_list)
    except Exception: pass
    try: database.save_stage2_only(date_str, stage2_list)
    except Exception: pass
    
    print("Saved all tabs successfully!")

if __name__ == "__main__":
    run_all_tabs_scan()
