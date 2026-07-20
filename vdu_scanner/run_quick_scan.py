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

def run_quick_nifty50_scan():
    print("Starting quick scan of NIFTY 50...")
    symbols = get_index_stocks("NIFTY 50")
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
    breakout_list = []
    squeeze_list = []
    
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
        if res.get("flagged"):
            res["flagged"]["market_cap_cr"] = 0
            breakout_list.append(res["flagged"])
        if res.get("ema_support"):
            res["ema_support"]["market_cap_cr"] = 0
            squeeze_list.append(res["ema_support"])
            
    trend_setups_list = []
    wt_list = []
    vpa_list = []
    vcs_list = []
    near_30sma_list = []
    gapup_list = []
    
    for res in results:
        if res.get("above_ma"): trend_setups_list.append(res["above_ma"])
        if res.get("support_ma"): trend_setups_list.append(res["support_ma"])
        if res.get("crossover_ma"): trend_setups_list.append(res["crossover_ma"])
        if res.get("minervini"): trend_setups_list.append(res["minervini"])
        if res.get("zanger"): trend_setups_list.append(res["zanger"])
        if res.get("wt_cross"): wt_list.append(res["wt_cross"])
        if res.get("vpa"): vpa_list.append(res["vpa"])
        if res.get("vcs"): vcs_list.append(res["vcs"])
        if res.get("near_30sma"): near_30sma_list.append(res["near_30sma"])
        if res.get("gapup"): gapup_list.append(res["gapup"])
        
    print(f"Found {len(breakout_list)} breakouts and {len(squeeze_list)} squeezes.")
    print(f"Found {len(trend_setups_list)} trend setups and {len(near_30sma_list)} near 30sma.")
    
    from cron_scanner import get_market_date
    date_str = get_market_date()
    database.save_scan_results(
        date_str=date_str,
        breakouts=breakout_list,
        squeezes=squeeze_list,
        gapups=gapup_list,
        trend_setups=trend_setups_list,
        wt_cross=wt_list,
        total_scanned=len(symbols),
        vcs_results=vcs_list,
        vpa_results=vpa_list
    )
    database.save_near_30sma_only(date_str, near_30sma_list)
    print("Saved successfully!")

if __name__ == "__main__":
    run_quick_nifty50_scan()
