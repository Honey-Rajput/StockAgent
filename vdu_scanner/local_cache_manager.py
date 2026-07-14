import os
import time
import pandas as pd
from datetime import datetime, timedelta

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 8 hours TTL. It will download fresh data if the cached file is older than 8 hours.
CACHE_TTL_SECONDS = 8 * 3600

def get_cached_ohlcv(symbol: str, timeframe: str = "1d") -> pd.DataFrame | None:
    """Returns the cached DataFrame if it exists and is not stale."""
    clean_sym = symbol.strip().upper().replace(".NS", "")
    file_path = os.path.join(CACHE_DIR, f"{clean_sym}_{timeframe}.parquet")
    
    if os.path.exists(file_path):
        mtime = os.path.getmtime(file_path)
        if time.time() - mtime < CACHE_TTL_SECONDS:
            try:
                df = pd.read_parquet(file_path)
                return df
            except Exception as e:
                print(f"Error reading cache for {symbol}: {e}")
                pass
    return None

def save_to_cache(symbol: str, df: pd.DataFrame, timeframe: str = "1d"):
    """Saves the DataFrame to the local Parquet cache."""
    if df is None or df.empty:
        return
        
    clean_sym = symbol.strip().upper().replace(".NS", "")
    file_path = os.path.join(CACHE_DIR, f"{clean_sym}_{timeframe}.parquet")
    
    try:
        df.to_parquet(file_path, index=False)
    except Exception as e:
        print(f"Error saving cache for {symbol}: {e}")
