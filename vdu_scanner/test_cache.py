from local_cache_manager import get_cached_ohlcv, save_to_cache
import yfinance as yf
import pandas as pd
import time

sym = 'TCS.NS'
timeframe = '1d'

# 1. Fetch from yfinance
print(f"Fetching {sym} from yfinance...")
start = time.time()
df = yf.download(sym, period='1mo', interval=timeframe, progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df = df.reset_index()
print(f"Fetch time: {time.time() - start:.2f}s")

# 2. Save to cache
print(f"Saving to cache...")
save_to_cache(sym, df, timeframe)

# 3. Read from cache
print(f"Reading from cache...")
start = time.time()
cached_df = get_cached_ohlcv(sym, timeframe)
print(f"Read time: {time.time() - start:.4f}s")
print(f"Cached rows: {len(cached_df)}")
