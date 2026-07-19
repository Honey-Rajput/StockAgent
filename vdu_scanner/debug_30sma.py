import pandas as pd
from local_cache_manager import bulk_get_cached_ohlcv
from scanner import scan_near_30sma

symbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "ABBOTINDIA", "MRF"]
bulk_data = bulk_get_cached_ohlcv(symbols, "1d")

matches = []
for sym, df in bulk_data.items():
    if not df.empty:
        res = scan_near_30sma(sym, df, 3.0)
        if res:
            matches.append(res)

print(f"Found {len(matches)} matches in sample.")
for m in matches:
    print(m)
