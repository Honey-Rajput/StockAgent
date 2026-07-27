import pandas as pd
from data_fetcher import get_top1000_nse_symbols, fetch_ohlcv
from scanner import scan_stock
import time

symbols = get_top1000_nse_symbols()[:100] # Test first 100
matches = []

for sym in symbols:
    df = fetch_ohlcv(sym)
    if df is not None and not df.empty:
        res = scan_stock(sym, df, min_volume_ratio=2.0, min_price_change=2.0)
        if res:
            matches.append(res)
    time.sleep(0.1)

print(f"Total matched: {len(matches)}")
for m in matches:
    print(m['symbol'], m['setup_type'])
