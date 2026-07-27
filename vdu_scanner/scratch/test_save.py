import pandas as pd
from data_fetcher import get_top1000_nse_symbols, fetch_ohlcv
from scanner import scan_stock
from indicators import precompute_indicators
import database

symbols = get_top1000_nse_symbols()[:50]
matches = []

for sym in symbols:
    df = fetch_ohlcv(sym)
    if df is not None and not df.empty:
        ind = precompute_indicators(df)
        res = scan_stock(sym, df, min_volume_ratio=2.2, min_price_change=2.0, indicators=ind)
        if res and res.get('signal_strength', 0) >= 30:
            matches.append(res)

print('Matches:', len(matches))
print(database.save_scan_results('2026-07-27', matches, [], [], [], [], 50, [], [], []))
