import pandas as pd
from data_fetcher import get_top1000_nse_symbols, fetch_ohlcv
from scanner import scan_stock
from indicators import precompute_indicators
import time

symbols = get_top1000_nse_symbols()[:200]
matches = []

for sym in symbols:
    df = fetch_ohlcv(sym)
    if df is not None and not df.empty:
        ind = precompute_indicators(df)
        res = scan_stock(sym, df, min_volume_ratio=2.2, min_price_change=2.0, indicators=ind)
        if res and res.get('signal_strength', 0) >= 30:
            matches.append(res)
    time.sleep(0.05)

print(f'Total matched: {len(matches)}')
for m in matches:
    print(m['symbol'], m['setup_type'], m['signal_strength'])
