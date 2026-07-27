"""
Full end-to-end VDU scanner test:
1. Fetches fresh data for first 200 symbols (triggers incremental yfinance updates for missing ones)
2. Runs scan_stock on each with the updated data
3. Shows results per date for today
"""
from data_fetcher import fetch_ohlcv, get_top1000_nse_symbols
from scanner import scan_stock
from indicators import precompute_indicators
import datetime

symbols = get_top1000_nse_symbols()[:200]
today = datetime.date.today()
print(f'Today: {today}')

matches_today = []
matches_other = []
stale_count = 0
no_data_count = 0
no_match_count = 0

for sym in symbols:
    df = fetch_ohlcv(sym)
    if df is None or df.empty:
        no_data_count += 1
        continue
    
    last_date = df['Date'].iloc[-1]
    if hasattr(last_date, 'date'):
        last_date = last_date.date()
    
    ind = precompute_indicators(df)
    res = scan_stock(sym, df, min_volume_ratio=2.2, min_price_change=2.0, indicators=ind)
    
    if res is None:
        if last_date < today:
            stale_count += 1
        else:
            no_match_count += 1
    else:
        if last_date >= today:
            matches_today.append(res)
        else:
            matches_other.append(res)

print(f'\nSummary (of {len(symbols)} symbols):')
print(f'  Matches for TODAY ({today}): {len(matches_today)}')
print(f'  Matches for STALE date: {len(matches_other)}')
print(f'  No match (fresh data): {no_match_count}')
print(f'  No match (stale data): {stale_count}')
print(f'  No data at all: {no_data_count}')

print('\nToday matches:')
for m in matches_today:
    print(f'  {m["symbol"]} {m["setup_type"]} score={m["signal_strength"]} vol={m["volume_ratio"]}x')

print('\nStale matches (not from today):')
for m in matches_other[:10]:
    print(f'  {m["symbol"]} {m["setup_type"]} score={m["signal_strength"]}')
