from data_fetcher import fetch_ohlcv, get_top1000_nse_symbols
from scanner import scan_stock
from indicators import precompute_indicators
import datetime

# Test a few specific stocks
test_symbols = ['TCS', 'RELIANCE', 'INFY', 'HDFCBANK', 'IRFC', 'AARON', 'ACE', 'ASAHIINDIA']
today = datetime.date.today()
print('Today:', today)

for sym in test_symbols:
    df = fetch_ohlcv(sym)
    if df is not None and not df.empty:
        last_date = str(df['Date'].iloc[-1]) if 'Date' in df.columns else str(df.index[-1])
        last_close = df['Close'].iloc[-1]
        last_vol = df['Volume'].iloc[-1]
        print(f'{sym}: Last date={last_date}, Close={last_close:.2f}, Vol={last_vol:,.0f}, rows={len(df)}')
        
        ind = precompute_indicators(df)
        res = scan_stock(sym, df, min_volume_ratio=2.2, min_price_change=2.0, indicators=ind)
        if res:
            print(f'  => {res["setup_type"]} score={res["signal_strength"]} vol_ratio={res["volume_ratio"]}')
        else:
            print(f'  => No match')
    else:
        print(f'{sym}: No data')
