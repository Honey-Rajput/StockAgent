import pandas as pd
import yfinance as yf
from scanner import scan_stock
from config import DRY_VOLUME_THRESHOLD

# Fetch a few stocks that moved a lot recently
symbols = ["RVNL.NS", "IREDA.NS", "IRFC.NS", "SUZLON.NS", "ZOMATO.NS", "RELIANCE.NS", "TCS.NS"]
df_bulk = yf.download(tickers=symbols, period="6mo", interval="1d", progress=False)

matched = 0
for sym in symbols:
    if isinstance(df_bulk.columns, pd.MultiIndex):
        df = df_bulk.xs(sym, axis=1, level=1).copy()
    else:
        df = df_bulk.copy()
    
    df = df.dropna(subset=["Close"])
    if df.empty: continue
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    
    # Force a 7% breakout today to see if it passes STEP 2
    df.loc[df.index[-1], 'Close'] = df['Close'].iloc[-2] * 1.075 # 7.5% up
    df.loc[df.index[-1], 'Volume'] = df['Volume'].mean() * 3.0 # huge volume
    
    # Try the exact scan
    res = scan_stock(sym.replace(".NS", ""), df, min_dry_days=0, max_dry_days=50, min_volume_ratio=2.0, min_price_change=7.0, min_dry_spikes=7)
    print(f"{sym}: {res is not None}")
    if res:
        matched += 1

print(f"Matched {matched} out of {len(symbols)}")
