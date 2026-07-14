import json
import yfinance as yf
import pandas as pd

syms = json.load(open('nse_all_symbols.json'))[:100]
chunk = [s + '.NS' for s in syms]
df = yf.download(chunk, period='1mo', progress=False)

tickers = df.columns.get_level_values(1).unique().tolist()
print(f'Downloaded {len(tickers)} tickers out of 100')

valid = 0
for t in tickers:
    t_df = df.xs(t, axis=1, level=1).dropna(subset=['Close'])
    if not t_df.empty:
        valid += 1

print(f'Valid DataFrames: {valid}')
