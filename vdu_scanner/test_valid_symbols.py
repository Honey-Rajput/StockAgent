import json
import yfinance as yf
import pandas as pd

syms = json.load(open('nse_all_symbols.json'))
print(f"Total symbols in JSON: {len(syms)}")
chunk = [s + '.NS' for s in syms]

df = yf.download(chunk, period='1mo', progress=False)

tickers = df.columns.get_level_values(1).unique().tolist()
valid = 0

for t in tickers:
    t_df = df.xs(t, axis=1, level=1).dropna(subset=['Close'])
    t_df = t_df[t_df['Volume'] > 0]
    if len(t_df) >= 5:
        valid += 1

print(f'Valid DataFrames with >=5 rows: {valid} / {len(syms)}')
