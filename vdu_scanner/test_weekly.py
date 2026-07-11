import pandas as pd
import yfinance as yf

# Load the bounds as they are defined in the app
sma20_upper_bound = 1.06
sma20_lower_bound = 0.94
sma50_upper_bound = 1.05
sma50_lower_bound = 0.92

df = yf.download('RELIANCE.NS', period='1100d', progress=False)

# Flatten
df = df.dropna(subset=['Close'])
df = df.reset_index()
if 'Date' not in df.columns:
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
df.set_index('Date', inplace=True)

df_weekly = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()

d_frame = df_weekly
print(f"Weekly length: {len(d_frame)}")
w_close = float(d_frame['Close'].iloc[-1])
                        
sma20_series = d_frame['Close'].rolling(window=20).mean()
sma50_series = d_frame['Close'].rolling(window=50).mean()

w_sma20 = float(sma20_series.iloc[-1])
w_sma50 = float(sma50_series.iloc[-1])
w_sma20_prev = float(sma20_series.iloc[-2]) if len(sma20_series) >= 2 else w_sma20
w_sma50_prev = float(sma50_series.iloc[-2]) if len(sma50_series) >= 2 else w_sma50

w_sma200 = float(d_frame['Close'].rolling(window=200).mean().iloc[-1]) if len(d_frame) >= 200 else float('nan')

print(f"w_sma200: {w_sma200}")

cond_20_gap = abs(w_close - w_sma20) / w_sma20 <= 0.10 if pd.notna(w_sma20) else False
cond_50_gap = abs(w_close - w_sma50) / w_sma50 <= 0.10 if pd.notna(w_sma50) else False
cond_200_gap = abs(w_close - w_sma200) / w_sma200 <= 0.10 if pd.notna(w_sma200) else False

high_5 = d_frame['High'].iloc[-5:].max()
low_5 = d_frame['Low'].iloc[-5:].min()
tightness_ok = ((high_5 - low_5) / low_5) <= 0.10 if low_5 > 0 else False

slope_ok = (w_sma20 > w_sma20_prev) and (w_sma50 > w_sma50_prev)

print(f"cond_20_gap: {cond_20_gap}")
print(f"cond_50_gap: {cond_50_gap}")
print(f"cond_200_gap: {cond_200_gap}")
print(f"tightness_ok: {tightness_ok}")
print(f"slope_ok: {slope_ok}")
print(f"sma20/sma50 bounds: {w_sma20 <= w_sma50 * sma20_upper_bound and w_sma20 >= w_sma50 * sma20_lower_bound}")
print(f"sma50/sma200 bounds: {w_sma50 <= w_sma200 * sma50_upper_bound and w_sma50 >= w_sma200 * sma50_lower_bound}")
