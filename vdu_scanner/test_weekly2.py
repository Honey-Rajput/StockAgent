import pandas as pd
import yfinance as yf
from config import NIFTY100_SYMBOLS

sma20_upper_bound = 1.06
sma20_lower_bound = 0.94
sma50_upper_bound = 1.05
sma50_lower_bound = 0.92

passed = []
for sym in NIFTY100_SYMBOLS[:30]:
    try:
        df = yf.download(f"{sym}.NS", period='1100d', progress=False)
        df.columns = [c[0] for c in df.columns]
        df = df.dropna(subset=['Close'])
        df = df.reset_index()
        if 'Date' not in df.columns:
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df.set_index('Date', inplace=True)
        
        d_frame = df.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        w_close = float(d_frame['Close'].iloc[-1])
                                
        sma20_series = d_frame['Close'].rolling(window=20).mean()
        sma50_series = d_frame['Close'].rolling(window=50).mean()
        
        w_sma20 = float(sma20_series.iloc[-1])
        w_sma50 = float(sma50_series.iloc[-1])
        w_sma20_prev = float(sma20_series.iloc[-2]) if len(sma20_series) >= 2 else w_sma20
        w_sma50_prev = float(sma50_series.iloc[-2]) if len(sma50_series) >= 2 else w_sma50
        
        w_sma200 = float(d_frame['Close'].rolling(window=200).mean().iloc[-1]) if len(d_frame) >= 200 else float('nan')
        
        cond_20_gap = abs(w_close - w_sma20) / w_sma20 <= 0.10 if pd.notna(w_sma20) else False
        cond_50_gap = abs(w_close - w_sma50) / w_sma50 <= 0.10 if pd.notna(w_sma50) else False
        cond_200_gap = abs(w_close - w_sma200) / w_sma200 <= 0.10 if pd.notna(w_sma200) else False
        
        high_5 = d_frame['High'].iloc[-5:].max()
        low_5 = d_frame['Low'].iloc[-5:].min()
        tightness_ok = ((high_5 - low_5) / low_5) <= 0.10 if low_5 > 0 else False
        
        slope_ok = (w_sma20 > w_sma20_prev) and (w_sma50 > w_sma50_prev)
        
        condition_weekly = (
            pd.notna(w_sma20) and pd.notna(w_sma50) and pd.notna(w_close) and pd.notna(w_sma200) and
            cond_20_gap and cond_50_gap and cond_200_gap and
            tightness_ok and slope_ok and
            (w_sma20 <= w_sma50 * sma20_upper_bound) and
            (w_sma20 >= w_sma50 * sma20_lower_bound) and
            (w_sma50 <= w_sma200 * sma50_upper_bound) and
            (w_sma50 >= w_sma200 * sma50_lower_bound) and
            (w_close >= w_sma200 * 0.98)
        )
        if condition_weekly:
            passed.append(sym)
        else:
            print(f"{sym} failed. w200: {w_sma200:.2f}, gap200: {cond_200_gap}, gap50: {cond_50_gap}, gap20: {cond_20_gap}, tightness: {tightness_ok}, slope: {slope_ok}")
    except Exception as e:
        pass
print("Passed:", passed)
