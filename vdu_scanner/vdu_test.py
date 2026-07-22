import pandas as pd
import numpy as np
import sys

# Generate dummy data
dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
df = pd.DataFrame({
    'Date': dates,
    'Open': np.linspace(100, 110, 100),
    'High': np.linspace(105, 115, 100),
    'Low': np.linspace(95, 105, 100),
    'Close': np.linspace(102, 112, 100),
    'Volume': [1000] * 99 + [3000] # Breakout today
})

# Make today a 8% gap up
df.loc[99, 'Open'] = 110
df.loc[99, 'Close'] = 110 * 1.08 

for i in range(50, 57):
    df.loc[i, 'Volume'] = 1500 # higher than threshold

sys.path.append('.')
from scanner import DRY_VOLUME_THRESHOLD
print("Threshold:", DRY_VOLUME_THRESHOLD)

# Let's manually trace STEP 2
baseline_subset = df.iloc[-90:] if len(df) >= 90 else df
baseline_avg_vol = baseline_subset['Volume'].mean()
print("Baseline vol:", baseline_avg_vol)

history_df = df.iloc[:-1].reset_index(drop=True)
is_not_dry_mask = history_df['Volume'] > (DRY_VOLUME_THRESHOLD * baseline_avg_vol)
print("Not dry count total:", is_not_dry_mask.sum())

min_found_avg_vol = float('inf')
best_window = None
search_start_idx = len(history_df) - 1
search_end_idx = max(0, len(history_df) - 10)

for idx in range(search_start_idx, search_end_idx - 1, -1):
    for L in range(10, 50 + 1):
        start_idx = idx - L + 1
        if start_idx < 0:
            continue
            
        dry_zone_df = history_df.iloc[start_idx : idx + 1]
        dry_avg_vol = dry_zone_df['Volume'].mean()
        
        if dry_avg_vol > 0 and dry_avg_vol <= (0.90 * baseline_avg_vol):
            not_dry_count = is_not_dry_mask.iloc[start_idx : idx + 1].sum()
            if not_dry_count >= 7:
                if dry_avg_vol < min_found_avg_vol:
                    min_found_avg_vol = dry_avg_vol
                    best_window = (start_idx, idx, L, int(not_dry_count), dry_avg_vol)

print("Best window:", best_window)
