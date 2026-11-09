import sys
from pathlib import Path

_SCRATCH_DIR = Path(__file__).resolve().parent
_VDU_DIR = _SCRATCH_DIR.parent
_ROOT_DIR = _VDU_DIR.parent

for _p in [str(_VDU_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import yfinance as yf

try:
    from vdu_scanner.scan_orchestrator import process_single_symbol
    from vdu_scanner.data_fetcher import NIFTY50_SYMBOLS
except ImportError:
    from scan_orchestrator import process_single_symbol  # type: ignore
    from data_fetcher import NIFTY50_SYMBOLS  # type: ignore

# Skip symbols with known corporate action / 404 delistings on Yahoo
BLACKLISTED = {"TATAMOTORS"}
symbols = [s for s in NIFTY50_SYMBOLS[:30] if s not in BLACKLISTED]
tickers = [f'{s}.NS' for s in symbols]
df_bulk = yf.download(tickers=tickers, period='2y', interval='1d', progress=False)

flagged = []
available_tickers = set(df_bulk.columns.get_level_values(1)) if isinstance(df_bulk.columns, pd.MultiIndex) else set()

for sym in symbols:
    t = f'{sym}.NS'
    try:
        if available_tickers and t not in available_tickers:
            print(f"Skipping {sym} (No data returned by Yahoo)")
            continue
        sub = df_bulk.xs(t, axis=1, level=1).copy().dropna(subset=['Close']).reset_index()
        sub.rename(columns={sub.columns[0]: 'Date'}, inplace=True)
        res = process_single_symbol(
            sym, sub, benchmark_df=None,
            min_dry=30, max_dry=60,
            min_vol_ratio=1.8, min_price_chg=2.0,
            min_dry_spikes=2, min_signal_str=25,
            above_50dma_only=False, above_200dma_only=False,
            vcp_max_tightness=7.0,
            sma20_lower_bound=0.98, sma20_upper_bound=1.05,
            sma50_lower_bound=0.95, sma50_upper_bound=1.15,
            sma20_min_volume=100000, sma_timeframe='Daily', scan_mode='full'
        )
        flagged_data = res.get('flagged')
        if isinstance(flagged_data, dict):
            score = flagged_data.get('signal_strength', 0)
            flagged.append((sym, score))
            print(f"FLAGGED: {sym} (Score: {score})")
    except Exception as e:
        print(f"{sym} err: {e}")

print(f"Total flagged out of {len(symbols)}: {len(flagged)}")

