import yfinance as yf
import json
import concurrent.futures
from data_fetcher import get_all_nse_symbols

def get_sector(symbol):
    try:
        yf_sym = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        info = yf.Ticker(yf_sym).info
        sec = info.get('sector') or info.get('industry') or 'Unknown'
        return symbol, sec
    except Exception:
        return symbol, 'Unknown'

if __name__ == "__main__":
    symbols = get_all_nse_symbols()
    print(f"Fetching sectors for {len(symbols)} symbols...")
    sector_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(get_sector, s): s for s in symbols}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            sym, sec = future.result()
            if sec != 'Unknown':
                sector_map[sym] = sec
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{len(symbols)} symbols. Found {len(sector_map)} sectors.")
                
    with open("sector_map.json", "w") as f:
        json.dump(sector_map, f)
    print(f"Saved {len(sector_map)} sectors to sector_map.json")
