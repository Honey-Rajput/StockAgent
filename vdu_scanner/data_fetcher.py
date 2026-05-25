# data_fetcher.py
import pandas as pd
import yfinance as yf
import streamlit as st
import requests
import io
from config import NIFTY50_SYMBOLS, NIFTY100_SYMBOLS, LOOKBACK_DAYS

@st.cache_data(ttl=900)
def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """
    Fetches the last 120 calendar days of daily OHLCV data for a given NSE symbol.
    Appends '.NS' suffix if not present, drops rows with zero volume,
    and returns a clean, structured pandas DataFrame.
    """
    # Clean symbol formatting
    formatted_symbol = symbol.strip().upper()
    if not formatted_symbol.endswith(".NS"):
        formatted_symbol = f"{formatted_symbol}.NS"
        
    import time
    retries = 0
    max_retries = 2
    backoff = 1.5
    
    while retries <= max_retries:
        try:
            # Download OHLCV data with auto_adjust=False to fetch raw unadjusted NSE prices
            df = yf.download(
                tickers=formatted_symbol, 
                period=f"{LOOKBACK_DAYS}d", 
                interval="1d", 
                progress=False,
                auto_adjust=False
            )
            
            if df is not None and not df.empty:
                # Clean multi-index headers if present in yfinance response
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                    
                # Standardize columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    return None
                    
                # Extract and clean
                df_clean = df[required_cols].copy()
                df_clean = df_clean.dropna(subset=['Close'])
                df_clean = df_clean[df_clean['Volume'] > 0]
                
                if df_clean.empty:
                    return None
                    
                # Reset index to make Date a column
                df_clean = df_clean.reset_index()
                # Rename date column if yfinance returned index with another name
                df_clean.rename(columns={df_clean.columns[0]: 'Date'}, inplace=True)
                
                # Ensure Date column is datetime
                df_clean['Date'] = pd.to_datetime(df_clean['Date'])
                
                return df_clean
            else:
                raise ValueError("Empty DataFrame from yfinance")
                
        except Exception as e:
            retries += 1
            if retries > max_retries:
                # Silently log the warning to standard output/logs and skip
                print(f"Error fetching OHLCV for symbol {symbol} after {max_retries} retries: {e}")
                return None
            time.sleep(backoff)
            backoff *= 2.0
            
    return None

@st.cache_data(ttl=3600)
def get_index_stocks(index_name: str) -> list[str]:
    """
    Resolves the stock list for the selected index name.
    Supports NIFTY 50, NIFTY 100, NIFTY 500, ALL NSE, and falls back gracefully.
    """
    index_name = index_name.strip().upper()
    
    if index_name == "NIFTY 50":
        return NIFTY50_SYMBOLS
    elif index_name == "NIFTY 100":
        return NIFTY100_SYMBOLS
    elif index_name == "NIFTY 500":
        return fetch_nifty500_constituent_symbols()
    elif index_name == "ALL NSE":
        return get_all_nse_symbols()
    
    return NIFTY50_SYMBOLS

def get_all_nse_symbols() -> list[str]:
    """
    Downloads the official constituent list of all listed equity shares on the 
    National Stock Exchange of India (NSE) directly from the nsearchives.
    Filters to keep mainboard active equities (SERIES == 'EQ').
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    # Try the newer, more stable nsearchives URL first
    urls = [
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L_CO_ME.csv"
    ]
    
    for url in urls:
        try:
            print(f"Fetching all NSE listed equities from: {url}")
            res = requests.get(url, headers=headers, timeout=12)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                
                # Normalize column headers by stripping whitespace and capitalizing
                df.columns = df.columns.str.strip().str.upper()
                
                if 'SYMBOL' in df.columns and 'SERIES' in df.columns:
                    # Filter to only keep active mainboard EQ series
                    eq_df = df[df['SERIES'].astype(str).str.strip().str.upper() == 'EQ']
                    symbols = eq_df['SYMBOL'].dropna().astype(str).tolist()
                    clean_symbols = [s.strip() for s in symbols if s.strip() and s.strip() != 'SYMBOL']
                    if len(clean_symbols) > 100:
                        print(f"Successfully loaded {len(clean_symbols)} active NSE symbols from {url}.")
                        return clean_symbols
        except Exception as e:
            print(f"Failed to fetch from {url}: {e}")
        
    # Graceful fallback to NIFTY 500 constituents if all downloads fail
    print("WARNING: Falling back to NIFTY 500 constituent list.")
    return fetch_nifty500_constituent_symbols()

def fetch_nifty500_constituent_symbols() -> list[str]:
    """
    Tries to retrieve the NIFTY 500 stock index list via nsepython.
    Gracefully falls back to downloading the official constituent CSV directly 
    from the NSE website archives.
    """
    # Step A: Try nsepython
    try:
        from nsepython import nse_get_index_list, get_indices_stocks
        # Let's call get_indices_stocks which returns the stocks for a index
        stocks_df = get_indices_stocks("NIFTY500")
        if isinstance(stocks_df, pd.DataFrame) and not stocks_df.empty:
            for col in stocks_df.columns:
                if col.lower() in ['symbol', 'ticker', 'code']:
                    symbols = stocks_df[col].dropna().astype(str).tolist()
                    clean_symbols = [s.strip() for s in symbols if s.strip() and s.strip() != 'Symbol']
                    if len(clean_symbols) > 50:
                        return clean_symbols
    except Exception as e:
        print(f"nsepython failed to fetch NIFTY 500 list: {e}. Attempting direct download from NSE archives...")
        
    # Step B: Direct official NSE URL download
    try:
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text))
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].dropna().astype(str).tolist()
                clean_symbols = [s.strip() for s in symbols if s.strip() and s.strip() != 'Symbol']
                if len(clean_symbols) > 50:
                    return clean_symbols
    except Exception as ex:
        print(f"NSE direct constituent list download failed: {ex}")
        
    # Step C: Fallback to static NIFTY 100 lists if everything fails
    print("WARNING: Returning static NIFTY 100 constituent list as backup.")
    return NIFTY100_SYMBOLS
