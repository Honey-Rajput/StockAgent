# data_fetcher.py
import pandas as pd
import yfinance as yf
import streamlit as st
import requests
import io
from config import NIFTY50_SYMBOLS, NIFTY100_SYMBOLS, LOOKBACK_DAYS


def _flatten_yf_dataframe(df: pd.DataFrame, symbol_ns: str = None) -> pd.DataFrame | None:
    """
    Normalizes a yfinance DataFrame to have flat column names (Open, High, Low, Close, Volume).
    Handles both single-ticker flat DataFrames and multi-ticker MultiIndex DataFrames.
    Compatible with yfinance 1.x (auto_adjust=True by default).
    """
    if df is None or df.empty:
        return None

    # Case 1: MultiIndex columns (multi-ticker download) — extract the specific ticker slice
    if isinstance(df.columns, pd.MultiIndex):
        if symbol_ns is not None:
            # Try exact match first, then case-insensitive
            all_tickers = df.columns.get_level_values(1).unique().tolist()
            matched_ticker = None
            for t in all_tickers:
                if t.upper() == symbol_ns.upper():
                    matched_ticker = t
                    break
            if matched_ticker is None:
                return None
            ticker_df = df.xs(matched_ticker, axis=1, level=1).copy()
        else:
            # Single ticker wrapped in MultiIndex — drop the ticker level
            ticker_df = df.copy()
            ticker_df.columns = ticker_df.columns.get_level_values(0)
    else:
        # Case 2: Already flat columns (single ticker download)
        ticker_df = df.copy()

    # In yfinance 1.x with auto_adjust=True, 'Adj Close' is gone; 'Close' is adjusted.
    # Rename 'Price' -> 'Close' if needed (some versions emit 'Price')
    if 'Price' in ticker_df.columns and 'Close' not in ticker_df.columns:
        ticker_df.rename(columns={'Price': 'Close'}, inplace=True)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in ticker_df.columns for col in required_cols):
        return None

    df_clean = ticker_df[required_cols].copy()
    df_clean = df_clean.dropna(subset=['Close'])
    df_clean = df_clean[df_clean['Volume'] > 0]

    if df_clean.empty:
        return None

    # Reset index so Date becomes a column
    df_clean = df_clean.reset_index()
    df_clean.rename(columns={df_clean.columns[0]: 'Date'}, inplace=True)
    df_clean['Date'] = pd.to_datetime(df_clean['Date']).dt.tz_localize(None)

    return df_clean


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    """
    Fetches the last LOOKBACK_DAYS of daily OHLCV data for a given NSE symbol.
    Compatible with yfinance 1.x (auto_adjust=True is the new default).
    """
    formatted_symbol = symbol.strip().upper()
    if not formatted_symbol.endswith(".NS"):
        formatted_symbol = f"{formatted_symbol}.NS"

    import time
    retries = 0
    max_retries = 2
    backoff = 1.5

    while retries <= max_retries:
        try:
            df = yf.download(
                tickers=formatted_symbol,
                period=f"{LOOKBACK_DAYS}d",
                interval="1d",
                progress=False,
                # NOTE: auto_adjust=True is the default in yfinance 1.x
                # threads param was removed in yfinance 1.x
            )

            result = _flatten_yf_dataframe(df)
            if result is not None:
                return result
            raise ValueError("Empty or invalid DataFrame from yfinance")

        except Exception as e:
            retries += 1
            if retries > max_retries:
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


@st.cache_data(ttl=86400)
def fetch_nse_company_names() -> dict:
    """
    Downloads the official constituent list of listed equities on the NSE
    and builds a dictionary mapping symbols to corporate names.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    urls = [
        "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://archives.nseindia.com/content/equities/EQUITY_L_CO_ME.csv"
    ]
    name_map = {}
    for url in urls:
        try:
            print(f"Downloading NSE company names from: {url}")
            res = requests.get(url, headers=headers, timeout=8)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                df.columns = df.columns.str.strip().str.upper()
                if 'SYMBOL' in df.columns and 'NAME OF COMPANY' in df.columns:
                    for _, row in df.iterrows():
                        sym = str(row['SYMBOL']).strip().upper()
                        name = str(row['NAME OF COMPANY']).strip()
                        if sym and name:
                            name_map[sym] = name
                    if len(name_map) > 100:
                        print(f"Successfully mapped {len(name_map)} company names from {url}.")
                        return name_map
        except Exception as e:
            print(f"Failed to fetch company names from {url}: {e}")
    return {}

@st.cache_data(ttl=86400 * 7)
def fetch_sector_map() -> dict:
    """
    Downloads multiple Nifty lists and uses local cache to map symbols to their industry/sector.
    Returns a dictionary of {symbol: industry}.
    """
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv"
    ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    sector_map = {}
    
    # Base from local JSON if generated
    import json
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(base_dir, "sector_map.json")
    if os.path.exists(map_path):
        try:
            with open(map_path, "r") as f:
                sector_map.update(json.load(f))
        except:
            pass

    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.text))
                if 'Symbol' in df.columns and 'Industry' in df.columns:
                    for _, row in df.iterrows():
                        sym = str(row['Symbol']).strip().upper()
                        ind = str(row['Industry']).strip()
                        if sym and ind and ind.lower() != 'nan':
                            sector_map[sym] = ind
        except Exception as ex:
            print(f"Failed to fetch sectors from {url}: {ex}")
            
    return sector_map

@st.cache_data(ttl=86400 * 7)
def get_stock_sector(symbol: str) -> str:
    """Returns the sector for a symbol. Tries static map first, then yfinance."""
    sym = symbol.strip().upper()
    base_sym = sym.replace('.NS', '')
    
    sector_map = fetch_sector_map()
    if base_sym in sector_map:
        return sector_map[base_sym]
    if sym in sector_map:
        return sector_map[sym]
    
    # Fallback to yfinance (can be slow so only fallback)
    try:
        yf_sym = f"{base_sym}.NS"
        info = yf.Ticker(yf_sym).info
        return info.get('sector') or info.get('industry') or 'Unknown'
    except Exception:
        return 'Unknown'



@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv_timeframe(symbol: str, interval: str = "1d", period: str = None) -> pd.DataFrame | None:
    """
    Fetches historical candles for a given NSE symbol, supporting customized intervals and lookback periods.
    Compatible with yfinance 1.x.
    """
    formatted_symbol = symbol.strip().upper()
    if not formatted_symbol.endswith(".NS"):
        formatted_symbol = f"{formatted_symbol}.NS"

    if period is None:
        if interval == "15m":
            period = "60d"
        elif interval in ["60m", "1h"]:
            period = "730d"
            interval = "60m"
        elif interval == "1wk":
            period = "2y"
        elif interval == "1mo":
            period = "5y"
        else:
            period = f"{LOOKBACK_DAYS}d"
            interval = "1d"

    import time
    retries = 0
    max_retries = 2
    backoff = 1.5

    while retries <= max_retries:
        try:
            df = yf.download(
                tickers=formatted_symbol,
                period=period,
                interval=interval,
                progress=False,
            )

            result = _flatten_yf_dataframe(df)
            if result is not None:
                # For intraday intervals, keep zero-volume bars
                if interval in ["15m", "60m"]:
                    # Re-fetch without dropping zero volume rows
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if 'Price' in df.columns and 'Close' not in df.columns:
                        df.rename(columns={'Price': 'Close'}, inplace=True)
                    req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in req_cols):
                        df_clean = df[req_cols].copy().dropna(subset=['Close'])
                        if not df_clean.empty:
                            df_clean = df_clean.reset_index()
                            df_clean.rename(columns={df_clean.columns[0]: 'Date'}, inplace=True)
                            df_clean['Date'] = pd.to_datetime(df_clean['Date']).dt.tz_localize(None)
                            return df_clean
                return result
            raise ValueError("Empty or invalid DataFrame from yfinance")

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Error fetching OHLCV for {symbol} ({interval}, {period}): {e}")
                return None
            time.sleep(backoff)
            backoff *= 2.0

    return None
