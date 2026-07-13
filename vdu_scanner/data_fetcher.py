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


import threading
import time

class ThreadSafeTTLCache:
    def __init__(self, ttl_seconds, max_size):
        self.cache = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())

_ohlcv_cache = ThreadSafeTTLCache(ttl_seconds=900, max_size=150)

def fetch_ohlcv(symbol: str) -> pd.DataFrame | None:
    """
    Fetches the last LOOKBACK_DAYS of daily OHLCV data for a given NSE symbol.
    Compatible with yfinance 1.x (auto_adjust=True is the new default).
    """
    formatted_symbol = symbol.strip().upper()
    if not formatted_symbol.endswith(".NS"):
        formatted_symbol = f"{formatted_symbol}.NS"
        
    cached_df = _ohlcv_cache.get(formatted_symbol)
    if cached_df is not None:
        return cached_df.copy()

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
                _ohlcv_cache.set(formatted_symbol, result)
                return result.copy()
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


MAX_NSE_SYMBOLS = 1800  # Limit to top 1800 NSE stocks by prominence to reduce scan time

def get_all_nse_symbols() -> list[str]:
    """
    Downloads the official constituent list of all listed equity shares on the
    National Stock Exchange of India (NSE) directly from the nsearchives.
    Filters to keep mainboard active equities (SERIES == 'EQ').
    Returns at most MAX_NSE_SYMBOLS (1800) stocks for faster scanning.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Load from the local JSON file to completely bypass NSE blocking on Streamlit Cloud
    import json
    import os
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'nse_all_symbols.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                symbols = json.load(f)
            if symbols and len(symbols) > 100:
                symbols = symbols[:MAX_NSE_SYMBOLS]
                print(f"Successfully loaded {len(symbols)} active NSE symbols from local JSON cache (capped at {MAX_NSE_SYMBOLS}).")
                return symbols
    except Exception as e:
        print(f"Failed to load local JSON cache: {e}")

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
                        clean_symbols = clean_symbols[:MAX_NSE_SYMBOLS]
                        print(f"Successfully loaded {len(clean_symbols)} active NSE symbols from {url} (capped at {MAX_NSE_SYMBOLS}).")
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
        except Exception as e:
            print(f"Warning: Failed to load sector_map.json: {e}")

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


_ohlcv_timeframe_cache = ThreadSafeTTLCache(ttl_seconds=900, max_size=150)


def fetch_ohlcv_timeframe(symbol: str, interval: str = "1d", period: str = None) -> pd.DataFrame | None:
    """
    Fetches historical candles for a given NSE symbol, supporting customized intervals and lookback periods.
    Compatible with yfinance 1.x.
    """
    formatted_symbol = symbol.strip().upper()
    if not formatted_symbol.endswith(".NS"):
        formatted_symbol = f"{formatted_symbol}.NS"
        
    cache_key = f"{formatted_symbol}_{interval}_{period}"
    cached_df = _ohlcv_timeframe_cache.get(cache_key)
    if cached_df is not None:
        return cached_df.copy()

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
                            _ohlcv_timeframe_cache.set(cache_key, df_clean)
                            return df_clean.copy()
                _ohlcv_timeframe_cache.set(cache_key, result)
                return result.copy()
            raise ValueError("Empty or invalid DataFrame from yfinance")

        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Error fetching OHLCV for {symbol} ({interval}, {period}): {e}")
                return None
            time.sleep(backoff)
            backoff *= 2.0

    return None


# =============================================================================
# MARKET CONDITION & RELATIVE STRENGTH HELPERS
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_market_condition() -> dict:
    """
    Returns Nifty 50 market health based on price vs 50-DMA and 200-DMA.
    - 'Bullish'  : CMP > SMA50 > SMA200  (strong uptrend — best time to buy breakouts)
    - 'Caution'  : CMP > SMA200 but below SMA50  (above long-term trend but weak near-term)
    - 'Bearish'  : CMP < SMA200  (downtrend — avoid new buys, reduce exposure)

    Cached for 15 minutes to avoid redundant API calls on each Streamlit re-render.
    """
    try:
        df = yf.download("^NSEI", period="250d", interval="1d", progress=False)
        if df is None or df.empty:
            return {'status': 'Unknown', 'emoji': '⚪', 'cmp': 0.0, 'sma50': 0.0, 'sma200': 0.0, 'change_pct': 0.0}

        # Flatten MultiIndex if needed (yfinance 1.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Price' in df.columns and 'Close' not in df.columns:
            df.rename(columns={'Price': 'Close'}, inplace=True)

        close = df['Close'].dropna()
        if len(close) < 50:
            return {'status': 'Unknown', 'emoji': '⚪', 'cmp': 0.0, 'sma50': 0.0, 'sma200': 0.0, 'change_pct': 0.0}

        cmp     = float(close.iloc[-1])
        sma50   = float(close.rolling(50).mean().iloc[-1])
        sma200  = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
        prev_close = float(close.iloc[-2]) if len(close) >= 2 else cmp
        change_pct = round((cmp - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0.0

        if cmp > sma50 and sma50 > sma200:
            status, emoji = 'Bullish', '🟢'
        elif cmp > sma200:
            status, emoji = 'Caution', '🟡'
        else:
            status, emoji = 'Bearish', '🔴'

        return {
            'status':     status,
            'emoji':      emoji,
            'cmp':        round(cmp, 2),
            'sma50':      round(sma50, 2),
            'sma200':     round(sma200, 2),
            'change_pct': change_pct,
        }
    except Exception as e:
        print(f"get_market_condition error: {e}")
        return {'status': 'Unknown', 'emoji': '⚪', 'cmp': 0.0, 'sma50': 0.0, 'sma200': 0.0, 'change_pct': 0.0}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nifty50_returns() -> dict:
    """
    Fetches Nifty 50 index trailing returns over 1M, 3M, 6M, 12M
    for use in per-stock Relative Strength (RS Rating) calculation.
    Cached for 1 hour — benchmark returns don't need sub-minute freshness.
    """
    try:
        df = yf.download("^NSEI", period="400d", interval="1d", progress=False)
        if df is None or df.empty:
            return {}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if 'Price' in df.columns and 'Close' not in df.columns:
            df.rename(columns={'Price': 'Close'}, inplace=True)

        close = df['Close'].dropna()
        n = len(close)
        cmp = float(close.iloc[-1])

        def _ret(periods: int) -> float:
            if n >= periods + 1:
                base = float(close.iloc[-(periods + 1)])
                return (cmp - base) / base * 100.0 if base > 0 else 0.0
            return 0.0

        return {
            '1m':  round(_ret(21),  2),
            '3m':  round(_ret(63),  2),
            '6m':  round(_ret(126), 2),
            '12m': round(_ret(252), 2),
        }
    except Exception as e:
        print(f"fetch_nifty50_returns error: {e}")
        return {}


def calculate_rs_rating(df: pd.DataFrame, nifty_returns: dict) -> float:
    """
    Calculates Relative Strength Rating of a stock vs Nifty 50.
    Uses O'Neil-style weighted trailing returns: 1M(40%), 3M(20%), 6M(20%), 12M(20%).

    Returns a score on 1-99 scale:
      - RS Rating > 80  : Strong outperformer (top 20% vs index)
      - RS Rating 50-80 : Moderate outperformer
      - RS Rating < 50  : Underperformer (avoid buying)

    Args:
        df: Stock OHLCV DataFrame with a 'Close' column.
        nifty_returns: Dict from fetch_nifty50_returns() with keys '1m','3m','6m','12m'.
    Returns:
        RS Rating float (1-99). Returns 50.0 as neutral default if data is insufficient.
    """
    if df is None or len(df) < 21 or not nifty_returns:
        return 50.0
    try:
        close = df['Close']
        n = len(close)
        cmp = float(close.iloc[-1])

        def _ret(periods: int) -> float:
            if n >= periods + 1:
                base = float(close.iloc[-(periods + 1)])
                return (cmp - base) / base * 100.0 if base > 0 else 0.0
            return 0.0

        stock_returns = {
            '1m':  _ret(21),
            '3m':  _ret(63),
            '6m':  _ret(126),
            '12m': _ret(252),
        }
        weights = {'1m': 0.40, '3m': 0.20, '6m': 0.20, '12m': 0.20}

        # Weighted excess return vs Nifty benchmark
        rs_delta = sum(
            (stock_returns.get(k, 0.0) - nifty_returns.get(k, 0.0)) * w
            for k, w in weights.items()
        )
        # Normalize to 1-99 scale centered at 50 (neutral = same as index)
        rs_rating = max(1.0, min(99.0, 50.0 + rs_delta))
        return round(rs_rating, 1)
    except Exception as e:
        print(f"calculate_rs_rating error: {e}")
        return 50.0


@st.cache_data(ttl=86400, show_spinner=False)
def get_market_cap_cr(symbol: str) -> float:
    """
    Returns market capitalisation in Indian Crores (₹ Cr) for a given NSE symbol.
    Cached for 24 hours — market cap doesn't change meaningfully intraday.
    Returns 0.0 if the lookup fails (symbol delisted, API error, etc.).
    """
    try:
        sym = symbol.strip().upper()
        if not sym.endswith('.NS'):
            sym = f"{sym}.NS"
        info = yf.Ticker(sym).info
        mc = info.get('marketCap', 0) or 0
        return round(mc / 1e7, 1)  # Convert Rupees → Crores (1 Crore = 10,000,000)
    except Exception:
        return 0.0
