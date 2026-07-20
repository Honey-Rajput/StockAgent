import os
import time
import pandas as pd
from datetime import datetime, timedelta
import database

# We no longer use local filesystem cache
# All cache is stored in the Neon PostgreSQL historical_ohlcv table

CACHE_TTL_SECONDS = 8 * 3600

def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resamples a daily OHLCV DataFrame to a higher timeframe."""
    if df is None or df.empty:
        return df
    
    reset_index = False
    if 'Date' in df.columns:
        df = df.set_index('Date')
        reset_index = True
        
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    agg_cols = {col: agg_dict[col] for col in df.columns if col in agg_dict}
    
    resampled = df.resample(freq).agg(agg_cols).dropna()
    
    if reset_index:
        resampled = resampled.reset_index()
    return resampled

def get_cached_ohlcv(symbol: str, timeframe: str = "1d", ignore_ttl: bool = False) -> pd.DataFrame | None:
    """Returns the cached DataFrame if it exists and is not stale (unless ignore_ttl=True)."""
    clean_sym = symbol.strip().upper().replace(".NS", "")
    try:
        conn = database.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT date, open, high, low, close, volume FROM historical_ohlcv WHERE symbol=%s AND timeframe=%s ORDER BY date ASC", (clean_sym, timeframe))
        rows = cur.fetchall()
        
        if not rows:
            return None
            
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=columns)
        df.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error reading DB cache for {clean_sym}: {e}")
        return None

def bulk_get_cached_ohlcv(symbols: list, timeframe: str = "1d") -> dict:
    """Returns a dictionary of DataFrames for multiple symbols in a single query."""
    if not symbols:
        return {}
        
    clean_syms = [s.strip().upper().replace(".NS", "") for s in symbols]
    result_dict = {}
    
    try:
        conn = database.get_connection()
        cur = conn.cursor()
        
        # Process in chunks of 500
        chunk_size = 500
        for i in range(0, len(clean_syms), chunk_size):
            chunk = clean_syms[i:i+chunk_size]
            placeholders = ','.join(['%s' for _ in chunk])
            query = f"SELECT symbol, date, open, high, low, close, volume FROM historical_ohlcv WHERE timeframe=%s AND symbol IN ({placeholders}) ORDER BY date ASC"
            
            args = [timeframe] + chunk
            cur.execute(query, args)
            rows = cur.fetchall()
            
            if rows:
                columns = [desc[0] for desc in cur.description]
                df_all = pd.DataFrame(rows, columns=columns)
                df_all.rename(columns={"symbol": "Symbol", "date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
                df_all['Date'] = pd.to_datetime(df_all['Date'])
                for sym, group in df_all.groupby('Symbol'):
                    df_sym = group.drop(columns=['Symbol']).reset_index(drop=True)
                    result_dict[sym] = df_sym
                    
        return result_dict
    except Exception as e:
        print(f"Error reading bulk DB cache: {e}")
        return result_dict

def save_to_cache(symbol: str, df: pd.DataFrame, timeframe: str = "1d"):
    """Saves the DataFrame to the Neon DB cache."""
    if df is None or df.empty:
        return
        
    clean_sym = symbol.strip().upper().replace(".NS", "")
    
    try:
        conn = database.get_connection()
        cur = conn.cursor()
        
        # Optimization: Only upsert the most recent day (which might update) and any new days
        cur.execute("SELECT MAX(date) FROM historical_ohlcv WHERE symbol=%s AND timeframe=%s", (clean_sym, timeframe))
        max_date_row = cur.fetchone()
        if max_date_row and max_date_row['max']:
            max_date = pd.to_datetime(max_date_row['max'])
            # Filter df to only keep dates >= max_date
            df = df[df['Date'] >= max_date]
            
        if df.empty:
            return
            
        query = """
        INSERT INTO historical_ohlcv (symbol, timeframe, date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT(symbol, timeframe, date) DO UPDATE SET
        open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
        """
        
        argslist = []
        for _, row in df.iterrows():
            date_str = str(row['Date']).split(' ')[0] if pd.notnull(row['Date']) else None
            if not date_str:
                continue
                
            args = (
                clean_sym,
                timeframe,
                date_str,
                float(row['Open']) if pd.notnull(row['Open']) else 0.0,
                float(row['High']) if pd.notnull(row['High']) else 0.0,
                float(row['Low']) if pd.notnull(row['Low']) else 0.0,
                float(row['Close']) if pd.notnull(row['Close']) else 0.0,
                int(row['Volume']) if pd.notnull(row['Volume']) else 0
            )
            argslist.append(args)
            
        database.execute_values(cur, query, argslist, page_size=200)
            
    except Exception as e:
        print(f"Error saving DB cache for {clean_sym}: {e}")

