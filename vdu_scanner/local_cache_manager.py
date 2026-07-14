import os
import time
import pandas as pd
from datetime import datetime, timedelta
import database

# We no longer use local filesystem cache
# All cache is stored in the Turso historical_ohlcv table

CACHE_TTL_SECONDS = 8 * 3600

def get_cached_ohlcv(symbol: str, timeframe: str = "1d", ignore_ttl: bool = False) -> pd.DataFrame | None:
    """Returns the cached DataFrame if it exists and is not stale (unless ignore_ttl=True)."""
    clean_sym = symbol.strip().upper().replace(".NS", "")
    try:
        conn = database.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT date, open, high, low, close, volume FROM historical_ohlcv WHERE symbol=? AND timeframe=? ORDER BY date ASC", (clean_sym, timeframe))
        rows = cur.fetchall()
        
        if not rows:
            return None
            
        df = pd.DataFrame(rows)
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
        
        # Process in chunks of 500 to avoid SQLite limits
        chunk_size = 500
        for i in range(0, len(clean_syms), chunk_size):
            chunk = clean_syms[i:i+chunk_size]
            placeholders = ','.join(['?' for _ in chunk])
            query = f"SELECT symbol, date, open, high, low, close, volume FROM historical_ohlcv WHERE timeframe=? AND symbol IN ({placeholders}) ORDER BY date ASC"
            
            args = [timeframe] + chunk
            cur.execute(query, args)
            rows = cur.fetchall()
            
            if rows:
                df_all = pd.DataFrame(rows)
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
    """Saves the DataFrame to the Turso DB cache."""
    if df is None or df.empty:
        return
        
    clean_sym = symbol.strip().upper().replace(".NS", "")
    
    try:
        conn = database.get_connection()
        client = conn.client
        import libsql_client
        
        query = """
        INSERT INTO historical_ohlcv (symbol, timeframe, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, date) DO UPDATE SET
        open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume
        """
        
        stmts = []
        for _, row in df.iterrows():
            date_str = str(row['Date']).split(' ')[0] if pd.notnull(row['Date']) else None
            if not date_str:
                continue
                
            args = [
                clean_sym,
                timeframe,
                date_str,
                float(row['Open']) if pd.notnull(row['Open']) else 0.0,
                float(row['High']) if pd.notnull(row['High']) else 0.0,
                float(row['Low']) if pd.notnull(row['Low']) else 0.0,
                float(row['Close']) if pd.notnull(row['Close']) else 0.0,
                int(row['Volume']) if pd.notnull(row['Volume']) else 0
            ]
            stmts.append(libsql_client.Statement(query, args))
            
        chunk_size = 200
        for j in range(0, len(stmts), chunk_size):
            client.execute_batch(stmts[j:j+chunk_size])
            
    except Exception as e:
        print(f"Error saving DB cache for {clean_sym}: {e}")

