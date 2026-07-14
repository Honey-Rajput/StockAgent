import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_data.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Enable WAL mode for better concurrency
    conn.execute('PRAGMA journal_mode=WAL;')
    return conn

def init_db():
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_daily (
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                PRIMARY KEY (symbol, date)
            )
        """)
        # Index for faster retrieval by symbol and date sorting
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_daily_symbol_date ON ohlcv_daily (symbol, date DESC)")
        conn.commit()
    finally:
        conn.close()

def get_latest_date(symbol: str) -> str | None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM ohlcv_daily WHERE symbol = ?", (symbol.strip().upper(),))
        row = cur.fetchone()
        if row and row[0]:
            return row[0][:10]  # Return YYYY-MM-DD
        return None
    finally:
        conn.close()

def get_historical_data(symbol: str, limit: int = 500) -> pd.DataFrame | None:
    conn = get_connection()
    try:
        query = """
            SELECT date as Date, open as Open, high as High, low as Low, close as Close, volume as Volume 
            FROM ohlcv_daily 
            WHERE symbol = ? 
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol.strip().upper(), limit))
        if df.empty:
            return None
        
        # Reverse to get chronological order (oldest to newest)
        df = df.iloc[::-1].reset_index(drop=True)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error reading from market_data_db for {symbol}: {e}")
        return None
    finally:
        conn.close()

def save_data(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
        
    conn = get_connection()
    try:
        # Create a copy to avoid mutating the original
        df_to_save = df.copy()
        
        # Ensure column names are correct
        col_map = {c: c.lower() for c in df_to_save.columns}
        df_to_save.rename(columns=col_map, inplace=True)
        
        if 'date' not in df_to_save.columns:
            if df_to_save.index.name == 'Date' or df_to_save.index.name == 'date':
                df_to_save = df_to_save.reset_index()
                df_to_save.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
            else:
                return
                
        # Convert date to string format YYYY-MM-DD HH:MM:SS
        df_to_save['date'] = pd.to_datetime(df_to_save['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        df_to_save['symbol'] = symbol.strip().upper()
        
        # Select only required columns
        req_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        # Add missing columns with NaN if necessary (shouldn't happen)
        for col in req_cols:
            if col not in df_to_save.columns:
                df_to_save[col] = None
                
        df_to_save = df_to_save[req_cols]
        
        # Insert using sqlite3 executemany with UPSERT
        records = df_to_save.to_records(index=False)
        
        sql = """
            INSERT INTO ohlcv_daily (symbol, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, date) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume
        """
        conn.executemany(sql, records.tolist())
        conn.commit()
    except Exception as e:
        print(f"Error saving data to market_data_db for {symbol}: {e}")
        conn.rollback()
    finally:
        conn.close()

# Initialize DB on import
init_db()
