# database.py
import os
import psycopg2
import pandas as pd
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Resolve and load environment variables from the parent directory's .env file
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(parent_dir, ".env")
load_dotenv(env_path)

DATABASE_URL = os.getenv("Database_URL")

def get_connection():
    """
    Establishes a connection to the PostgreSQL (Neon) database.
    """
    if not DATABASE_URL:
        raise ValueError("Database_URL is not set in the environment or .env file.")
    return psycopg2.connect(DATABASE_URL)

def init_db() -> bool:
    """
    Initializes the database schema by creating the ai_chart_patterns, 
    scanned_breakouts, scanned_squeezes, and scan_logs tables if not present.
    """
    queries = [
        """
        CREATE TABLE IF NOT EXISTS ai_chart_patterns (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            pattern_name VARCHAR(50) NOT NULL,
            confidence VARCHAR(20) NOT NULL,
            direction VARCHAR(20) NOT NULL,
            analysis_text TEXT NOT NULL,
            price_data_snapshot TEXT NOT NULL,
            analyzed_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, analyzed_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_breakouts (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            today_volume BIGINT,
            dry_avg_vol DOUBLE PRECISION,
            volume_ratio DOUBLE PRECISION,
            dry_days_count INT,
            dry_spikes INT,
            market_cap_cr DOUBLE PRECISION,
            signal_strength DOUBLE PRECISION,
            above_50dma BOOLEAN,
            dry_start_date DATE,
            dry_end_date DATE,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_squeezes (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            range_5d DOUBLE PRECISION,
            range_prev DOUBLE PRECISION,
            vol_ratio DOUBLE PRECISION,
            squeeze_score DOUBLE PRECISION,
            market_cap_cr DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_gapups (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            prev_close DOUBLE PRECISION,
            open_price DOUBLE PRECISION,
            cmp DOUBLE PRECISION,
            gap_pct DOUBLE PRECISION,
            volume BIGINT,
            day_change_pct DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_trend_setups (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            setup_type VARCHAR(50) NOT NULL,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, setup_type, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scan_logs (
            scan_date DATE PRIMARY KEY,
            total_scanned INT NOT NULL,
            breakouts_found INT NOT NULL,
            squeezes_found INT NOT NULL,
            completed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]
    
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        for q in queries:
            cur.execute(q)
        conn.commit()
        cur.close()
        print("Database initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing PostgreSQL database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_pattern_by_date(symbol: str, date_str: str) -> dict | None:
    """
    Tries to retrieve a cached technical pattern analysis for a stock on a specific date.
    Helps prevent repeated external API requests within the same day.
    """
    query = """
    SELECT symbol, pattern_name, confidence, direction, analysis_text, price_data_snapshot, analyzed_date
    FROM ai_chart_patterns
    WHERE UPPER(symbol) = %s AND analyzed_date = %s;
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (symbol.strip().upper(), date_str))
        row = cur.fetchone()
        cur.close()
        if row:
            row_dict = dict(row)
            # Standardize date output as string
            row_dict['analyzed_date'] = row_dict['analyzed_date'].strftime("%Y-%m-%d")
            return row_dict
        return None
    except Exception as e:
        print(f"Error loading cached pattern from database: {e}")
        return None
    finally:
        if conn:
            conn.close()

def save_pattern(
    symbol: str, 
    pattern_name: str, 
    confidence: str, 
    direction: str, 
    analysis_text: str, 
    price_data_snapshot: str, 
    date_str: str
) -> bool:
    """
    Saves a newly generated technical chart pattern analysis into the database.
    Uses ON CONFLICT to perform an UPSERT in case a write collision occurs.
    """
    insert_query = """
    INSERT INTO ai_chart_patterns (symbol, pattern_name, confidence, direction, analysis_text, price_data_snapshot, analyzed_date)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (symbol, analyzed_date) 
    DO UPDATE SET 
        pattern_name = EXCLUDED.pattern_name,
        confidence = EXCLUDED.confidence,
        direction = EXCLUDED.direction,
        analysis_text = EXCLUDED.analysis_text,
        price_data_snapshot = EXCLUDED.price_data_snapshot,
        created_at = CURRENT_TIMESTAMP;
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(insert_query, (
            symbol.strip().upper(),
            pattern_name.strip(),
            confidence.strip(),
            direction.strip(),
            analysis_text.strip(),
            price_data_snapshot.strip(),
            date_str
        ))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving pattern analysis to database: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_recent_patterns(limit: int = 10) -> list[dict]:
    """
    Fetches recently scanned chart pattern entries from the database.
    Allows user to quickly review previous stock findings.
    """
    query = """
    SELECT symbol, pattern_name, confidence, direction, analysis_text, analyzed_date, created_at
    FROM ai_chart_patterns
    ORDER BY created_at DESC
    LIMIT %s;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (limit,))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['analyzed_date'] = r_dict['analyzed_date'].strftime("%Y-%m-%d")
            r_dict['created_at'] = r_dict['created_at'].strftime("%Y-%m-%d %I:%M %p")
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading recent patterns from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def has_scanned_today(date_str: str) -> dict | None:
    """
    Checks if a full market scan was logged on a specific day.
    """
    query = "SELECT scan_date, total_scanned, breakouts_found, squeezes_found FROM scan_logs WHERE scan_date = %s;"
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str,))
        row = cur.fetchone()
        cur.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"Error checking daily scan log in database: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_cached_breakouts(date_str: str) -> list[dict]:
    """
    Retrieves the cached VDU breakouts scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, today_volume, dry_avg_vol, 
           volume_ratio, dry_days_count, dry_spikes, market_cap_cr, signal_strength, 
           above_50dma, dry_start_date, dry_end_date, scan_date
    FROM scanned_breakouts
    WHERE scan_date = %s;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str,))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['dry_start_date'] = pd.to_datetime(r_dict['dry_start_date'])
            r_dict['dry_end_date'] = pd.to_datetime(r_dict['dry_end_date'])
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached breakouts from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_cached_squeezes(date_str: str) -> list[dict]:
    """
    Retrieves the cached coiled VCP squeezes scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, range_5d, range_prev, vol_ratio, 
           squeeze_score, market_cap_cr, scan_date
    FROM scanned_squeezes
    WHERE scan_date = %s;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str,))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached squeezes from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_cached_gapups(date_str: str) -> list[dict]:
    """
    Retrieves the cached Gap-Up setups scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, prev_close, open_price, cmp, gap_pct, volume, day_change_pct, scan_date
    FROM scanned_gapups
    WHERE scan_date = %s;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str,))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached gapups from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_cached_trend_setups(date_str: str, setup_type: str) -> list[dict]:
    """
    Retrieves the cached technical trend setups scanned on a specific date for a setup_type.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, setup_type, scan_date
    FROM scanned_trend_setups
    WHERE scan_date = %s AND setup_type = %s;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str, setup_type))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached trend setups for {setup_type} from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def save_scan_results(date_str: str, breakouts: list[dict], squeezes: list[dict], gapups: list[dict], trend_setups: list[dict], total_scanned: int) -> bool:
    """
    Saves the full market scan results and logs the completion.
    Uses clean transactions to perform daily upsert overrides.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # 1. Clean existing records for this date
        cur.execute("DELETE FROM scanned_breakouts WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_squeezes WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_gapups WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_trend_setups WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scan_logs WHERE scan_date = %s;", (date_str,))
        
        # 2. Insert new breakouts
        insert_breakout_query = """
        INSERT INTO scanned_breakouts (symbol, company_name, cmp, day_change_pct, today_volume, 
                                      dry_avg_vol, volume_ratio, dry_days_count, dry_spikes, 
                                      market_cap_cr, signal_strength, above_50dma, dry_start_date, 
                                      dry_end_date, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in breakouts:
            cur.execute(insert_breakout_query, (
                str(r['symbol']), 
                str(r['company_name']) if r['company_name'] else "", 
                float(r['cmp']), 
                float(r['day_change_pct']), 
                int(r['today_volume']),
                float(r['dry_avg_vol']), 
                float(r['volume_ratio']), 
                int(r['dry_days_count']), 
                int(r['dry_spikes']),
                float(r['market_cap_cr']), 
                float(r['signal_strength']), 
                bool(r['above_50dma']),
                r['dry_start_date'].strftime("%Y-%m-%d") if hasattr(r['dry_start_date'], 'strftime') else str(r['dry_start_date']), 
                r['dry_end_date'].strftime("%Y-%m-%d") if hasattr(r['dry_end_date'], 'strftime') else str(r['dry_end_date']),
                date_str
            ))
            
        # 3. Insert new squeezes
        insert_squeeze_query = """
        INSERT INTO scanned_squeezes (symbol, company_name, cmp, range_5d, range_prev, 
                                     vol_ratio, squeeze_score, market_cap_cr, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in squeezes:
            cur.execute(insert_squeeze_query, (
                str(r['symbol']), 
                str(r['company_name']) if r['company_name'] else "", 
                float(r['cmp']), 
                float(r['range_5d']), 
                float(r['range_prev']),
                float(r['vol_ratio']), 
                float(r['squeeze_score']), 
                float(r['market_cap_cr']), 
                date_str
            ))
            
        # 3.5. Insert new gapups
        insert_gapup_query = """
        INSERT INTO scanned_gapups (symbol, company_name, prev_close, open_price, cmp, gap_pct, volume, day_change_pct, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in gapups:
            cur.execute(insert_gapup_query, (
                str(r['symbol']), 
                str(r['company_name']) if r['company_name'] else "", 
                float(r['prev_close']),
                float(r['open_price']),
                float(r['cmp']), 
                float(r['gap_pct']),
                int(r['volume']), 
                float(r['day_change_pct']), 
                date_str
            ))
            
        # 3.8. Insert new trend setups (above_ma, support_ma, crossover_ma)
        insert_trend_query = """
        INSERT INTO scanned_trend_setups (symbol, company_name, cmp, day_change_pct, setup_type, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s);
        """
        for r in trend_setups:
            cur.execute(insert_trend_query, (
                str(r['symbol']),
                str(r['company_name']) if r['company_name'] else "",
                float(r['cmp']),
                float(r['day_change_pct']),
                str(r['setup_type']),
                date_str
            ))
            
        # 4. Insert execution log
        cur.execute("""
        INSERT INTO scan_logs (scan_date, total_scanned, breakouts_found, squeezes_found)
        VALUES (%s, %s, %s, %s);
        """, (date_str, total_scanned, len(breakouts), len(squeezes)))
        
        conn.commit()
        cur.close()
        print(f"Cached {len(breakouts)} breakouts, {len(squeezes)} squeezes, {len(gapups)} gapups, and {len(trend_setups)} trend setups in Neon for {date_str}.")
        return True
    except Exception as e:
        if conn:
            conn.close()
        print(f"Error saving daily scan results to PostgreSQL: {e}")
        return False
    finally:
        if conn:
            conn.close()
