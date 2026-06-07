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
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
    try:
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 8000;")
        cur.close()
    except Exception as e:
        print(f"Warning: Could not set session statement_timeout: {e}")
    return conn

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
            run_up_200 DOUBLE PRECISION,
            run_up_52w DOUBLE PRECISION,
            is_early BOOLEAN,
            UNIQUE(symbol, setup_type, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_wt_cross (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            wt_value DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            volume BIGINT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
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
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_monthly_momentum (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            ema8 DOUBLE PRECISION,
            ema12 DOUBLE PRECISION,
            ema20 DOUBLE PRECISION,
            roc6 DOUBLE PRECISION,
            rsi_monthly DOUBLE PRECISION,
            volume BIGINT,
            vol_sma12 DOUBLE PRECISION,
            market_cap_cr DOUBLE PRECISION,
            momentum_score DOUBLE PRECISION,
            buy_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            target_price DOUBLE PRECISION,
            confidence VARCHAR(50),
            recommendation TEXT,
            return_1m DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_weekly_momentum (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            weekly_chg_pct DOUBLE PRECISION,
            prev_close DOUBLE PRECISION,
            curr_open DOUBLE PRECISION,
            close_sma20 DOUBLE PRECISION,
            rsi_weekly DOUBLE PRECISION,
            cci_weekly DOUBLE PRECISION,
            volume BIGINT,
            vol_sma20 DOUBLE PRECISION,
            vol_ratio DOUBLE PRECISION,
            market_cap_cr DOUBLE PRECISION,
            weekly_score DOUBLE PRECISION,
            buy_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            target_price DOUBLE PRECISION,
            confidence VARCHAR(50),
            recommendation TEXT,
            return_1m DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_vcs (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            vcs_score DOUBLE PRECISION,
            volume BIGINT,
            buy_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            target_price DOUBLE PRECISION,
            confidence VARCHAR(50),
            recommendation TEXT,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_vpa (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            volume BIGINT,
            vpa_score INT,
            daily_major INT,
            daily_mid INT,
            daily_minor INT,
            weekly_major INT,
            weekly_mid INT,
            weekly_minor INT,
            monthly_major INT,
            monthly_mid INT,
            monthly_minor INT,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """
    ]
    
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        for q in queries:
            cur.execute(q)
            
        # Safely migrate existing tables if columns are missing
        migrations = [
            "ALTER TABLE scanned_breakouts ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_breakouts ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_breakouts ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_breakouts ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_breakouts ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            
            "ALTER TABLE scanned_squeezes ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_squeezes ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_squeezes ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_squeezes ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_squeezes ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            
            "ALTER TABLE scanned_gapups ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_gapups ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_gapups ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_gapups ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_gapups ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS run_up_200 DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS run_up_52w DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS is_early BOOLEAN;",
            "ALTER TABLE scanned_breakouts ADD COLUMN IF NOT EXISTS above_200dma BOOLEAN DEFAULT FALSE;",
            
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS wt2_value DOUBLE PRECISION;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS buy_signal BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS wt_diff DOUBLE PRECISION;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS above_20sma BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS above_50sma BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS above_200sma BOOLEAN DEFAULT FALSE;",
            "ALTER TABLE scanned_wt_cross ADD COLUMN IF NOT EXISTS volume BIGINT;",
            
            # Scanned Monthly Momentum Table Migrations
            "ALTER TABLE scanned_monthly_momentum ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_monthly_momentum ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_monthly_momentum ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_monthly_momentum ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_monthly_momentum ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            "ALTER TABLE scanned_monthly_momentum ADD COLUMN IF NOT EXISTS return_1m DOUBLE PRECISION;",
            
            # Scanned Weekly Momentum Table Migrations
            "ALTER TABLE scanned_weekly_momentum ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_weekly_momentum ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_weekly_momentum ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_weekly_momentum ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_weekly_momentum ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            "ALTER TABLE scanned_weekly_momentum ADD COLUMN IF NOT EXISTS return_1m DOUBLE PRECISION;",

            "ALTER TABLE scanned_vcs ADD COLUMN IF NOT EXISTS buy_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_vcs ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_vcs ADD COLUMN IF NOT EXISTS target_price DOUBLE PRECISION;",
            "ALTER TABLE scanned_vcs ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_vcs ADD COLUMN IF NOT EXISTS recommendation TEXT;"
        ]
        for m in migrations:
            try:
                cur.execute(m)
            except Exception as mig_ex:
                print(f"Migration column note/error: {mig_ex}")
                conn.rollback() # in case statement fails, rollback transaction so we can continue
                
        conn.commit()
        cur.close()
        print("Database initialized and migrated successfully.")
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

def get_available_scan_dates() -> list[str]:
    """
    Retrieves all dates that have completed daily scan logs, sorted descending.
    """
    query = "SELECT DISTINCT scan_date FROM scan_logs ORDER BY scan_date DESC;"
    conn = None
    dates = []
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        dates = [r[0].strftime("%Y-%m-%d") if hasattr(r[0], 'strftime') else str(r[0]) for r in rows]
    except Exception as e:
        print(f"Error loading scan dates from database: {e}")
    finally:
        if conn:
            conn.close()
    return dates


def get_cached_breakouts(date_str: str) -> list[dict]:
    """
    Retrieves the cached VDU breakouts scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, today_volume, dry_avg_vol, 
           volume_ratio, dry_days_count, dry_spikes, market_cap_cr, signal_strength, 
           above_50dma, above_200dma, dry_start_date, dry_end_date, scan_date,
           buy_price, exit_price, target_price, confidence, recommendation
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
           squeeze_score, market_cap_cr, scan_date,
           buy_price, exit_price, target_price, confidence, recommendation
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
    SELECT symbol, company_name, prev_close, open_price, cmp, gap_pct, volume, day_change_pct, scan_date,
           buy_price, exit_price, target_price, confidence, recommendation
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
    SELECT symbol, company_name, cmp, day_change_pct, setup_type, scan_date,
           buy_price, exit_price, target_price, confidence, recommendation,
           run_up_200, run_up_52w, is_early
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

def get_cached_wt_cross(date_str: str) -> list[dict]:
    """
    Retrieves the cached WT Cross setups scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, wt_value, scan_date,
           buy_price, exit_price, target_price, confidence, recommendation,
           wt2_value, buy_signal, wt_diff, above_20sma, above_50sma, above_200sma, volume
    FROM scanned_wt_cross
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
            # Ensure buy_signal is always a bool (may be None from old rows)
            r_dict['buy_signal'] = bool(r_dict.get('buy_signal', False))
            r_dict['wt2_value'] = float(r_dict.get('wt2_value') or 0.0)
            r_dict['wt_diff'] = float(r_dict.get('wt_diff') or 0.0)
            r_dict['above_20sma'] = bool(r_dict.get('above_20sma', False))
            r_dict['above_50sma'] = bool(r_dict.get('above_50sma', False))
            r_dict['above_200sma'] = bool(r_dict.get('above_200sma', False))
            r_dict['volume'] = int(r_dict.get('volume') or 0)
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached WT Cross from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_cached_vcs(date_str: str) -> list[dict]:
    """
    Retrieves the cached VCS setups scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, vcs_score, volume, scan_date,
           buy_price, exit_price, target_price, confidence, recommendation
    FROM scanned_vcs
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
            r_dict['vcs_score'] = float(r_dict.get('vcs_score') or 0.0)
            r_dict['volume'] = int(r_dict.get('volume') or 0)
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached VCS from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_cached_vpa(date_str: str) -> list[dict]:
    """
    Retrieves the cached VPA trend setups scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, volume, vpa_score,
           daily_major, daily_mid, daily_minor,
           weekly_major, weekly_mid, weekly_minor,
           monthly_major, monthly_mid, monthly_minor,
           scan_date
    FROM scanned_vpa
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
            r_dict['volume'] = int(r_dict.get('volume') or 0)
            r_dict['score'] = int(r_dict.get('vpa_score') or 0)
            
            r_dict['daily'] = {
                "major": int(r_dict.get('daily_major') or 0),
                "mid": int(r_dict.get('daily_mid') or 0),
                "minor": int(r_dict.get('daily_minor') or 0)
            }
            r_dict['weekly'] = {
                "major": int(r_dict.get('weekly_major') or 0),
                "mid": int(r_dict.get('weekly_mid') or 0),
                "minor": int(r_dict.get('weekly_minor') or 0)
            }
            r_dict['monthly'] = {
                "major": int(r_dict.get('monthly_major') or 0),
                "mid": int(r_dict.get('monthly_mid') or 0),
                "minor": int(r_dict.get('monthly_minor') or 0)
            }
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached VPA from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def save_scan_results(date_str: str, breakouts: list[dict], squeezes: list[dict], gapups: list[dict], trend_setups: list[dict], wt_cross: list[dict], total_scanned: int, vcs_results: list[dict] = None, vpa_results: list[dict] = None) -> bool:
    """
    Saves the full market scan results and logs the completion.
    Uses clean transactions to perform daily upsert overrides.
    """
    if vcs_results is None:
        vcs_results = []
    if vpa_results is None:
        vpa_results = []
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # 1. Clean existing records for this date
        cur.execute("DELETE FROM scanned_breakouts WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_squeezes WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_gapups WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_trend_setups WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_wt_cross WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_vcs WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scanned_vpa WHERE scan_date = %s;", (date_str,))
        cur.execute("DELETE FROM scan_logs WHERE scan_date = %s;", (date_str,))
        
        # 2. Insert new breakouts
        insert_breakout_query = """
        INSERT INTO scanned_breakouts (symbol, company_name, cmp, day_change_pct, today_volume, 
                                      dry_avg_vol, volume_ratio, dry_days_count, dry_spikes, 
                                      market_cap_cr, signal_strength, above_50dma, above_200dma, dry_start_date, 
                                      dry_end_date, scan_date, buy_price, exit_price, target_price, 
                                      confidence, recommendation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
                bool(r.get('above_50dma', False)),
                bool(r.get('above_200dma', False)),
                r['dry_start_date'].strftime("%Y-%m-%d") if hasattr(r['dry_start_date'], 'strftime') else str(r['dry_start_date']), 
                r['dry_end_date'].strftime("%Y-%m-%d") if hasattr(r['dry_end_date'], 'strftime') else str(r['dry_end_date']),
                date_str,
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None
            ))
            
        # 3. Insert new squeezes
        insert_squeeze_query = """
        INSERT INTO scanned_squeezes (symbol, company_name, cmp, range_5d, range_prev, 
                                     vol_ratio, squeeze_score, market_cap_cr, scan_date,
                                     buy_price, exit_price, target_price, confidence, recommendation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
                date_str,
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None
            ))
            
        # 3.5. Insert new gapups
        insert_gapup_query = """
        INSERT INTO scanned_gapups (symbol, company_name, prev_close, open_price, cmp, gap_pct, volume, 
                                   day_change_pct, scan_date, buy_price, exit_price, target_price, 
                                   confidence, recommendation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
                date_str,
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None
            ))
            
        # 3.8. Insert new trend setups (above_ma, support_ma, crossover_ma)
        insert_trend_query = """
        INSERT INTO scanned_trend_setups (symbol, company_name, cmp, day_change_pct, setup_type, scan_date,
                                         buy_price, exit_price, target_price, confidence, recommendation,
                                         run_up_200, run_up_52w, is_early)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in trend_setups:
            cur.execute(insert_trend_query, (
                str(r['symbol']),
                str(r['company_name']) if r['company_name'] else "",
                float(r['cmp']),
                float(r['day_change_pct']),
                str(r['setup_type']),
                date_str,
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None,
                float(r['run_up_200']) if r.get('run_up_200') is not None else None,
                float(r['run_up_52w']) if r.get('run_up_52w') is not None else None,
                bool(r['is_early']) if r.get('is_early') is not None else None
            ))
 
        # 3.9. Insert new WT Cross setups
        insert_wt_query = """
        INSERT INTO scanned_wt_cross (symbol, company_name, cmp, day_change_pct, wt_value, scan_date,
                                     buy_price, exit_price, target_price, confidence, recommendation,
                                     wt2_value, buy_signal, wt_diff, above_20sma, above_50sma, above_200sma, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in wt_cross:
            cur.execute(insert_wt_query, (
                str(r['symbol']),
                str(r['company_name']) if r['company_name'] else "",
                float(r['cmp']),
                float(r['day_change_pct']),
                float(r['wt_value']),
                date_str,
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None,
                float(r['wt2_value']) if r.get('wt2_value') is not None else None,
                bool(r.get('buy_signal', False)),
                float(r['wt_diff']) if r.get('wt_diff') is not None else None,
                bool(r.get('above_20sma', False)),
                bool(r.get('above_50sma', False)),
                bool(r.get('above_200sma', False)),
                int(r.get('volume', 0))
            ))
            
        # 3.10. Insert new VCS setups
        insert_vcs_query = """
        INSERT INTO scanned_vcs (symbol, company_name, cmp, day_change_pct, vcs_score, volume, scan_date,
                                 buy_price, exit_price, target_price, confidence, recommendation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in vcs_results:
            cur.execute(insert_vcs_query, (
                str(r['symbol']),
                str(r['company_name']) if r['company_name'] else "",
                float(r['cmp']),
                float(r['day_change_pct']),
                float(r['vcs_score']),
                int(r.get('volume', 0)),
                date_str,
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None
            ))
            
        # 3.11. Insert new VPA setups
        insert_vpa_query = """
        INSERT INTO scanned_vpa (symbol, company_name, cmp, day_change_pct, volume, vpa_score, 
                                 daily_major, daily_mid, daily_minor, 
                                 weekly_major, weekly_mid, weekly_minor, 
                                 monthly_major, monthly_mid, monthly_minor, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in vpa_results:
            cur.execute(insert_vpa_query, (
                str(r['symbol']),
                str(r['company_name']) if r['company_name'] else "",
                float(r['cmp']),
                float(r['day_change_pct']),
                int(r.get('volume', 0)),
                int(r.get('score', 0)),
                int(r.get('daily', {}).get('major', 0)),
                int(r.get('daily', {}).get('mid', 0)),
                int(r.get('daily', {}).get('minor', 0)),
                int(r.get('weekly', {}).get('major', 0)),
                int(r.get('weekly', {}).get('mid', 0)),
                int(r.get('weekly', {}).get('minor', 0)),
                int(r.get('monthly', {}).get('major', 0)),
                int(r.get('monthly', {}).get('mid', 0)),
                int(r.get('monthly', {}).get('minor', 0)),
                date_str
            ))
            
        # 4. Insert execution log
        cur.execute("""
        INSERT INTO scan_logs (scan_date, total_scanned, breakouts_found, squeezes_found)
        VALUES (%s, %s, %s, %s);
        """, (date_str, total_scanned, len(breakouts), len(squeezes)))
        
        conn.commit()
        cur.close()
        print(f"Cached {len(breakouts)} breakouts, {len(squeezes)} squeezes, {len(gapups)} gapups, {len(trend_setups)} trend setups, {len(wt_cross)} WT Cross setups, {len(vcs_results)} VCS setups, and {len(vpa_results)} VPA setups in Neon for {date_str}.")
        return True
    except Exception as e:
        if conn:
            conn.close()
        print(f"Error saving daily scan results to PostgreSQL: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_monthly_momentum_results(date_str: str, results: list[dict]) -> bool:
    """
    Saves the Monthly Momentum scan results to PostgreSQL, supporting daily overwrites.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # 1. Clean existing records for this date
        cur.execute("DELETE FROM scanned_monthly_momentum WHERE scan_date = %s;", (date_str,))
        
        # 2. Insert new results
        insert_query = """
        INSERT INTO scanned_monthly_momentum (
            symbol, company_name, cmp, day_change_pct, ema8, ema12, ema20, roc6, rsi_monthly, 
            volume, vol_sma12, market_cap_cr, momentum_score, buy_price, exit_price, target_price, 
            confidence, recommendation, return_1m, scan_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in results:
            cur.execute(insert_query, (
                str(r['symbol']),
                str(r['company_name']) if r['company_name'] else "",
                float(r['cmp']),
                float(r['day_change_pct']),
                float(r['ema8']),
                float(r['ema12']),
                float(r['ema20']),
                float(r['roc6']),
                float(r['rsi_monthly']),
                int(r['volume']),
                float(r['vol_sma12']),
                float(r['market_cap_cr']),
                float(r['momentum_score']),
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None,
                float(r['return_1m']) if r.get('return_1m') is not None else None,
                date_str
            ))
            
        conn.commit()
        cur.close()
        print(f"Cached {len(results)} Monthly Momentum scan results in PostgreSQL for {date_str}.")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error saving Monthly Momentum results to PostgreSQL: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_weekly_momentum_results(date_str: str, results: list[dict]) -> bool:
    """
    Saves the Weekly Momentum scan results to PostgreSQL, supporting daily overwrites.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # 1. Clean existing records for this date
        cur.execute("DELETE FROM scanned_weekly_momentum WHERE scan_date = %s;", (date_str,))
        
        # 2. Insert new results
        insert_query = """
        INSERT INTO scanned_weekly_momentum (
            symbol, company_name, cmp, weekly_chg_pct, prev_close, curr_open, close_sma20, 
            rsi_weekly, cci_weekly, volume, vol_sma20, vol_ratio, market_cap_cr, weekly_score, 
            buy_price, exit_price, target_price, confidence, recommendation, return_1m, scan_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in results:
            cur.execute(insert_query, (
                str(r['symbol']),
                str(r['company_name']) if r['company_name'] else "",
                float(r['cmp']),
                float(r['weekly_chg_pct']),
                float(r['prev_close']),
                float(r['curr_open']),
                float(r['close_sma20']),
                float(r['rsi_weekly']),
                float(r['cci_weekly']),
                int(r['volume']),
                float(r['vol_sma20']),
                float(r['vol_ratio']),
                float(r['market_cap_cr']),
                float(r['weekly_score']),
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') is not None else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None,
                float(r['return_1m']) if r.get('return_1m') is not None else None,
                date_str
            ))
            
        conn.commit()
        cur.close()
        print(f"Cached {len(results)} Weekly Momentum scan results in PostgreSQL for {date_str}.")
        return True
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error saving Weekly Momentum results to PostgreSQL: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_cached_monthly_momentum(date_str: str) -> list[dict]:
    """
    Retrieves the cached Monthly Momentum results for a specific date from PostgreSQL.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, ema8, ema12, ema20, roc6, rsi_monthly, 
           volume, vol_sma12, market_cap_cr, momentum_score, buy_price, exit_price, target_price, 
           confidence, recommendation, return_1m, scan_date
    FROM scanned_monthly_momentum
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
        print(f"Error loading cached Monthly Momentum from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_cached_weekly_momentum(date_str: str) -> list[dict]:
    """
    Retrieves the cached Weekly Momentum results for a specific date from PostgreSQL.
    """
    query = """
    SELECT symbol, company_name, cmp, weekly_chg_pct, prev_close, curr_open, close_sma20, 
           rsi_weekly, cci_weekly, volume, vol_sma20, vol_ratio, market_cap_cr, weekly_score, 
           buy_price, exit_price, target_price, confidence, recommendation, return_1m, scan_date
    FROM scanned_weekly_momentum
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
        print(f"Error loading cached Weekly Momentum from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_monthly_base_date(year: int, month: int) -> str | None:
    """
    Returns the earliest scan_date in the specified month and year that has cached monthly momentum results.
    """
    query = """
    SELECT MIN(scan_date) 
    FROM scanned_monthly_momentum 
    WHERE EXTRACT(YEAR FROM scan_date) = %s AND EXTRACT(MONTH FROM scan_date) = %s;
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query, (year, month))
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d") if hasattr(row[0], 'strftime') else str(row[0])
        return None
    except Exception as e:
        print(f"Error getting monthly base date from database: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_weekly_base_date(start_date_str: str, end_date_str: str) -> str | None:
    """
    Returns the earliest scan_date in the specified date range that has cached weekly momentum results.
    """
    query = """
    SELECT MIN(scan_date) 
    FROM scanned_weekly_momentum 
    WHERE scan_date >= %s AND scan_date <= %s;
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query, (start_date_str, end_date_str))
        row = cur.fetchone()
        cur.close()
        if row and row[0]:
            return row[0].strftime("%Y-%m-%d") if hasattr(row[0], 'strftime') else str(row[0])
        return None
    except Exception as e:
        print(f"Error getting weekly base date from database: {e}")
        return None
    finally:
        if conn:
            conn.close()
