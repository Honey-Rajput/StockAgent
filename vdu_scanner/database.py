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
            dist_20sma_pct DOUBLE PRECISION,
            dist_50sma_pct DOUBLE PRECISION,
            dist_65sma_pct DOUBLE PRECISION,
            dist_200sma_pct DOUBLE PRECISION,
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
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_stage2 (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            buy_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            target_price DOUBLE PRECISION,
            confidence VARCHAR(50),
            score DOUBLE PRECISION,
            recommendation TEXT,
            historical_high DOUBLE PRECISION,
            base_bottom DOUBLE PRECISION,
            sma7 DOUBLE PRECISION,
            extension DOUBLE PRECISION,
            rsi DOUBLE PRECISION,
            cci DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_volume_profile (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            market_cap_cr DOUBLE PRECISION,
            daily_zone VARCHAR(50),
            daily_pos DOUBLE PRECISION,
            weekly_zone VARCHAR(50),
            weekly_pos DOUBLE PRECISION,
            monthly_zone VARCHAR(50),
            monthly_pos DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_support_rsi (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            rsi DOUBLE PRECISION,
            cci DOUBLE PRECISION,
            support_price DOUBLE PRECISION,
            support_touches INT,
            distance_to_support_pct DOUBLE PRECISION,
            above_20sma BOOLEAN DEFAULT FALSE,
            above_50sma BOOLEAN DEFAULT FALSE,
            above_200sma BOOLEAN DEFAULT FALSE,
            volume BIGINT,
            score DOUBLE PRECISION,
            buy_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            target_price DOUBLE PRECISION,
            confidence VARCHAR(50),
            recommendation TEXT,
            market_cap_cr DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS scanned_rsi_wt_combo (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            rsi DOUBLE PRECISION,
            wt_value DOUBLE PRECISION,
            wt2_value DOUBLE PRECISION,
            wt_diff DOUBLE PRECISION,
            buy_signal BOOLEAN DEFAULT FALSE,
            support_price DOUBLE PRECISION,
            support_touches INT,
            distance_to_support_pct DOUBLE PRECISION,
            above_20sma BOOLEAN DEFAULT FALSE,
            above_50sma BOOLEAN DEFAULT FALSE,
            above_200sma BOOLEAN DEFAULT FALSE,
            volume BIGINT,
            score DOUBLE PRECISION,
            confidence VARCHAR(50),
            buy_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            target_price DOUBLE PRECISION,
            recommendation TEXT,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """
    ]
    
    # Inject BB Squeeze Table
    queries.append(
        """
        CREATE TABLE IF NOT EXISTS scanned_bb_squeeze (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            daily_squeeze BOOLEAN,
            weekly_squeeze BOOLEAN,
            monthly_squeeze BOOLEAN,
            daily_bb_width DOUBLE PRECISION,
            weekly_bb_width DOUBLE PRECISION,
            monthly_bb_width DOUBLE PRECISION,
            above_50dma BOOLEAN,
            market_cap_cr DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        """
    )
    
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
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS dist_20sma_pct DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS dist_50sma_pct DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS dist_65sma_pct DOUBLE PRECISION;",
            "ALTER TABLE scanned_trend_setups ADD COLUMN IF NOT EXISTS dist_200sma_pct DOUBLE PRECISION;",
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
            "ALTER TABLE scanned_vcs ADD COLUMN IF NOT EXISTS recommendation TEXT;",
            
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS daily_rsi DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS daily_cci DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS weekly_rsi DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS weekly_cci DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS monthly_rsi DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS monthly_cci DOUBLE PRECISION;",
            
            # VPA raw RWI value columns for action signal logic
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS daily_major_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS daily_mid_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS daily_minor_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS weekly_major_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS weekly_mid_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS weekly_minor_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS monthly_major_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS monthly_mid_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_vpa ADD COLUMN IF NOT EXISTS monthly_minor_val DOUBLE PRECISION;",
            
            # Volume Profile level columns (POC, VAL, VAH per timeframe)
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS daily_poc DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS daily_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS daily_vah DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS weekly_poc DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS weekly_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS weekly_vah DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS monthly_poc DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS monthly_val DOUBLE PRECISION;",
            "ALTER TABLE scanned_volume_profile ADD COLUMN IF NOT EXISTS monthly_vah DOUBLE PRECISION;",

            # BB Squeeze: score, confidence, score_breakdown columns
            "ALTER TABLE scanned_bb_squeeze ADD COLUMN IF NOT EXISTS squeeze_score INTEGER;",
            "ALTER TABLE scanned_bb_squeeze ADD COLUMN IF NOT EXISTS confidence VARCHAR(50);",
            "ALTER TABLE scanned_bb_squeeze ADD COLUMN IF NOT EXISTS score_breakdown TEXT;"
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

def get_all_patterns_by_date(date_str: str) -> dict:
    """
    Retrieves all cached technical pattern analyses for a specific date in one query.
    Returns a dictionary mapping symbol to pattern dict to prevent N+1 queries.
    """
    query = """
    SELECT symbol, pattern_name, confidence, direction, analysis_text, price_data_snapshot, analyzed_date
    FROM ai_chart_patterns
    WHERE analyzed_date = %s;
    """
    conn = None
    results = {}
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str,))
        rows = cur.fetchall()
        cur.close()
        for row in rows:
            row_dict = dict(row)
            # Standardize date output as string
            row_dict['analyzed_date'] = row_dict['analyzed_date'].strftime("%Y-%m-%d")
            results[row_dict['symbol']] = row_dict
    except Exception as e:
        print(f"Error loading cached patterns from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

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
           run_up_200, run_up_52w, is_early,
           dist_20sma_pct, dist_50sma_pct, dist_65sma_pct, dist_200sma_pct
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
           daily_major, daily_mid, daily_minor, daily_rsi, daily_cci,
           daily_major_val, daily_mid_val, daily_minor_val,
           weekly_major, weekly_mid, weekly_minor, weekly_rsi, weekly_cci,
           weekly_major_val, weekly_mid_val, weekly_minor_val,
           monthly_major, monthly_mid, monthly_minor, monthly_rsi, monthly_cci,
           monthly_major_val, monthly_mid_val, monthly_minor_val,
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
            
            r_dict['daily'] = {
                "major": int(r_dict.get('daily_major') or 0),
                "mid": int(r_dict.get('daily_mid') or 0),
                "minor": int(r_dict.get('daily_minor') or 0),
                "rsi": float(r_dict.get('daily_rsi') or 0.0),
                "cci": float(r_dict.get('daily_cci') or 0.0),
                "major_val": float(r_dict.get('daily_major_val') or 0.0),
                "mid_val": float(r_dict.get('daily_mid_val') or 0.0),
                "minor_val": float(r_dict.get('daily_minor_val') or 0.0)
            }
            r_dict['weekly'] = {
                "major": int(r_dict.get('weekly_major') or 0),
                "mid": int(r_dict.get('weekly_mid') or 0),
                "minor": int(r_dict.get('weekly_minor') or 0),
                "rsi": float(r_dict.get('weekly_rsi') or 0.0),
                "cci": float(r_dict.get('weekly_cci') or 0.0),
                "major_val": float(r_dict.get('weekly_major_val') or 0.0),
                "mid_val": float(r_dict.get('weekly_mid_val') or 0.0),
                "minor_val": float(r_dict.get('weekly_minor_val') or 0.0)
            }
            r_dict['monthly'] = {
                "major": int(r_dict.get('monthly_major') or 0),
                "mid": int(r_dict.get('monthly_mid') or 0),
                "minor": int(r_dict.get('monthly_minor') or 0),
                "rsi": float(r_dict.get('monthly_rsi') or 0.0),
                "cci": float(r_dict.get('monthly_cci') or 0.0),
                "major_val": float(r_dict.get('monthly_major_val') or 0.0),
                "mid_val": float(r_dict.get('monthly_mid_val') or 0.0),
                "minor_val": float(r_dict.get('monthly_minor_val') or 0.0)
            }
            
            # Recompute normalized score (0-100) from trend values
            # Same formula as scan_vpa_trend: raw_score = sum(major*3 + mid*2 + minor*1) per timeframe
            raw_score = 0
            for tf_data in [r_dict['daily'], r_dict['weekly'], r_dict['monthly']]:
                raw_score += tf_data['major'] * 3
                raw_score += tf_data['mid'] * 2
                raw_score += tf_data['minor'] * 1
            normalized_score = round((raw_score + 18) / 36 * 100)
            r_dict['score'] = normalized_score
            
            # Derive confidence from normalized score
            if normalized_score >= 80:
                r_dict['confidence'] = "High"
            elif normalized_score >= 50:
                r_dict['confidence'] = "Medium"
            else:
                r_dict['confidence'] = "Low"
            
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached VPA from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def save_vcs_only(date_str: str, vcs_results: list[dict]) -> bool:
    """
    UPSERTs only the VCS results for the given date.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM scanned_vcs WHERE scan_date = %s;", (date_str,))
        
        insert_vcs_query = """
        INSERT INTO scanned_vcs (symbol, company_name, cmp, day_change_pct, vcs_score, volume, scan_date,
                                 buy_price, exit_price, target_price, confidence, recommendation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in vcs_results:
            cur.execute(insert_vcs_query, (
                str(r['symbol']),
                str(r.get('company_name', "")),
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
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving VCS only: {e}")
        return False
    finally:
        if conn:
            conn.close()

def save_vpa_only(date_str: str, vpa_results: list[dict]) -> bool:
    """
    UPSERTs only the VPA results for the given date.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM scanned_vpa WHERE scan_date = %s;", (date_str,))
        
        insert_vpa_query = """
        INSERT INTO scanned_vpa (symbol, company_name, cmp, day_change_pct, volume, vpa_score, 
                                 daily_major, daily_mid, daily_minor, daily_rsi, daily_cci,
                                 daily_major_val, daily_mid_val, daily_minor_val,
                                 weekly_major, weekly_mid, weekly_minor, weekly_rsi, weekly_cci,
                                 weekly_major_val, weekly_mid_val, weekly_minor_val,
                                 monthly_major, monthly_mid, monthly_minor, monthly_rsi, monthly_cci,
                                 monthly_major_val, monthly_mid_val, monthly_minor_val,
                                 scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in vpa_results:
            cur.execute(insert_vpa_query, (
                str(r['symbol']),
                str(r.get('company_name', "")),
                float(r['cmp']),
                float(r['day_change_pct']),
                int(r.get('volume', 0)),
                int(r.get('score', 0)),
                int(r.get('daily', {}).get('major', 0)),
                int(r.get('daily', {}).get('mid', 0)),
                int(r.get('daily', {}).get('minor', 0)),
                float(r.get('daily', {}).get('rsi', 0.0)),
                float(r.get('daily', {}).get('cci', 0.0)),
                float(r.get('daily', {}).get('major_val', 0.0)),
                float(r.get('daily', {}).get('mid_val', 0.0)),
                float(r.get('daily', {}).get('minor_val', 0.0)),
                int(r.get('weekly', {}).get('major', 0)),
                int(r.get('weekly', {}).get('mid', 0)),
                int(r.get('weekly', {}).get('minor', 0)),
                float(r.get('weekly', {}).get('rsi', 0.0)),
                float(r.get('weekly', {}).get('cci', 0.0)),
                float(r.get('weekly', {}).get('major_val', 0.0)),
                float(r.get('weekly', {}).get('mid_val', 0.0)),
                float(r.get('weekly', {}).get('minor_val', 0.0)),
                int(r.get('monthly', {}).get('major', 0)),
                int(r.get('monthly', {}).get('mid', 0)),
                int(r.get('monthly', {}).get('minor', 0)),
                float(r.get('monthly', {}).get('rsi', 0.0)),
                float(r.get('monthly', {}).get('cci', 0.0)),
                float(r.get('monthly', {}).get('major_val', 0.0)),
                float(r.get('monthly', {}).get('mid_val', 0.0)),
                float(r.get('monthly', {}).get('minor_val', 0.0)),
                date_str
            ))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving VPA only: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_cached_stage2(date_str: str) -> list[dict]:
    """
    Retrieves the cached Early Stage 2 setups scanned on a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, buy_price, exit_price, target_price, confidence, score, recommendation,
           historical_high, base_bottom, sma7, extension, rsi, cci, scan_date
    FROM scanned_stage2
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
            # Convert decimal back to float if needed
            for k in ['cmp', 'buy_price', 'exit_price', 'target_price', 'score', 'historical_high', 'base_bottom', 'sma7', 'extension', 'rsi', 'cci']:
                if r_dict.get(k) is not None:
                    r_dict[k] = float(r_dict[k])
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached stage2 from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def save_volume_profile_only(date_str: str, vp_results: list[dict]) -> bool:
    """Saves Volume Profile results including POC/VAL/VAH levels."""
    if not vp_results:
        return True
        
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        insert_vp_query = """
        INSERT INTO scanned_volume_profile 
        (symbol, company_name, cmp, market_cap_cr, 
         daily_zone, daily_pos, daily_poc, daily_val, daily_vah,
         weekly_zone, weekly_pos, weekly_poc, weekly_val, weekly_vah,
         monthly_zone, monthly_pos, monthly_poc, monthly_val, monthly_vah,
         scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, scan_date) DO UPDATE SET
            cmp = EXCLUDED.cmp,
            market_cap_cr = EXCLUDED.market_cap_cr,
            daily_zone = EXCLUDED.daily_zone, daily_pos = EXCLUDED.daily_pos,
            daily_poc = EXCLUDED.daily_poc, daily_val = EXCLUDED.daily_val, daily_vah = EXCLUDED.daily_vah,
            weekly_zone = EXCLUDED.weekly_zone, weekly_pos = EXCLUDED.weekly_pos,
            weekly_poc = EXCLUDED.weekly_poc, weekly_val = EXCLUDED.weekly_val, weekly_vah = EXCLUDED.weekly_vah,
            monthly_zone = EXCLUDED.monthly_zone, monthly_pos = EXCLUDED.monthly_pos,
            monthly_poc = EXCLUDED.monthly_poc, monthly_val = EXCLUDED.monthly_val, monthly_vah = EXCLUDED.monthly_vah;
        """
        
        def _extract_tf(v, tf_key):
            """Extract zone, pos, poc, val, vah from a timeframe dict."""
            tf = v.get(tf_key)
            if isinstance(tf, dict) and tf:
                return (
                    tf.get('zone', ''),
                    tf.get('position_pct', None),
                    tf.get('poc', None),
                    tf.get('val', None),
                    tf.get('vah', None)
                )
            return ('', None, None, None, None)
        
        vp_data = []
        for v in vp_results:
            sym = v.get('symbol', v.get('Symbol', ''))
            cmp = v.get('cmp', v.get('CMP', 0))
            mcap = v.get('market_cap_cr', v.get('Market Cap (Cr)', 0))
            clean_sym = str(sym).replace('.NS', '').strip().upper()
            
            d_zone, d_pos, d_poc, d_val, d_vah = _extract_tf(v, 'daily')
            w_zone, w_pos, w_poc, w_val, w_vah = _extract_tf(v, 'weekly')
            m_zone, m_pos, m_poc, m_val, m_vah = _extract_tf(v, 'monthly')
            
            vp_data.append((
                clean_sym, '',
                float(cmp) if cmp else 0,
                float(mcap) if mcap else 0,
                d_zone, float(d_pos) if d_pos is not None else None,
                float(d_poc) if d_poc is not None else None,
                float(d_val) if d_val is not None else None,
                float(d_vah) if d_vah is not None else None,
                w_zone, float(w_pos) if w_pos is not None else None,
                float(w_poc) if w_poc is not None else None,
                float(w_val) if w_val is not None else None,
                float(w_vah) if w_vah is not None else None,
                m_zone, float(m_pos) if m_pos is not None else None,
                float(m_poc) if m_poc is not None else None,
                float(m_val) if m_val is not None else None,
                float(m_vah) if m_vah is not None else None,
                date_str
            ))
            
        from psycopg2.extras import execute_batch
        execute_batch(cur, insert_vp_query, vp_data)
        conn.commit()
        return True
    except Exception as e:
        print(f"Database error in save_volume_profile_only: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def get_cached_volume_profile(date_str: str) -> list[dict]:
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        cur.execute("""
            SELECT * FROM scanned_volume_profile 
            WHERE scan_date = %s
            ORDER BY market_cap_cr DESC
        """, (date_str,))
        
        rows = cur.fetchall()
        results = []
        for row in rows:
            def _safe_float(val):
                return float(val) if val is not None else None
            
            results.append({
                'symbol': row['symbol'],
                'company_name': row['company_name'],
                'cmp': row['cmp'],
                'market_cap_cr': row['market_cap_cr'],
                'daily': {
                    'zone': row['daily_zone'] or '',
                    'position_pct': row['daily_pos'] if row['daily_pos'] is not None else "",
                    'poc': _safe_float(row.get('daily_poc')),
                    'val': _safe_float(row.get('daily_val')),
                    'vah': _safe_float(row.get('daily_vah'))
                },
                'weekly': {
                    'zone': row['weekly_zone'] or '',
                    'position_pct': row['weekly_pos'] if row['weekly_pos'] is not None else "",
                    'poc': _safe_float(row.get('weekly_poc')),
                    'val': _safe_float(row.get('weekly_val')),
                    'vah': _safe_float(row.get('weekly_vah'))
                },
                'monthly': {
                    'zone': row['monthly_zone'] or '',
                    'position_pct': row['monthly_pos'] if row['monthly_pos'] is not None else "",
                    'poc': _safe_float(row.get('monthly_poc')),
                    'val': _safe_float(row.get('monthly_val')),
                    'vah': _safe_float(row.get('monthly_vah'))
                }
            })
        return results
    except Exception as e:
        print(f"Database error reading cached Volume Profile: {e}")
        return []
    finally:
        if conn:
            cur.close()
            conn.close()

def save_stage2_only(date_str: str, stage2_results: list[dict]) -> bool:
    """
    UPSERTs only the Stage 2 results for the given date.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM scanned_stage2 WHERE scan_date = %s;", (date_str,))
        
        insert_query = """
        INSERT INTO scanned_stage2 (symbol, company_name, cmp, buy_price, exit_price, target_price, confidence, 
                                    score, recommendation, historical_high, base_bottom, sma7, extension, rsi, cci, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in stage2_results:
            cur.execute(insert_query, (
                str(r['symbol']),
                str(r.get('company_name', "")),
                float(r['cmp']),
                float(r['buy_price']),
                float(r['exit_price']),
                float(r['target_price']),
                str(r['confidence']),
                float(r['score']),
                str(r['recommendation']),
                float(r['historical_high']),
                float(r['base_bottom']),
                float(r['sma7']),
                float(r['extension']),
                float(r.get('rsi', 0.0)),
                float(r.get('cci', 0.0)),
                date_str
            ))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving stage2 only: {e}")
        return False
    finally:
        if conn:
            conn.close()

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
                                         run_up_200, run_up_52w, is_early,
                                         dist_20sma_pct, dist_50sma_pct, dist_65sma_pct, dist_200sma_pct)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
                str(r['confidence']) if r.get('confidence') else None,
                str(r['recommendation']) if r.get('recommendation') is not None else None,
                float(r['run_up_200']) if r.get('run_up_200') is not None else None,
                float(r['run_up_52w']) if r.get('run_up_52w') is not None else None,
                bool(r['is_early']) if r.get('is_early') is not None else None,
                float(r['dist_20sma_pct']) if r.get('dist_20sma_pct') is not None else None,
                float(r['dist_50sma_pct']) if r.get('dist_50sma_pct') is not None else None,
                float(r['dist_65sma_pct']) if r.get('dist_65sma_pct') is not None else None,
                float(r['dist_200sma_pct']) if r.get('dist_200sma_pct') is not None else None
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
                                 daily_major, daily_mid, daily_minor, daily_rsi, daily_cci,
                                 daily_major_val, daily_mid_val, daily_minor_val,
                                 weekly_major, weekly_mid, weekly_minor, weekly_rsi, weekly_cci,
                                 weekly_major_val, weekly_mid_val, weekly_minor_val,
                                 monthly_major, monthly_mid, monthly_minor, monthly_rsi, monthly_cci,
                                 monthly_major_val, monthly_mid_val, monthly_minor_val,
                                 scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
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
                float(r.get('daily', {}).get('rsi', 0.0)),
                float(r.get('daily', {}).get('cci', 0.0)),
                float(r.get('daily', {}).get('major_val', 0.0)),
                float(r.get('daily', {}).get('mid_val', 0.0)),
                float(r.get('daily', {}).get('minor_val', 0.0)),
                int(r.get('weekly', {}).get('major', 0)),
                int(r.get('weekly', {}).get('mid', 0)),
                int(r.get('weekly', {}).get('minor', 0)),
                float(r.get('weekly', {}).get('rsi', 0.0)),
                float(r.get('weekly', {}).get('cci', 0.0)),
                float(r.get('weekly', {}).get('major_val', 0.0)),
                float(r.get('weekly', {}).get('mid_val', 0.0)),
                float(r.get('weekly', {}).get('minor_val', 0.0)),
                int(r.get('monthly', {}).get('major', 0)),
                int(r.get('monthly', {}).get('mid', 0)),
                int(r.get('monthly', {}).get('minor', 0)),
                float(r.get('monthly', {}).get('rsi', 0.0)),
                float(r.get('monthly', {}).get('cci', 0.0)),
                float(r.get('monthly', {}).get('major_val', 0.0)),
                float(r.get('monthly', {}).get('mid_val', 0.0)),
                float(r.get('monthly', {}).get('minor_val', 0.0)),
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


def get_frequent_stocks(days_lookback: int = 15) -> list[dict]:
    """
    Retrieves stocks that have been frequently flagged across any scanner
    over the last N distinct scan dates.
    """
    query = """
    WITH all_dates AS (
        SELECT scan_date FROM scan_logs
        UNION SELECT scan_date FROM scanned_breakouts
        UNION SELECT scan_date FROM scanned_trend_setups
        UNION SELECT scan_date FROM scanned_monthly_momentum
        UNION SELECT scan_date FROM scanned_vcs
        UNION SELECT scan_date FROM scanned_stage2
        UNION SELECT scan_date FROM scanned_gapups
        UNION SELECT scan_date FROM scanned_wt_cross
        UNION SELECT scan_date FROM scanned_vpa
        UNION SELECT scan_date FROM scanned_volume_profile
        UNION SELECT scan_date FROM scanned_support_rsi
        UNION SELECT scan_date FROM scanned_weekly_momentum
    ),
    recent_dates AS (
        SELECT DISTINCT scan_date FROM all_dates ORDER BY scan_date DESC LIMIT %s
    ),
    all_scans AS (
        SELECT symbol, scan_date, 'VDU Breakout' as source, signal_strength as score FROM scanned_breakouts WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'Minervini Stage-2' as source, 0 AS score FROM scanned_trend_setups WHERE setup_type = 'minervini' AND scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'VCS' as source, vcs_score as score FROM scanned_vcs WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'Monthly Momentum' as source, momentum_score as score FROM scanned_monthly_momentum WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'Stage-2 Breakout' as source, score FROM scanned_stage2 WHERE scan_date IN (SELECT scan_date FROM recent_dates)
    ),
    aggregated AS (
        SELECT symbol, 
               COUNT(*) as total_appearances, 
               COUNT(DISTINCT scan_date) as days_appeared,
               MIN(scan_date) as first_seen_date, 
               MAX(scan_date) as last_seen_date,
               STRING_AGG(DISTINCT source, ', ') as strategies,
               MAX(score) as max_score
        FROM all_scans
        GROUP BY symbol
        HAVING COUNT(*) > 1
    )
    SELECT a.*, v.daily_rsi as rsi, v.daily_cci as cci
    FROM aggregated a
    LEFT JOIN LATERAL (
        SELECT daily_rsi, daily_cci FROM scanned_vpa
        WHERE symbol = a.symbol
        ORDER BY scan_date DESC LIMIT 1
    ) v ON TRUE
    ORDER BY a.days_appeared DESC, a.total_appearances DESC
    LIMIT 200;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        from psycopg2.extras import RealDictCursor
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (days_lookback,))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['first_seen_date'] = r_dict['first_seen_date'].strftime('%Y-%m-%d')
            r_dict['last_seen_date'] = r_dict['last_seen_date'].strftime('%Y-%m-%d')
            r_dict['rsi'] = float(r_dict.get('rsi') or 0.0)
            r_dict['cci'] = float(r_dict.get('cci') or 0.0)
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading frequent stocks from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def get_wt_vp_confluence(date_str: str) -> list[dict]:
    """
    Returns stocks that have BOTH:
      1. A WaveTrend buy signal (buy_signal = TRUE) on the given date
      2. A Volume Profile daily zone of 'Can Buy (Near Support)' on the same date
    Fetches from database by JOINing scanned_wt_cross and scanned_volume_profile.
    """
    query = """
    SELECT
        wt.symbol,
        wt.company_name,
        wt.cmp,
        wt.day_change_pct,
        wt.wt_value,
        wt.wt2_value,
        wt.wt_diff,
        wt.above_20sma,
        wt.above_50sma,
        wt.above_200sma,
        wt.volume,
        wt.buy_price,
        wt.exit_price,
        wt.target_price,
        wt.confidence,
        wt.recommendation,
        vp.daily_zone,
        vp.daily_pos,
        vp.daily_poc,
        vp.daily_val,
        vp.daily_vah
    FROM scanned_wt_cross wt
    INNER JOIN scanned_volume_profile vp
        ON wt.symbol = vp.symbol AND wt.scan_date = vp.scan_date
    WHERE wt.scan_date = %s
      AND wt.buy_signal = TRUE
      AND vp.daily_zone LIKE %s
    ORDER BY wt.wt_value ASC;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str, '%Can Buy%'))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['wt_value'] = float(r_dict.get('wt_value') or 0.0)
            r_dict['wt2_value'] = float(r_dict.get('wt2_value') or 0.0)
            r_dict['wt_diff'] = float(r_dict.get('wt_diff') or 0.0)
            r_dict['above_20sma'] = bool(r_dict.get('above_20sma', False))
            r_dict['above_50sma'] = bool(r_dict.get('above_50sma', False))
            r_dict['above_200sma'] = bool(r_dict.get('above_200sma', False))
            r_dict['volume'] = int(r_dict.get('volume') or 0)
            r_dict['daily_pos'] = float(r_dict.get('daily_pos') or 0.0)
            r_dict['daily_poc'] = float(r_dict.get('daily_poc') or 0.0)
            r_dict['daily_val'] = float(r_dict.get('daily_val') or 0.0)
            r_dict['daily_vah'] = float(r_dict.get('daily_vah') or 0.0)
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading WT+VP confluence from database: {e}")
    finally:
        if conn:
            conn.close()
    return results

def save_support_rsi_only(date_str: str, results: list[dict]) -> bool:
    """
    Saves the Support + RSI Oversold scan results for the given date.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM scanned_support_rsi WHERE scan_date = %s;", (date_str,))
        
        insert_query = """
        INSERT INTO scanned_support_rsi (symbol, company_name, cmp, day_change_pct, rsi, cci,
                                         support_price, support_touches, distance_to_support_pct,
                                         above_20sma, above_50sma, above_200sma, volume, score,
                                         buy_price, exit_price, target_price, confidence, recommendation,
                                         market_cap_cr, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in results:
            cur.execute(insert_query, (
                str(r['symbol']),
                str(r.get('company_name', '')),
                float(r['cmp']),
                float(r['day_change_pct']),
                float(r['rsi']),
                float(r.get('cci', 0.0)),
                float(r['support_price']),
                int(r['support_touches']),
                float(r['distance_to_support_pct']),
                bool(r.get('above_20sma', False)),
                bool(r.get('above_50sma', False)),
                bool(r.get('above_200sma', False)),
                int(r.get('volume', 0)),
                float(r.get('score', 0.0)),
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r['confidence']) if r.get('confidence') else None,
                str(r['recommendation']) if r.get('recommendation') else None,
                float(r.get('market_cap_cr', 0.0)),
                date_str
            ))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving support RSI results: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_cached_support_rsi(date_str: str) -> list[dict]:
    """
    Retrieves the cached Support + RSI Oversold results for a specific date.
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, rsi, cci,
           support_price, support_touches, distance_to_support_pct,
           above_20sma, above_50sma, above_200sma, volume, score,
           buy_price, exit_price, target_price, confidence, recommendation,
           market_cap_cr, scan_date
    FROM scanned_support_rsi
    WHERE scan_date = %s
    ORDER BY score DESC;
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
            r_dict['rsi'] = float(r_dict.get('rsi') or 0.0)
            r_dict['cci'] = float(r_dict.get('cci') or 0.0)
            r_dict['support_price'] = float(r_dict.get('support_price') or 0.0)
            r_dict['support_touches'] = int(r_dict.get('support_touches') or 0)
            r_dict['distance_to_support_pct'] = float(r_dict.get('distance_to_support_pct') or 0.0)
            r_dict['above_20sma'] = bool(r_dict.get('above_20sma', False))
            r_dict['above_50sma'] = bool(r_dict.get('above_50sma', False))
            r_dict['above_200sma'] = bool(r_dict.get('above_200sma', False))
            r_dict['volume'] = int(r_dict.get('volume') or 0)
            r_dict['score'] = float(r_dict.get('score') or 0.0)
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading cached support RSI from database: {e}")
    finally:
        if conn:
            conn.close()
    return results


def get_latest_support_rsi() -> tuple[list[dict], str | None]:
    """
    Retrieves the most recent Support + RSI Oversold scan results,
    regardless of date. Returns (results_list, scan_date_str) or ([], None).
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, rsi, cci,
           support_price, support_touches, distance_to_support_pct,
           above_20sma, above_50sma, above_200sma, volume, score,
           buy_price, exit_price, target_price, confidence, recommendation,
           market_cap_cr, scan_date
    FROM scanned_support_rsi
    WHERE scan_date = (SELECT MAX(scan_date) FROM scanned_support_rsi)
    ORDER BY score DESC;
    """
    conn = None
    results = []
    scan_date = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            scan_date = r_dict['scan_date'].strftime("%Y-%m-%d") if scan_date is None else scan_date
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            r_dict['rsi'] = float(r_dict.get('rsi') or 0.0)
            r_dict['cci'] = float(r_dict.get('cci') or 0.0)
            r_dict['support_price'] = float(r_dict.get('support_price') or 0.0)
            r_dict['support_touches'] = int(r_dict.get('support_touches') or 0)
            r_dict['distance_to_support_pct'] = float(r_dict.get('distance_to_support_pct') or 0.0)
            r_dict['above_20sma'] = bool(r_dict.get('above_20sma', False))
            r_dict['above_50sma'] = bool(r_dict.get('above_50sma', False))
            r_dict['above_200sma'] = bool(r_dict.get('above_200sma', False))
            r_dict['volume'] = int(r_dict.get('volume') or 0)
            r_dict['score'] = float(r_dict.get('score') or 0.0)
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading latest support RSI from database: {e}")
    finally:
        if conn:
            conn.close()
    return results, scan_date


def get_rsi_wt_combo(date_str: str, rsi_threshold: float = 35.0, wt_threshold: float = -40.0) -> list[dict]:
    """
    Finds stocks that appear in BOTH scanned_wt_cross (WT oversold + buy signal)
    and scanned_support_rsi (RSI oversold) on the same scan date.
    JOINs the two tables for fast database-only retrieval.
    Computes a combo score and confidence level.
    """
    query = """
    SELECT
        wt.symbol,
        wt.company_name,
        wt.cmp,
        wt.day_change_pct,
        sr.rsi,
        sr.cci,
        wt.wt_value,
        wt.wt2_value,
        wt.wt_diff,
        wt.buy_signal,
        sr.support_price,
        sr.support_touches,
        sr.distance_to_support_pct,
        wt.above_20sma,
        wt.above_50sma,
        sr.above_200sma,
        wt.volume,
        wt.buy_price,
        wt.exit_price,
        wt.target_price,
        wt.confidence AS wt_confidence,
        wt.recommendation,
        wt.scan_date
    FROM scanned_wt_cross wt
    INNER JOIN scanned_support_rsi sr
        ON wt.symbol = sr.symbol AND wt.scan_date = sr.scan_date
    WHERE wt.scan_date = %s
      AND wt.wt_value <= %s
      AND sr.rsi <= %s
    ORDER BY sr.rsi ASC, wt.wt_value ASC;
    """
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (date_str, wt_threshold, rsi_threshold))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            rsi_val = float(r_dict.get('rsi') or 0.0)
            wt_val = float(r_dict.get('wt_value') or 0.0)
            r_dict['rsi'] = rsi_val
            r_dict['cci'] = float(r_dict.get('cci') or 0.0)
            r_dict['wt_value'] = wt_val
            r_dict['wt2_value'] = float(r_dict.get('wt2_value') or 0.0)
            r_dict['wt_diff'] = float(r_dict.get('wt_diff') or 0.0)
            r_dict['buy_signal'] = bool(r_dict.get('buy_signal', False))
            r_dict['support_price'] = float(r_dict.get('support_price') or 0.0)
            r_dict['support_touches'] = int(r_dict.get('support_touches') or 0)
            r_dict['distance_to_support_pct'] = float(r_dict.get('distance_to_support_pct') or 0.0)
            r_dict['above_20sma'] = bool(r_dict.get('above_20sma', False))
            r_dict['above_50sma'] = bool(r_dict.get('above_50sma', False))
            r_dict['above_200sma'] = bool(r_dict.get('above_200sma', False))
            r_dict['volume'] = int(r_dict.get('volume') or 0)

            # Compute combo score
            rsi_bonus = max(0, (rsi_threshold - rsi_val)) * 2.0
            wt_bonus = max(0, abs(wt_val) - abs(wt_threshold)) * 1.5
            signal_bonus = 25.0 if r_dict['buy_signal'] else 0.0
            touch_bonus = r_dict['support_touches'] * 10.0
            combo_score = round(rsi_bonus + wt_bonus + signal_bonus + touch_bonus, 1)
            r_dict['score'] = combo_score

            # Compute confidence
            if rsi_val <= 30 and wt_val <= -50 and r_dict['buy_signal'] and r_dict['support_touches'] >= 3:
                r_dict['confidence'] = 'High'
            elif rsi_val <= rsi_threshold and wt_val <= wt_threshold and r_dict['buy_signal']:
                r_dict['confidence'] = 'Medium'
            else:
                r_dict['confidence'] = 'Low'

            results.append(r_dict)

        # Sort by combo score descending
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
    except Exception as e:
        print(f"Error loading RSI+WT combo from database: {e}")
    finally:
        if conn:
            conn.close()
    return results


def save_rsi_wt_combo(date_str: str, results: list[dict]) -> bool:
    """
    Saves the RSI + WaveTrend combo scan results for the given date.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM scanned_rsi_wt_combo WHERE scan_date = %s;", (date_str,))

        insert_query = """
        INSERT INTO scanned_rsi_wt_combo (symbol, company_name, cmp, day_change_pct, rsi,
                                          wt_value, wt2_value, wt_diff, buy_signal,
                                          support_price, support_touches, distance_to_support_pct,
                                          above_20sma, above_50sma, above_200sma, volume, score,
                                          confidence, buy_price, exit_price, target_price,
                                          recommendation, scan_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        for r in results:
            cur.execute(insert_query, (
                str(r['symbol']),
                str(r.get('company_name', '')),
                float(r.get('cmp', 0)),
                float(r.get('day_change_pct', 0)),
                float(r.get('rsi', 0)),
                float(r.get('wt_value', 0)),
                float(r.get('wt2_value', 0)),
                float(r.get('wt_diff', 0)),
                bool(r.get('buy_signal', False)),
                float(r.get('support_price', 0)) if r.get('support_price') is not None else None,
                int(r.get('support_touches', 0)),
                float(r.get('distance_to_support_pct', 0)),
                bool(r.get('above_20sma', False)),
                bool(r.get('above_50sma', False)),
                bool(r.get('above_200sma', False)),
                int(r.get('volume', 0)),
                float(r.get('score', 0)),
                str(r.get('confidence', 'Low')),
                float(r['buy_price']) if r.get('buy_price') is not None else None,
                float(r['exit_price']) if r.get('exit_price') is not None else None,
                float(r['target_price']) if r.get('target_price') is not None else None,
                str(r.get('recommendation', '')),
                date_str
            ))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving RSI+WT combo results: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_latest_rsi_wt_combo() -> tuple[list[dict], str | None]:
    """
    Retrieves the most recent RSI + WaveTrend combo results,
    regardless of date. Returns (results_list, scan_date_str) or ([], None).
    """
    query = """
    SELECT symbol, company_name, cmp, day_change_pct, rsi,
           wt_value, wt2_value, wt_diff, buy_signal,
           support_price, support_touches, distance_to_support_pct,
           above_20sma, above_50sma, above_200sma, volume, score,
           confidence, buy_price, exit_price, target_price,
           recommendation, scan_date
    FROM scanned_rsi_wt_combo
    WHERE scan_date = (SELECT MAX(scan_date) FROM scanned_rsi_wt_combo)
    ORDER BY score DESC;
    """
    conn = None
    results = []
    scan_date = None
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            scan_date = r_dict['scan_date'].strftime("%Y-%m-%d") if scan_date is None else scan_date
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            r_dict['rsi'] = float(r_dict.get('rsi') or 0.0)
            r_dict['wt_value'] = float(r_dict.get('wt_value') or 0.0)
            r_dict['wt2_value'] = float(r_dict.get('wt2_value') or 0.0)
            r_dict['wt_diff'] = float(r_dict.get('wt_diff') or 0.0)
            r_dict['buy_signal'] = bool(r_dict.get('buy_signal', False))
            r_dict['support_price'] = float(r_dict.get('support_price') or 0.0)
            r_dict['support_touches'] = int(r_dict.get('support_touches') or 0)
            r_dict['distance_to_support_pct'] = float(r_dict.get('distance_to_support_pct') or 0.0)
            r_dict['above_20sma'] = bool(r_dict.get('above_20sma', False))
            r_dict['above_50sma'] = bool(r_dict.get('above_50sma', False))
            r_dict['above_200sma'] = bool(r_dict.get('above_200sma', False))
            r_dict['volume'] = int(r_dict.get('volume') or 0)
            r_dict['score'] = float(r_dict.get('score') or 0.0)
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading latest RSI+WT combo from database: {e}")
    finally:
        if conn:
            conn.close()
    return results, scan_date

def save_bb_squeeze_only(date_str: str, bb_results: list) -> bool:
    if not bb_results:
        return True
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        for r in bb_results:
            cur.execute("""
                INSERT INTO scanned_bb_squeeze
                (symbol, company_name, cmp, day_change_pct,
                 daily_squeeze, weekly_squeeze, monthly_squeeze,
                 daily_bb_width, weekly_bb_width, monthly_bb_width,
                 above_50dma, market_cap_cr, scan_date,
                 squeeze_score, confidence, score_breakdown)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, scan_date) DO UPDATE SET
                    cmp = EXCLUDED.cmp,
                    day_change_pct = EXCLUDED.day_change_pct,
                    daily_squeeze = EXCLUDED.daily_squeeze,
                    weekly_squeeze = EXCLUDED.weekly_squeeze,
                    monthly_squeeze = EXCLUDED.monthly_squeeze,
                    daily_bb_width = EXCLUDED.daily_bb_width,
                    weekly_bb_width = EXCLUDED.weekly_bb_width,
                    monthly_bb_width = EXCLUDED.monthly_bb_width,
                    above_50dma = EXCLUDED.above_50dma,
                    squeeze_score = EXCLUDED.squeeze_score,
                    confidence = EXCLUDED.confidence,
                    score_breakdown = EXCLUDED.score_breakdown
            """, (
                r['symbol'], r.get('company_name', ''), float(r['cmp']), float(r['day_change_pct']),
                bool(r.get('daily_squeeze', False)), bool(r.get('weekly_squeeze', False)), bool(r.get('monthly_squeeze', False)),
                float(r.get('daily_bb_width', 0.0) or 0.0), float(r.get('weekly_bb_width', 0.0) or 0.0), float(r.get('monthly_bb_width', 0.0) or 0.0),
                bool(r.get('above_50dma', False)), float(r.get('market_cap_cr', 0.0)), date_str,
                int(r.get('squeeze_score', 0) or 0), r.get('confidence', ''), r.get('score_breakdown', '')
            ))
        conn.commit()
        cur.close()
        return True
    except Exception as e:
        print(f"Error saving BB Squeeze to DB: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_cached_bb_squeeze(date_str: str) -> list:
    conn = None
    results = []
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM scanned_bb_squeeze WHERE scan_date = %s", (date_str,))
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            r_dict = dict(r)
            r_dict['scan_date'] = r_dict['scan_date'].strftime("%Y-%m-%d")
            results.append(r_dict)
    except Exception as e:
        print(f"Error fetching cached BB Squeeze: {e}")
    finally:
        if conn:
            conn.close()
    return results
