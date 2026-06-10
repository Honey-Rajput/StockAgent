import re

with open('database.py', 'r', encoding='utf-8') as f:
    code = f.read()

new_func = """
def get_frequent_stocks(days_lookback: int = 15) -> list[dict]:
    \"\"\"
    Retrieves stocks that have been frequently flagged across any scanner
    over the last N distinct scan dates.
    \"\"\"
    query = \"\"\"
    WITH recent_dates AS (
        SELECT DISTINCT scan_date FROM scan_logs ORDER BY scan_date DESC LIMIT %s
    ),
    all_scans AS (
        SELECT symbol, scan_date, 'VDU Breakout' as source FROM scanned_breakouts WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'Coiled Squeeze' as source FROM scanned_squeezes WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'Gap Up' as source FROM scanned_gapups WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, setup_type as source FROM scanned_trend_setups WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'WT Cross' as source FROM scanned_wt_cross WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'VCS' as source FROM scanned_vcs WHERE scan_date IN (SELECT scan_date FROM recent_dates)
        UNION ALL
        SELECT symbol, scan_date, 'VPA' as source FROM scanned_vpa WHERE scan_date IN (SELECT scan_date FROM recent_dates)
    )
    SELECT symbol, COUNT(*) as total_appearances, 
           COUNT(DISTINCT scan_date) as days_appeared,
           MIN(scan_date) as first_seen_date, 
           MAX(scan_date) as last_seen_date,
           STRING_AGG(DISTINCT source, ', ') as strategies
    FROM all_scans
    GROUP BY symbol
    HAVING COUNT(DISTINCT scan_date) > 1
    ORDER BY days_appeared DESC, total_appearances DESC
    LIMIT 200;
    \"\"\"
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
            results.append(r_dict)
    except Exception as e:
        print(f"Error loading frequent stocks from database: {e}")
    finally:
        if conn:
            conn.close()
    return results
"""

if 'def get_frequent_stocks' not in code:
    code += '\n' + new_func + '\n'
    with open('database.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print('Added get_frequent_stocks to database.py')
else:
    print('Function already exists')
