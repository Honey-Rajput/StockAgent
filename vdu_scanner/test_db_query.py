import database

def main():
    conn = database.get_connection()
    cur = conn.cursor(cursor_factory=database.RealDictCursor)
    query = """
    WITH recent_dates AS (
        SELECT DISTINCT scan_date FROM scan_logs ORDER BY scan_date DESC LIMIT 15
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
    LIMIT 10;
    """
    try:
        cur.execute(query)
        rows = cur.fetchall()
        for r in rows:
            print(r)
    except Exception as e:
        print(e)
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()
