import database

conn = database.get_connection()
cur = conn.cursor()

# Delete stale scan_log for 2026-07-17 (had TEST dummy data with breakouts_found=1)
cur.execute("DELETE FROM scan_logs WHERE scan_date = '2026-07-17'")
print(f"Deleted {cur.rowcount} scan_log rows for 2026-07-17")

# Also clean ALL remaining test/dummy data from all breakout tables
tables = [
    "scanned_breakouts", "scanned_gapups", "scanned_trend_setups",
    "scanned_vcs", "scanned_vpa", "scanned_vpa_squeeze",
    "scanned_vcp_minervini", "scanned_zanger", "scanned_stage2",
    "scanned_volume_profile", "scanned_ema_support", "scanned_support_rsi",
    "scanned_stage_analysis", "scanned_monthly_momentum",
    "scanned_weekly_momentum", "scanned_near_30sma",
]
for table in tables:
    try:
        cur.execute(f"DELETE FROM {table} WHERE scan_date = '2026-07-17'")
        if cur.rowcount > 0:
            print(f"  Cleared {cur.rowcount} rows from {table} for 2026-07-17")
    except Exception as e:
        print(f"  {table}: {e}")

conn.commit()
print("Done! Database is now clean. Please run the scanner fresh to get real data.")
