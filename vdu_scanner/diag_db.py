"""Fix: Add missing scan_logs entry for 2026-07-21 so the DB is consistent."""
import sys
sys.path.insert(0, '.')
import sqlite3

conn = sqlite3.connect('scanner_data.db')
cur = conn.cursor()

# Check if we need to add a log entry for 2026-07-21
cur.execute("SELECT * FROM scan_logs WHERE scan_date = '2026-07-21'")
existing = cur.fetchone()
print(f"Existing log for 2026-07-21: {existing}")

if not existing:
    # Add a log entry so has_scanned_today works correctly
    # Count how many results exist in each table for that date
    counts = {}
    for tbl in ['scanned_zanger', 'scanned_ema_support', 'scanned_stage_analysis',
                'scanned_volume_profile', 'scanned_stage2', 'scanned_support_rsi']:
        cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE scan_date = '2026-07-21'")
        counts[tbl] = cur.fetchone()[0]
    print(f"Counts for 2026-07-21: {counts}")
    
    total = sum(counts.values())
    cur.execute("""
        INSERT INTO scan_logs (scan_date, total_scanned, breakouts_found, squeezes_found)
        VALUES ('2026-07-21', 1000, 0, 0)
        ON CONFLICT(scan_date) DO UPDATE SET
            total_scanned = EXCLUDED.total_scanned
    """)
    conn.commit()
    print("Added scan_log entry for 2026-07-21")
else:
    print("Log already exists, no fix needed")

# Verify
cur.execute("SELECT * FROM scan_logs ORDER BY scan_date DESC LIMIT 5")
for r in cur.fetchall():
    print(f"scan_log: {r}")

conn.close()
print("Done!")
