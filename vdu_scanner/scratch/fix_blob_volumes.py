"""
Fix corrupt blob volumes in ohlcv_daily table.
The volume column was stored as 8-byte little-endian integers (blob) instead of numeric.
This converts all blobs to proper integers.
"""
import sqlite3
import struct

conn = sqlite3.connect('scanner_data.db')
cur = conn.cursor()

print("Reading all rows with blob volumes...")
cur.execute("SELECT rowid, volume FROM ohlcv_daily WHERE typeof(volume) = 'blob'")
rows = cur.fetchall()
print(f"Found {len(rows)} rows to fix")

# Convert each blob to integer
fixed = []
for rowid, vol_bytes in rows:
    try:
        # Decode as 8-byte little-endian int64
        vol_int = struct.unpack('<q', vol_bytes)[0]
        fixed.append((vol_int, rowid))
    except Exception as e:
        print(f"Error converting rowid={rowid}: {e}")
        fixed.append((0, rowid))

print(f"Fixing {len(fixed)} rows...")
cur.executemany("UPDATE ohlcv_daily SET volume = ? WHERE rowid = ?", fixed)
conn.commit()
print("Done! Verifying...")

cur.execute("SELECT COUNT(*) FROM ohlcv_daily WHERE typeof(volume) = 'blob'")
remaining = cur.fetchone()[0]
print(f"Remaining blob rows: {remaining}")

# Quick sanity check
cur.execute("SELECT symbol, date, volume FROM ohlcv_daily ORDER BY date DESC LIMIT 5")
print("\nSample rows after fix:")
for row in cur.fetchall():
    print(row)

conn.close()
