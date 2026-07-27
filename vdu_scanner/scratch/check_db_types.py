import sqlite3
conn = sqlite3.connect('scanner_data.db')
cur = conn.cursor()

# Find symbols with blob/bytes data
cur.execute("SELECT symbol, date, typeof(volume), volume FROM ohlcv_daily WHERE typeof(volume) != 'real' AND typeof(volume) != 'integer' LIMIT 10")
rows = cur.fetchall()
print("Non-numeric volume rows:")
for row in rows:
    print(row)

# Count how many rows have bad data
cur.execute("SELECT COUNT(*) FROM ohlcv_daily WHERE typeof(volume) NOT IN ('real', 'integer', 'null')")
print(f"\nTotal bad volume rows: {cur.fetchone()[0]}")

cur.execute("SELECT COUNT(*) FROM ohlcv_daily WHERE typeof(close) NOT IN ('real', 'integer', 'null')")
print(f"Total bad close rows: {cur.fetchone()[0]}")

# How many total rows?
cur.execute("SELECT COUNT(*) FROM ohlcv_daily")
print(f"Total rows: {cur.fetchone()[0]}")

conn.close()
