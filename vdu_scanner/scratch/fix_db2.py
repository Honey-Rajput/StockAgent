import sqlite3

def add_cols():
    conn = sqlite3.connect('c:/D_Drive/Stock/Codewithgoogle/StockswithDryVolume/vdu_scanner/vdu_scanner.db')
    cur = conn.cursor()
    tables = ['scanned_vpa_squeeze', 'scanned_vpa_squeeze_weekly', 'scanned_vpa_squeeze_monthly']
    for t in tables:
        for col in ['rsi', 'cci']:
            try:
                cur.execute(f"ALTER TABLE {t} ADD COLUMN {col} DOUBLE PRECISION")
                print(f"Added {col} to {t}")
            except Exception as e:
                print(f"{t}.{col}: {e}")
    conn.commit()
    conn.close()

if __name__ == '__main__':
    add_cols()
