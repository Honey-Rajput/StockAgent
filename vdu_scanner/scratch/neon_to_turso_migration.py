import psycopg2
from psycopg2.extras import RealDictCursor
import libsql_client
import os
from dotenv import load_dotenv

load_dotenv()

# Neon DB Connection
neon_url = os.getenv("NEON_DATABASE_URL", "postgresql://user:pass@host/db")
neon_conn = psycopg2.connect(neon_url)

# Turso DB Connection
turso_url = os.getenv("TURSO_DATABASE_URL", "https://...")
turso_token = os.getenv("TURSO_AUTH_TOKEN", "eyJ...")

os.environ["TURSO_DATABASE_URL"] = turso_url
os.environ["TURSO_AUTH_TOKEN"] = turso_token

turso_conn = libsql_client.create_client_sync(turso_url, auth_token=turso_token)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import database
database.init_db()

tables = [
    "ai_chart_patterns",
    "scanned_breakouts",
    "scanned_squeezes",
    "scanned_gapups",
    "scanned_trend_setups",
    "scanned_wt_cross",
    "scan_logs",
    "scanned_monthly_momentum",
    "scanned_weekly_momentum",
    "scanned_vcs",
    "scanned_vpa",
    "scanned_stage2",
    "scanned_volume_profile",
    "scanned_stage_analysis",
    "scanned_support_rsi",
    "scanned_rsi_wt_combo",
    "scanned_zanger",
    "scanned_bb_squeeze",
    "scanned_vcp_minervini",
    "ohlcv_daily"
]

print("Starting migration from Neon to Turso...")

with neon_conn.cursor(cursor_factory=RealDictCursor) as cursor:
    for table in tables:
        try:
            print(f"Fetching data for {table} from Neon...")
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            
            if not rows:
                print(f"  -> Table {table} is empty. Skipping.")
                continue
                
            print(f"  -> Found {len(rows)} rows. Inserting into Turso...")
            
            cols = list(rows[0].keys())
            try:
                turso_cols_res = turso_conn.execute(f"PRAGMA table_info({table})")
                turso_valid_cols = [r[1] for r in turso_cols_res.rows]
            except Exception:
                turso_valid_cols = cols

            # Filter columns to only those that exist in Turso
            valid_cols = [c for c in cols if c in turso_valid_cols]
            
            if not valid_cols:
                print(f"  -> No matching columns found for {table}.")
                continue
                
            col_names = ", ".join(valid_cols)
            placeholders = ", ".join(["?" for _ in valid_cols])
            query = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
            
            stmts = []
            for row in rows:
                # Convert list values or dict values to string (SQLite doesn't support arrays natively)
                args = []
                import datetime
                import math
                for c in valid_cols:
                    val = row[c]
                    if isinstance(val, (datetime.date, datetime.datetime)):
                        val = val.isoformat()
                    elif isinstance(val, list) or isinstance(val, dict):
                        val = str(val)
                    elif isinstance(val, float):
                        if math.isnan(val) or math.isinf(val):
                            val = None
                    args.append(val)
                stmts.append(libsql_client.Statement(query, args))
                
            # Batch insert in chunks of 500
            for i in range(0, len(stmts), 500):
                try:
                    turso_conn.batch(stmts[i:i+500])
                except Exception as e:
                    if "UNIQUE constraint failed" in str(e):
                        pass # Ignore duplicates for scan_logs
                    else:
                        print(f"  -> Batch Error: {e}")
                
            print(f"  -> Successfully migrated {len(rows)} rows for {table}.")
            
        except Exception as e:
            if "does not exist" in str(e):
                print(f"  -> Table {table} does not exist in Neon. Skipping.")
            else:
                print(f"  -> Error migrating table {table}: {e}")

neon_conn.close()
turso_conn.close()
print("Migration complete!")
