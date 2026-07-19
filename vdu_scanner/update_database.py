import sys

with open('c:\\D_Drive\\Stock\\Codewithgoogle\\StockswithDryVolume\\vdu_scanner\\database.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add table to init_db
table_sql = """
        \"\"\"
        CREATE TABLE IF NOT EXISTS scanned_near_30sma (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(20) NOT NULL,
            company_name VARCHAR(200),
            cmp DOUBLE PRECISION,
            day_change_pct DOUBLE PRECISION,
            volume BIGINT,
            sma30 DOUBLE PRECISION,
            dist_pct DOUBLE PRECISION,
            scan_date DATE NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, scan_date)
        );
        \"\"\",
"""
if "CREATE TABLE IF NOT EXISTS scanned_near_30sma" not in content:
    content = content.replace('queries = [', 'queries = [\n' + table_sql)

# 2. Update save_scan_results signature
if "def save_scan_results(date_str: str, breakouts: list[dict]," in content and "near_30sma_list: list[dict] = None" not in content:
    content = content.replace(
        "def save_scan_results(date_str: str, breakouts: list[dict], squeezes: list[dict], gapups: list[dict], trend_setups: list[dict], wt_cross: list[dict], total_scanned: int, vcs_results: list[dict] = None, vpa_results: list[dict] = None, vpa_squeeze_results: list[dict] = None, structural_vcp_results: list[dict] = None, stage2_results: list[dict] = None, volume_profile_results: list[dict] = None, ema_support_results: list[dict] = None, stage_analysis_results: list[dict] = None) -> bool:",
        "def save_scan_results(date_str: str, breakouts: list[dict], squeezes: list[dict], gapups: list[dict], trend_setups: list[dict], wt_cross: list[dict], total_scanned: int, vcs_results: list[dict] = None, vpa_results: list[dict] = None, vpa_squeeze_results: list[dict] = None, structural_vcp_results: list[dict] = None, stage2_results: list[dict] = None, volume_profile_results: list[dict] = None, ema_support_results: list[dict] = None, stage_analysis_results: list[dict] = None, near_30sma_list: list[dict] = None) -> bool:"
    )

# 3. Add deletions
if "DELETE FROM scanned_near_30sma" not in content:
    content = content.replace(
        "cur.execute(\"DELETE FROM scan_logs WHERE scan_date = %s;\", (date_str,))",
        "cur.execute(\"DELETE FROM scanned_near_30sma WHERE scan_date = %s;\", (date_str,))\n        cur.execute(\"DELETE FROM scan_logs WHERE scan_date = %s;\", (date_str,))"
    )

# 4. Add insertions
insert_code = """
        # Insert Near 30 SMA
        if near_30sma_list:
            near_30sma_query = \"\"\"
            INSERT INTO scanned_near_30sma (symbol, company_name, cmp, day_change_pct, volume, sma30, dist_pct, scan_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            \"\"\"
            for r in near_30sma_list:
                cur.execute(near_30sma_query, (
                    str(r['symbol']),
                    str(r.get('company_name', '')),
                    float(r['cmp']),
                    float(r['day_change_pct']),
                    int(r.get('volume', 0)),
                    float(r['sma30']),
                    float(r['dist_pct']),
                    date_str
                ))
"""
if "Insert Near 30 SMA" not in content:
    content = content.replace(
        "# 4. Insert execution log",
        insert_code + "\n        # 4. Insert execution log"
    )

# 5. Add getter
getter_code = """
def get_cached_near_30sma(date_str: str) -> list[dict]:
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM scanned_near_30sma WHERE scan_date = %s ORDER BY dist_pct ASC", (date_str,))
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f"Error fetching near 30 SMA: {e}")
        return []
    finally:
        if conn: conn.close()
"""
if "def get_cached_near_30sma" not in content:
    content += "\n" + getter_code + "\n"

with open('c:\\D_Drive\\Stock\\Codewithgoogle\\StockswithDryVolume\\vdu_scanner\\database.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated database.py")
