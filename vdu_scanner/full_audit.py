"""
Full audit script: checks DB state, reload logic, and save paths for every tab.
"""
import database
import sys

sys.stdout.reconfigure(encoding='utf-8')

conn = database.get_connection()
c = conn.cursor()

print("=" * 60)
print("1. ROW COUNTS & LATEST DATE PER TABLE")
print("=" * 60)
scan_tables = [
    "scan_logs", "scanned_breakouts", "scanned_gapups", "scanned_trend_setups",
    "scanned_wt_cross", "scanned_vcs", "scanned_vpa", "scanned_vpa_squeeze",
    "scanned_volume_profile", "scanned_ema_support", "scanned_stage2",
    "scanned_stage_analysis", "scanned_support_rsi", "scanned_monthly_momentum",
    "scanned_weekly_momentum", "scanned_zanger", "scanned_vcp_minervini",
    "scanned_rsi_wt_combo"
]
for tbl in scan_tables:
    try:
        c.execute(f"SELECT COUNT(*), MAX(scan_date) FROM {tbl};")
        cnt, latest = c.fetchone()
        status = "OK " if cnt > 0 else "NO "
        print(f"  {status} {tbl:35} rows={cnt:5}  latest={latest}")
    except Exception as e:
        print(f"  ERR {tbl}: ERROR - {e}")

print()
print("=" * 60)
print("2. get_available_scan_dates() result")
print("=" * 60)
print(" ", database.get_available_scan_dates())

print()
print("=" * 60)
print("3. get_latest_date_for_table() for each tab")
print("=" * 60)
tab_table_map = {
    "VPA Tab":             "scanned_vpa",
    "VPA Squeeze Tab":     "scanned_vpa_squeeze",
    "WaveTrend Tab":       "scanned_wt_cross",
    "VCS Tab":             "scanned_vcs",
    "Stage-2 Tab":         "scanned_stage2",
    "Stage Analysis Tab":  "scanned_stage_analysis",
    "Volume Profile Tab":  "scanned_volume_profile",
    "BB/9-21 EMA Tab":     "scanned_ema_support",
    "Support RSI Tab":     "scanned_support_rsi",
    "Monthly Tab":         "scanned_monthly_momentum",
    "Weekly Tab":          "scanned_weekly_momentum",
    "Dan Zanger Tab":      "scanned_zanger",
    "VCP+Minervini Tab":   "scanned_vcp_minervini",
    "VDCU Results Tab":    "scanned_breakouts",
    "Minervini/MA/SMA Tab":"scanned_trend_setups",
}
for tab, tbl in tab_table_map.items():
    d = database.get_latest_date_for_table(tbl)
    status = "OK " if d else "NO DATA"
    print(f"  {status}  {tab:30} -> {d or 'empty'}")

print()
print("=" * 60)
print("4. SPOT CHECK: Load actual data for tables that have dates")
print("=" * 60)
checks = [
    ("scanned_vpa", database.get_cached_vpa, "VPA"),
    ("scanned_vpa_squeeze", database.get_cached_vpa_squeeze, "VPA Squeeze"),
    ("scanned_wt_cross", database.get_cached_wt_cross, "WaveTrend"),
    ("scanned_ema_support", database.get_cached_ema_support, "BB Squeeze"),
    ("scanned_stage2", database.get_cached_stage2, "Stage-2"),
    ("scanned_stage_analysis", database.get_cached_stage_analysis, "Stage Analysis"),
    ("scanned_volume_profile", database.get_cached_volume_profile, "Vol Profile"),
    ("scanned_support_rsi", database.get_cached_support_rsi, "Support RSI"),
    ("scanned_weekly_momentum", database.get_cached_weekly_momentum, "Weekly"),
]
for tbl, fn, label in checks:
    d = database.get_latest_date_for_table(tbl)
    if d:
        data = fn(d)
        status = f"OK  {len(data)} records" if data else "WARN getter returned empty"
    else:
        status = "NO  no date in table"
    print(f"  {label:25} [{d or 'N/A'}] -> {status}")

conn.close()
print()
print("AUDIT COMPLETE.")
