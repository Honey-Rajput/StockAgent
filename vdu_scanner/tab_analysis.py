"""
Database has data on 2026-07-17 for:
  - scanned_ema_support       -> 354 rows
  - scanned_stage2           -> 119 rows
  - scanned_stage_analysis   -> 515 rows
  - scanned_support_rsi      -> 10 rows
  - scanned_vcs              -> 3 rows
  - scanned_volume_profile   -> 698 rows
  - scanned_vpa              -> 684 rows
  - scanned_vpa_squeeze      -> 15 rows
  - scanned_weekly_momentum  -> 14 rows
  - scanned_wt_cross         -> 16 rows

EMPTY (no data ever saved):
  - scan_logs                -> 0 (main scanner not run)
  - scanned_breakouts        -> 0
  - scanned_gapups           -> 0
  - scanned_monthly_momentum -> 0
  - scanned_trend_setups     -> 0  (Minervini/20-50 SMA/MA Cross/65 SMA live here!)
  - scanned_vcp_minervini    -> 0
  - scanned_zanger           -> 0
  - scanned_rsi_wt_combo     -> 0
"""

# Tab -> DB session_state variable -> DB function
TAB_STATUS = {
    # TABS THAT WILL LOAD ON REFRESH (data exists in DB):
    "VPA Tab":              ("vpa_results",             "get_cached_vpa",            "2026-07-17", "✅ WILL LOAD"),
    "VPA Squeeze Tab":      ("vpa_squeeze_results",     "get_cached_vpa_squeeze",    "2026-07-17", "✅ WILL LOAD"),
    "WaveTrend Tab":        ("wt_results",              "get_cached_wt_cross",       "2026-07-17", "✅ WILL LOAD"),
    "VCS Tab":              ("vcs_results",             "get_cached_vcs",            "2026-07-17", "✅ WILL LOAD"),
    "Stage-2 Breakout Tab": ("stage2_results",          "get_cached_stage2",         "2026-07-17", "✅ WILL LOAD"),
    "Stage Analysis Tab":   ("stage_analysis_results",  "get_cached_stage_analysis", "2026-07-17", "✅ WILL LOAD"),
    "Volume Profile Tab":   ("vp_results",              "get_cached_volume_profile", "2026-07-17", "✅ WILL LOAD"),
    "BB Squeeze Tab":       ("ema_support_results",      "get_cached_ema_support",     "2026-07-17", "✅ WILL LOAD"),
    "9/21 EMA Tab":         ("support_rsi_results",     "get_cached_support_rsi",    "2026-07-17", "✅ WILL LOAD"),
    "Weekly Tab":           ("weekly_momentum_results", "get_cached_weekly_momentum","2026-07-17", "✅ WILL LOAD"),

    # TABS THAT WON'T LOAD ON REFRESH (NO data in DB):
    "VDCU Results Tab":     ("scan_results",            "get_cached_breakouts",      "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "Detail Tab":           ("scan_results",            "get_cached_breakouts",      "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "Minervini Tab":        ("minervini_results",       "get_cached_trend_setups",   "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "20&50 SMA Tab":        ("above_ma_results",        "get_cached_trend_setups",   "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "MA Cross Tab":         ("crossover_ma_results",    "get_cached_trend_setups",   "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "65 SMA Tab":           ("support_ma_results",      "get_cached_trend_setups",   "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "VCP+Minervini Tab":    ("vcp_minervini_results",   "get_cached_vcp_minervini",  "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "Dan Zanger Tab":       ("zanger_results",          "get_cached_zanger",         "EMPTY",       "❌ NEEDS RUN SCANNER"),
    "Monthly Tab":          ("monthly_momentum_results","get_cached_monthly_momentum","EMPTY",      "❌ NEEDS RUN SCANNER"),
    "Historical Tab":       ("N/A - uses dropdown",     "get_available_scan_dates",  "EMPTY scan_logs","❌ scan_logs is empty"),
}

for tab, (session_var, db_func, date, status) in TAB_STATUS.items():
    print(f"{status:30} | {tab:30} | session_state.{session_var}")
