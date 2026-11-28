from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import yfinance as yf

from config import IST_TIMEZONE
import database


def get_market_date(for_display=False):
    today = datetime.now(IST_TIMEZONE)
    if today.isoweekday() == 7:
        target_date = (today - timedelta(days=2)).strftime('%Y-%m-%d')
    elif today.isoweekday() == 6:
        target_date = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        target_date = today.strftime('%Y-%m-%d')

    if for_display:
        if not database.has_scanned_today(target_date):
            avail = database.get_available_scan_dates()
            if avail:
                return avail[0]

    return target_date


def render():
    st.markdown("### 📉 VPA Green + MA Squeeze")
    st.info("Finds stocks where VPA is Green (Minor, Mid, Major) and the 10/21/50 SMA are tightly clustered. Adjust the parameters below and run a custom scan.")

    if "vpa_squeeze_results" not in st.session_state:
        st.session_state.vpa_squeeze_results = []
    if "vpa_squeeze_weekly_results" not in st.session_state:
        st.session_state.vpa_squeeze_weekly_results = []
    if "vpa_squeeze_monthly_results" not in st.session_state:
        st.session_state.vpa_squeeze_monthly_results = []

    if not st.session_state.get('vpa_squeeze_results'):
        try:
            vpa_sq_d = database.get_all_latest_scan_dates().get("scanned_vpa_squeeze") or database.get_best_scan_date()
            if vpa_sq_d:
                st.session_state.vpa_squeeze_results = database.get_cached_vpa_squeeze(vpa_sq_d) or []
                st.session_state.vpa_squeeze_weekly_results = database.get_cached_vpa_squeeze_weekly(vpa_sq_d) or []
                st.session_state.vpa_squeeze_monthly_results = database.get_cached_vpa_squeeze_monthly(vpa_sq_d) or []
            if not st.session_state.get('vpa_squeeze_results'):
                today_str = get_market_date()
                st.session_state.vpa_squeeze_results = database.get_cached_vpa_squeeze(today_str) or []
                st.session_state.vpa_squeeze_weekly_results = database.get_cached_vpa_squeeze_weekly(today_str) or []
                st.session_state.vpa_squeeze_monthly_results = database.get_cached_vpa_squeeze_monthly(today_str) or []
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # PARAMETER PANEL — user-editable scan settings
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("⚙️ Scan Parameters (click to customise)", expanded=True):
        st.markdown(
            "<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:8px;'>"
            "Change these values to tighten or loosen the squeeze filter, then hit <b>🚀 Run VPA Squeeze Scan</b>."
            "</p>", unsafe_allow_html=True
        )
        p1, p2, p3 = st.columns(3)
        with p1:
            ma_gap_pct = st.number_input(
                "📐 Max MA Compression Gap (%)",
                min_value=1.0, max_value=20.0, value=6.0, step=0.5,
                help="Maximum % gap allowed between the 10, 21 and 50 SMAs. "
                     "Lower = tighter squeeze. Default 6%. Try 3–4% for very tight coils.",
                key="vpa_sq_ma_gap"
            )
        with p2:
            cmp_ext_pct = st.number_input(
                "🚀 Max CMP Extension above MA (%)",
                min_value=0.5, max_value=15.0, value=4.0, step=0.5,
                help="If CMP is more than this % above the highest SMA, the stock is considered "
                     "'already running' and is excluded. Default 4%.",
                key="vpa_sq_cmp_ext"
            )
        with p3:
            min_bars = st.number_input(
                "📊 Min Historical Bars Required",
                min_value=50, max_value=500, value=200, step=10,
                help="Minimum number of trading bars needed to run the scan. "
                     "200 = ~1 year of daily data (required for 200-SMA). "
                     "Set lower (e.g. 100) to include newer listings.",
                key="vpa_sq_min_bars"
            )

        st.markdown(
            f"<p style='color:#64748b; font-size:0.78rem;'>"
            f"Current settings → Compression ≤ <b>{ma_gap_pct}%</b> | "
            f"CMP extension ≤ <b>{cmp_ext_pct}%</b> above MA cluster | "
            f"Min <b>{int(min_bars)}</b> bars</p>",
            unsafe_allow_html=True
        )

    run_vpa_squeeze_btn = st.button("🚀 Run VPA Squeeze Scan", type="primary", use_container_width=True, key="run_vpa_squeeze_scan_btn")
    if run_vpa_squeeze_btn:
        st.session_state.vpa_squeeze_results = []
        with st.spinner(f"Running VPA Squeeze Scan (gap≤{ma_gap_pct}%, ext≤{cmp_ext_pct}%, bars≥{int(min_bars)})..."):
            try:
                from data_fetcher import get_all_nse_symbols
                from scanner import scan_vpa_ma_squeeze
                from local_cache_manager import bulk_get_cached_ohlcv

                raw_symbols = get_all_nse_symbols()
                symbols_to_scan = [s.strip().upper().replace('.NS', '') for s in raw_symbols if str(s).strip()]

                st.info("Step 1/2 — Fetching history from Master Cache...")
                prog = st.progress(0)
                status = st.empty()

                # Fetch all cached data at once
                bulk_cached = bulk_get_cached_ohlcv(symbols_to_scan, "1d")

                # Check for missing symbols or stubs (< min_bars bars)
                missing_symbols = [s for s in symbols_to_scan if s not in bulk_cached or len(bulk_cached[s]) < int(min_bars)]
                if missing_symbols:
                    chunk_size = 40
                    sym_chunks = [missing_symbols[i:i + chunk_size] for i in range(0, len(missing_symbols), chunk_size)]
                    downloaded_dict = {}
                    for c_idx, chunk in enumerate(sym_chunks):
                        prog.progress((c_idx + 1) / (len(sym_chunks) + 1))
                        status.text(f"Downloading historical data chunk {c_idx+1}/{len(sym_chunks)}...")
                        chunk_ns = [f"{s}.NS" for s in chunk]
                        try:
                            df_bulk = yf.download(tickers=chunk_ns, period="1y", interval="1d", progress=False, threads=False, timeout=20)
                            if df_bulk is not None and not df_bulk.empty:
                                for s in chunk:
                                    s_ns = f"{s}.NS"
                                    if isinstance(df_bulk.columns, pd.MultiIndex):
                                        all_t = df_bulk.columns.get_level_values(1).unique().tolist()
                                        matched = next((t for t in all_t if t.upper() == s_ns.upper()), None)
                                        if matched:
                                            s_df = df_bulk.xs(matched, axis=1, level=1).dropna(subset=['Close']).copy()
                                        else:
                                            continue
                                    else:
                                        s_df = df_bulk.dropna(subset=['Close']).copy()
                                    if not s_df.empty and len(s_df) >= 50:
                                        s_df = s_df.reset_index()
                                        s_df.rename(columns={s_df.columns[0]: 'Date'}, inplace=True)
                                        bulk_cached[s] = s_df
                                        downloaded_dict[s] = s_df
                        except Exception:
                            pass

                    if downloaded_dict:
                        try:
                            from local_cache_manager import save_to_cache_bulk
                            save_to_cache_bulk(downloaded_dict, "1d")
                        except Exception:
                            pass

                st.info("Step 2/2 — Calculating MA Squeeze (Instant)...")
                status.empty()
                prog.progress(1.0)

                results = []
                for sym, df_sym in bulk_cached.items():
                    res = scan_vpa_ma_squeeze(
                        sym, df_sym,
                        max_ma_gap_pct=float(ma_gap_pct),
                        max_cmp_ext_pct=float(cmp_ext_pct),
                        min_bars=int(min_bars)
                    )
                    if res is not None:
                        results.append(res)

                prog.empty()
                status.empty()
                st.session_state.vpa_squeeze_results = results

                # Save to database
                today_str = get_market_date()
                database.save_vpa_squeeze_only(today_str, results)

                st.success(f"✅ Scan complete! Found **{len(results)}** matching stocks (gap≤{ma_gap_pct}%, ext≤{cmp_ext_pct}%).")

            except Exception as e:
                st.error(f"Error running scan: {e}")

    st.markdown("---")
    r1, r2, r3 = st.columns([2, 2, 3])
    with r1:
        tf_vpa_sq = st.selectbox("📅 Timeframe", ["Daily", "Weekly", "Monthly"], key="tf_vpa_sq")
    with r2:
        sort_vpa_sq = st.selectbox(
            "Sort By",
            ["Compression Gap ↑ (Tightest)", "Chg % ↓", "RSI ↑", "CMP ↑"],
            key="vpa_sq_sort"
        )
    with r3:
        search_vpa_sq = st.text_input("🔍 Search Symbol", "", placeholder="e.g. INFY, RELIANCE...", key="vpa_sq_search").strip().upper()

    if tf_vpa_sq == "Daily":
        display_results = st.session_state.get('vpa_squeeze_results', [])
    elif tf_vpa_sq == "Weekly":
        display_results = st.session_state.get('vpa_squeeze_weekly_results', [])
    else:
        display_results = st.session_state.get('vpa_squeeze_monthly_results', [])

    # Apply symbol search + global max price
    if search_vpa_sq:
        display_results = [r for r in display_results if search_vpa_sq in str(r.get('symbol', '')).upper()]
    _max_cmp_sq = float(st.session_state.get("global_max_cmp", 0))
    if _max_cmp_sq > 0:
        display_results = [r for r in display_results if float(r.get('cmp') or 0.0) <= _max_cmp_sq]

    if display_results:
        df_res = pd.DataFrame(display_results)
        df_res['tv_url'] = df_res['symbol'].apply(
            lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}"
        )

        # Sort
        if sort_vpa_sq == "Compression Gap ↑ (Tightest)" and 'ma_gap_pct' in df_res.columns:
            df_res = df_res.sort_values('ma_gap_pct', ascending=True)
        elif sort_vpa_sq == "Chg % ↓" and 'day_change_pct' in df_res.columns:
            df_res = df_res.sort_values('day_change_pct', ascending=False)
        elif sort_vpa_sq == "RSI ↑" and 'rsi' in df_res.columns:
            df_res = df_res.sort_values('rsi', ascending=False)
        elif sort_vpa_sq == "CMP ↑" and 'cmp' in df_res.columns:
            df_res = df_res.sort_values('cmp', ascending=False)

        st.markdown(f"<p style='color:#94a3b8; font-size:0.85rem;'>Showing <b>{len(df_res)}</b> matching stocks</p>", unsafe_allow_html=True)

        display_cols = ['tv_url', 'company_name', 'cmp', 'day_change_pct', 'sma10', 'sma21', 'sma50', 'ma_gap_pct', 'dist_to_200_pct', 'rsi', 'cci']
        # Keep only columns that actually exist (handles older cached results without the new field)
        display_cols = [c for c in display_cols if c in df_res.columns]
        st.dataframe(
            df_res[display_cols],
            use_container_width=True,
            column_config={
                "tv_url": st.column_config.LinkColumn(
                    "Stock (TradingView)",
                    display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)",
                    width="medium"
                ),
                "company_name": st.column_config.TextColumn("Company", width="medium"),
                "cmp": st.column_config.NumberColumn("CMP (₹)", format="₹ %.2f", width="small"),
                "day_change_pct": st.column_config.NumberColumn("Chg %", format="%.2f%%", width="small"),
                "sma10": st.column_config.NumberColumn("10 SMA", format="₹ %.2f", width="small"),
                "sma21": st.column_config.NumberColumn("21 SMA", format="₹ %.2f", width="small"),
                "sma50": st.column_config.NumberColumn("50 SMA", format="₹ %.2f", width="small"),
                "ma_gap_pct": st.column_config.ProgressColumn(
                    "Compression Gap %",
                    min_value=0, max_value=10, format="%.2f%%",
                    help="Max distance % between the 10, 21, and 50 SMAs. Lower = tighter coil.",
                    width="small"
                ),
                "dist_to_200_pct": st.column_config.NumberColumn("Dist to 200 SMA %", format="%.2f%%", width="small"),
                "rsi": st.column_config.NumberColumn("RSI", format="%.1f", width="small"),
                "cci": st.column_config.NumberColumn("CCI", format="%.1f", width="small"),
            },
            hide_index=True,
            height=min(500, 45 + len(df_res) * 38)
        )
    else:
        st.info("No stocks found matching the VPA Squeeze criteria. Adjust parameters above and run the scanner.")
