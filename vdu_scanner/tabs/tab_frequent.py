import streamlit as st
import pandas as pd
import database
from collections import defaultdict


def collect_all_strategy_alerts():
    """
    Aggregates active stocks across ALL 22 scanner strategies in the application.
    Checks st.session_state first, falls back to background scan status,
    and falls back to database cache so alerts are fully populated even on fresh launch.
    """
    live_alerts = defaultdict(lambda: {
        'symbol': '',
        'company_name': '',
        'cmp': 0.0,
        'strategies': set(),
        'max_score': 0.0,
        'rsi': 0.0,
        'cci': 0.0
    })

    # Pre-fetch latest scan dates if database fallback is needed
    best_date = None
    all_latest = {}
    try:
        all_latest = database.get_all_latest_scan_dates()
        best_date = database.get_best_scan_date()
    except Exception:
        pass

    bg_status = st.session_state.get('ALL_TAB_SCAN_STATUS', {})

    def _get_strat_data(ss_keys, db_fn=None, db_arg=None, table_name=None):
        if isinstance(ss_keys, str):
            ss_keys = [ss_keys]
        for k in ss_keys:
            d = st.session_state.get(k)
            if d:
                return d
            if bg_status.get(k):
                return bg_status[k]
        if db_fn:
            try:
                target_date = all_latest.get(table_name) or best_date
                if target_date:
                    if db_arg is not None:
                        return db_fn(target_date, db_arg) or []
                    return db_fn(target_date) or []
            except Exception:
                pass
        return []

    def _ingest(data_list, strat_name, default_score=0.0):
        if not data_list:
            return
        for r in data_list:
            if not isinstance(r, dict):
                continue
            sym = str(r.get('symbol', '')).strip().upper().replace('.NS', '')
            if not sym:
                continue
            item = live_alerts[sym]
            item['symbol'] = sym
            if not item['company_name'] and r.get('company_name'):
                item['company_name'] = str(r['company_name']).strip()
            cmp_val = float(r.get('cmp') or r.get('price') or r.get('close') or 0.0)
            if cmp_val > 0 and not item['cmp']:
                item['cmp'] = cmp_val
            item['strategies'].add(strat_name)

            score_val = float(r.get('signal_strength') or r.get('trend_score') or r.get('score') or r.get('vcs_score') or default_score or 0.0)
            if score_val > item['max_score']:
                item['max_score'] = score_val

            if r.get('rsi') and not item['rsi']:
                try: item['rsi'] = float(r['rsi'])
                except Exception: pass
            if r.get('cci') and not item['cci']:
                try: item['cci'] = float(r['cci'])
                except Exception: pass

    # 1. VDU Breakout
    _ingest(_get_strat_data('scan_results', database.get_cached_breakouts, table_name='scan_results'), 'VDU Breakout')

    # 2. Minervini Stage 2 Setups
    _ingest(_get_strat_data('minervini_results', database.get_cached_trend_setups, db_arg='minervini', table_name='scanned_trend_setups'), 'Minervini Stage 2')

    # 3. 20 & 50 SMA Support Setups
    _ingest(_get_strat_data('above_ma_results', database.get_cached_trend_setups, db_arg='above_ma', table_name='scanned_trend_setups'), '20&50 SMA')

    # 4. 65 SMA Support Setups
    _ingest(_get_strat_data('support_ma_results', database.get_cached_trend_setups, db_arg='support_ma', table_name='scanned_trend_setups'), '65 SMA Support')

    # 5. 50/200 MA Golden Crossover
    _ingest(_get_strat_data('crossover_ma_results', database.get_cached_trend_setups, db_arg='crossover_ma', table_name='scanned_trend_setups'), '50/200 MA Cross')

    # 6. WaveTrend Oversold Buy Cross
    _ingest(_get_strat_data('wt_results', database.get_cached_wt_cross, table_name='scanned_wt_cross'), 'WaveTrend Buy')

    # 7. Gap-Up Momentum Setups
    _ingest(_get_strat_data('gapup_results', database.get_cached_gapups, table_name='scanned_gapups'), 'Gap-Up')

    # 8. 9/21 EMA Dynamic Support
    _ingest(_get_strat_data('ema_support_results', database.get_cached_ema_support, table_name='scanned_ema_support'), '9/21 EMA Support')

    # 9. Volume Profile Buy Zone
    vp_data = _get_strat_data('vp_results', database.get_cached_volume_profile, table_name='scanned_volume_profile')
    vp_buys = []
    for r in vp_data:
        if isinstance(r, dict):
            daily_zone = r.get('daily')
            if isinstance(daily_zone, dict) and 'Can Buy' in str(daily_zone.get('zone', '')):
                vp_buys.append(r)
            elif 'Can Buy' in str(r.get('zone', '')):
                vp_buys.append(r)
    _ingest(vp_buys, 'VP Buy Zone')

    # 10. Dan Zanger VCS
    _ingest(_get_strat_data(['vcs_results', 'dan_zanger_results', 'zanger_results'], database.get_cached_vcs, table_name='scanned_vcs'), 'Dan Zanger VCS')

    # 11. VCP Minervini
    _ingest(_get_strat_data(['vcp_minervini_results', 'structural_vcp_results'], database.get_cached_vcp_minervini, table_name='scanned_vcp_minervini'), 'VCP Minervini')

    # 12. Stage 2 Breakout
    _ingest(_get_strat_data('stage2_results', database.get_cached_stage2, table_name='scanned_stage2'), 'Stage 2 Breakout')

    # 13. VPA Matrix
    _ingest(_get_strat_data('vpa_results', database.get_cached_vpa, table_name='scanned_vpa'), 'VPA Matrix')

    # 14. Flat Base Squeeze
    _ingest(_get_strat_data('flat_base_results', database.get_cached_flat_base, table_name='scanned_flat_base'), 'Flat Base Squeeze')

    # 15. VPA Squeeze
    _ingest(_get_strat_data('vpa_squeeze_results', database.get_cached_vpa_squeeze, table_name='scanned_vpa_squeeze'), 'VPA Squeeze')

    # 16. Near 30 SMA
    _ingest(_get_strat_data('near_30sma_results', database.get_cached_near_30sma, table_name='scanned_near_30sma'), 'Near 30 SMA')

    # 17. HH/HL Uptrend (Dow Theory Market Structure)
    _ingest(_get_strat_data('hh_hl_results', database.get_cached_hh_hl, table_name='scanned_hh_hl'), 'HH/HL Uptrend')

    # 18. Stage Analysis (Stan Weinstein)
    _ingest(_get_strat_data('stage_analysis_results', database.get_cached_stage_analysis, table_name='scanned_stage_analysis'), 'Stage Analysis')

    # 19. Support RSI Oversold
    _ingest(_get_strat_data(['support_rsi_results', 'rsi_oversold_results'], database.get_cached_support_rsi, table_name='scanned_support_rsi'), 'Support RSI')

    # 20. Monthly Momentum
    _ingest(_get_strat_data('monthly_momentum_results', database.get_cached_monthly_momentum, table_name='scanned_monthly_momentum'), 'Monthly Momentum')

    # 21. Weekly Momentum
    _ingest(_get_strat_data('weekly_momentum_results', database.get_cached_weekly_momentum, table_name='scanned_weekly_momentum'), 'Weekly Momentum')

    # 22. Breadth Ledger Fresh High Leaders
    try:
        import breadth_ledger
        breadth_data = breadth_ledger.fetch_breadth_data()
        for g in breadth_data.get('groups', []):
            if g.get('flag', 0) >= 2:
                for s in g.get('stocks', [])[:10]:
                    sym = str(s).strip().upper().replace('.NS', '')
                    if sym:
                        live_alerts[sym]['symbol'] = sym
                        live_alerts[sym]['strategies'].add('Breadth Leader')
    except Exception:
        pass

    return live_alerts


def render():
    st.markdown("### 🚨 High-Conviction Confluence & Consistent Alerts")
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.92rem; margin-top:-6px;'>"
        "Aggregates stocks triggering across <b>all 22 scanner strategies</b> in real-time, highlighting institutional "
        "confluences where multiple independent edges align simultaneously on the same stock."
        "</p>",
        unsafe_allow_html=True
    )

    # ── Section 1: Live Active Session Confluence Alerts ──────────────────────
    with st.spinner("Aggregating live signals across all 22 scanner strategies..."):
        live_alerts = collect_all_strategy_alerts()

    tv_col_config = {
        "Symbol": st.column_config.LinkColumn("Stock (TradingView)", display_text=r"https://in\.tradingview\.com/chart/\?symbol=NSE:(.*)"),
        "Company": st.column_config.TextColumn("Company Name", width="medium"),
        "CMP": st.column_config.NumberColumn("CMP (₹)", format="%.2f"),
        "Confluence": st.column_config.TextColumn("Confluence", width="small"),
        "Triggered Strategies": st.column_config.TextColumn("Triggered Strategies", width="large"),
        "Score": st.column_config.NumberColumn("Top Score", format="%.1f"),
    }

    if live_alerts:
        # Sort by confluence (strategies count desc), then score desc
        live_list = sorted(live_alerts.values(), key=lambda x: (len(x['strategies']), x['max_score']), reverse=True)

        confluence_2plus = [x for x in live_list if len(x['strategies']) >= 2]
        confluence_3plus = [x for x in live_list if len(x['strategies']) >= 3]
        max_confluence = max(len(x['strategies']) for x in live_list) if live_list else 0
        top_stock = live_list[0]['symbol'] if live_list else "—"

        st.markdown(f"#### ⚡ Today's Live Multi-Strategy Alerts ({len(live_list)} Stocks Active)")
        st.caption("Stocks actively meeting scanner criteria across all 22 strategies today. High-confluence stocks (matching 2+ strategies) represent the strongest institutional setups.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Active Alerts", f"{len(live_list):,} stocks")
        m2.metric("🔥 High Confluence (2+)", f"{len(confluence_2plus)} stocks")
        m3.metric("⚡ Strong Confluence (3+)", f"{len(confluence_3plus)} stocks")
        m4.metric("Top Confluence Peak", f"{max_confluence} Strategies", f"Leader: {top_stock}")

        # Controls & Filters
        f_col1, f_col2, f_col3 = st.columns([2.5, 2.5, 3])
        with f_col1:
            search_query = st.text_input("🔍 Search Symbol / Company:", "", key="alerts_search_sym").strip().upper()
        with f_col2:
            min_conf_sel = st.selectbox(
                "🎯 Minimum Confluence Level:",
                ["All Active Alerts (1+)", "🔥 High Confluence (2+ Strategies)", "⚡ Strong Confluence (3+ Strategies)", "🎯 Elite Confluence (4+ Strategies)", "🌟 Maximum Confluence (5+ Strategies)"],
                index=1,
                key="alerts_min_conf_sel"
            )
        with f_col3:
            all_strats_available = sorted(list({s for item in live_list for s in item['strategies']}))
            strat_filter = st.multiselect("Filter by Strategy:", all_strats_available, default=[], key="alerts_strat_filter")

        # Filter live list
        min_count = 1
        if "2+" in min_conf_sel: min_count = 2
        elif "3+" in min_conf_sel: min_count = 3
        elif "4+" in min_conf_sel: min_count = 4
        elif "5+" in min_conf_sel: min_count = 5

        filtered_list = []
        for x in live_list:
            if len(x['strategies']) < min_count:
                continue
            if search_query:
                sym_match = search_query in x['symbol'].upper()
                comp_match = search_query in str(x.get('company_name', '')).upper()
                if not (sym_match or comp_match):
                    continue
            if strat_filter:
                if not any(s in x['strategies'] for s in strat_filter):
                    continue
            _max_cmp_al = float(st.session_state.get("global_max_cmp", 0))
            if _max_cmp_al > 0 and float(x.get('cmp') or 0.0) > _max_cmp_al:
                continue
            filtered_list.append(x)

        st.caption(f"Showing **{len(filtered_list)}** stocks matching filter criteria:")

        live_rows = []
        for rank, x in enumerate(filtered_list, 1):
            strat_count = len(x['strategies'])
            badge = f"🌟 {strat_count} Strategies" if strat_count >= 5 else (
                f"🎯 {strat_count} Strategies" if strat_count >= 4 else (
                    f"⚡ {strat_count} Strategies" if strat_count >= 3 else (
                        f"🔥 {strat_count} Strategies" if strat_count >= 2 else "1 Strategy"
                    )
                )
            )
            live_rows.append({
                'Rank': rank,
                'Symbol': f"https://in.tradingview.com/chart/?symbol=NSE:{x['symbol']}",
                'Company': x.get('company_name', ''),
                'CMP': x['cmp'],
                'Confluence': badge,
                'Triggered Strategies': " | ".join(sorted(x['strategies'])),
                'Score': x['max_score']
            })

        df_live = pd.DataFrame(live_rows)
        if not df_live.empty:
            st.dataframe(
                df_live,
                use_container_width=True,
                hide_index=True,
                column_config=tv_col_config,
                height=min(500, 45 + len(df_live) * 35)
            )
        else:
            st.info("No stocks match the selected confluence criteria.")

        st.markdown("---")

    # ── Section 2: Historical Multi-Day Database Alerts ───────────────────────
    st.subheader("🏆 Multi-Day Frequent Alerts (Database Archive)")
    st.caption("Tracks stocks repeatedly flagged across completed daily scanner runs across all strategies.")

    col1, col2 = st.columns([2, 5])
    with col1:
        lookback_days = st.slider("Lookback Window (Days)", min_value=3, max_value=30, value=15, step=1,
                                 help="Number of recent distinct scan dates to search across.")

    with st.spinner(f"Aggregating alerts for the last {lookback_days} scan days..."):
        frequent_stocks = database.get_frequent_stocks(days_lookback=lookback_days)

    if not frequent_stocks:
        st.info(f"No multi-day repeated stocks found in the database for the last {lookback_days} days. As daily scans complete, frequent stocks will appear here.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(frequent_stocks)

    # Process Strategy tags into nice labels
    def format_strategies(strategies_str):
        if not strategies_str:
            return ""
        strats = set([s.strip() for s in strategies_str.split(',') if s.strip()])
        return " | ".join(sorted(strats))

    df['strategies'] = df['strategies'].apply(format_strategies)

    # Split into New Today and Repeated
    new_today_df = df[df.get('is_new_today', pd.Series([False]*len(df))) == True]
    repeated_df = df[df.get('is_new_today', pd.Series([True]*len(df))) != True]

    def apply_tv_links(disp_df):
        disp_df['Symbol'] = disp_df['Symbol'].apply(lambda x: f"https://in.tradingview.com/chart/?symbol=NSE:{str(x).replace('.NS', '')}")
        return disp_df

    # --- NEW TODAY SECTION ---
    if len(new_today_df) > 0:
        st.subheader(f"🆕 New in Archive — First Time Flagged ({len(new_today_df)} stocks)")
        st.caption("These stocks just appeared in the historical archive for the first time.")

        new_display = pd.DataFrame()
        new_display['Symbol'] = new_today_df['symbol'].astype(str)
        new_display['Score'] = pd.to_numeric(new_today_df['max_score'], errors='coerce').fillna(0).astype(float).round(1)
        new_display['RSI'] = pd.to_numeric(new_today_df['rsi'], errors='coerce').fillna(0).astype(float).round(2)
        new_display['CCI'] = pd.to_numeric(new_today_df['cci'], errors='coerce').fillna(0).astype(float).round(2)
        new_display['Strategies'] = new_today_df['strategies'].astype(str)
        new_display['Total Hits'] = pd.to_numeric(new_today_df['total_appearances'], errors='coerce').fillna(0).astype(int)

        new_display = apply_tv_links(new_display)

        st.dataframe(
            new_display,
            use_container_width=True,
            hide_index=True,
            column_config=tv_col_config,
            height=min(400, 35 + len(new_display) * 35)
        )

        st.write("---")

    # --- REPEATED SECTION ---
    if len(repeated_df) > 0:
        repeated_df = repeated_df.sort_values(by=['days_appeared', 'total_appearances'], ascending=[False, False])

        rep_display = pd.DataFrame()
        rep_display['Symbol'] = repeated_df['symbol'].astype(str)
        rep_display['Consistency %'] = (pd.to_numeric(repeated_df['days_appeared'], errors='coerce').fillna(0) / lookback_days * 100).round(1).astype(str) + "%"
        rep_display['Days Appeared'] = pd.to_numeric(repeated_df['days_appeared'], errors='coerce').fillna(0).astype(int)
        rep_display['Score'] = pd.to_numeric(repeated_df['max_score'], errors='coerce').fillna(0).astype(float).round(1)
        rep_display['First Alert'] = repeated_df['first_seen_date'].astype(str)
        rep_display['Most Recent'] = repeated_df['last_seen_date'].astype(str)
        rep_display['Triggered Strategies'] = repeated_df['strategies'].astype(str)

        rep_display = apply_tv_links(rep_display)

        st.subheader(f"🔁 Repeated Multi-Day Alerts (Last {lookback_days} Scans) — {len(repeated_df)} stocks")

        st.dataframe(
            rep_display,
            use_container_width=True,
            hide_index=True,
            column_config=tv_col_config,
            height=600
        )

    # CSV Download — full data
    full_display = pd.DataFrame()
    full_display['Symbol'] = df['symbol'].astype(str)
    full_display['New Today'] = df.get('is_new_today', False).apply(lambda x: '🆕 Yes' if x else 'No').astype(str)
    full_display['Consistency %'] = (pd.to_numeric(df['days_appeared'], errors='coerce').fillna(0) / lookback_days * 100).round(1).astype(str) + "%"
    full_display['Days Appeared'] = pd.to_numeric(df['days_appeared'], errors='coerce').fillna(0).astype(int)
    full_display['Score'] = pd.to_numeric(df['max_score'], errors='coerce').fillna(0).astype(float).round(1)
    full_display['First Alert'] = df['first_seen_date'].astype(str)
    full_display['Most Recent'] = df['last_seen_date'].astype(str)
    full_display['Triggered Strategies'] = df['strategies'].astype(str)

    csv_data = full_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download All Alerts (CSV)",
        data=csv_data,
        file_name="all_alerts.csv",
        mime="text/csv",
        key="dl_all_alerts_csv"
    )

    st.write("---")
    st.caption(f"Showing {len(new_today_df)} new today + {len(repeated_df)} repeated = {len(df)} total stocks across the last {lookback_days} scans.")
